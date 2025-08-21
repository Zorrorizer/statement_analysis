# mrag_app.py
# MVP dostosowane: upload exactly 1 PDF (3-month history), używa tych samych ustawień co rag w /var/www/apintelligence/app.
# Frontend będzie w /var/www/apintelligence/bank (PHP/HTML/JS) – tu backend API w Pythonie.

import os
import re
import json
import uuid
import sqlite3
from typing import List, Dict, Any, Tuple

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from openai import OpenAI
from pdfminer.layout import LTTextContainer, LTTextLineHorizontal
import sqlparse
# -----------------------------
# Stałe ustawienia (z Twojej appki RAG)
# -----------------------------
APP_TITLE = "Mini RAG for Bank Statements"
DB_PATH = "/var/www/apintelligence/bank/mrag.sqlite3"
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = None
QDRANT_COLLECTION = "bank_tx_1536"
EMBED_MODEL = "text-embedding-3-small"  # 1536 dims
LLM_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TOP_K = 12
MAX_SNIPPETS = 3
SESSION_COOKIE = "mrag_session"
QDRANT_HTTP_TIMEOUT = 300  # sekundy
QDRANT_BATCH_UPSERT = 256  # punkty na jeden upsert (zmniejsz, jeśli dalej timeout)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DB
# -----------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  page INTEGER NOT NULL,
  tx_date TEXT,
  amount REAL,
  currency TEXT,
  description TEXT,
  raw_line TEXT,
  category TEXT,
  tx_month TEXT
);
"""

SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": "Execute a READ-ONLY SQL query against the local SQLite database.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "Single SELECT statement returning aggregated spend or rows."
                }
            },
            "required": ["sql"]
        }
    }
}


SQL_SYSTEM_PROMPT = (
    "You translate a user question into a single safe SQL SELECT over the schema.\n"
    "Rules:\n"
    "- Return ONLY via the run_sql tool.\n"
    "- One statement, SELECT only, no sub-queries that modify data, no PRAGMA, no semicolons.\n"
    "- Use table 'transactions(session_id, filename, page, tx_date, amount, currency, description, raw_line, category, tx_month)'.\n"
    "- tx_date is 'YYYY-MM-DD', tx_month is 'YYYY-MM'. amount is positive for income or expense depending on parsing. Treat negative amounts as spend.\n"
    "- For 'spend', sum ABS of negative amounts or filter amount < 0.\n"
    "- When month is given (e.g. July), map to MM using: 01-Jan, 02-Feb, ..., 07-Jul, 08-Aug, etc. Use tx_month.\n"
    "- For 'food', use category='food'. If category missing, try description LIKE patterns: '%restaurac%', '%pizza%', '%kebab%', '%lidl%', '%biedronka%', '%zabka%'.\n"
)

# --- Millennium-specific parsing helpers ---
DATE_LINE_RE = re.compile(r'^\s*(\d{4}-\d{2}-\d{2}|\d{2}[.\-/]\d{2}[.\-/]\d{4})\s*$')
AMOUNT_PAIR_RE = re.compile(
    r'^\s*([+\-]?\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2}))\s+'
    r'([+\-]?\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2}))\s*$'
)

SUMMARY_HINTS = (
    "Saldo początkowe", "Suma uznań", "Suma obciążeń", "Saldo końcowe",
    "Data wystawienia dokumentu", "Lista transakcji została", "Dokument nie jest wyciągiem",
    "Bank Millennium S.A.", "Opis", "Obciążenia", "Uznania", "Saldo",
    "Data", "transakcji", "księgowania"
)

FOOD_HINTS = ("BIEDRONKA", "LIDL", "ALDI", "KAUFLAND", "CARREFOUR", "ŻABKA", "ZABKA", "PEPCO",
              "SKLEP RYBNY", "KIK", "STACJA BENZYNOWA", "A330", "SPUTNIK", "KOSCIERZYNA",
              "PSB", "BRICOMARCHE")

# liczba w osobnym wierszu (kwota lub saldo)
AMOUNT_ONLY_RE = re.compile(r'^\s*[+\-]?\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2})\s*$')

# data występująca gdziekolwiek w linii (np. "POL 2025-08-16")
DATE_ANY_INLINE = re.compile(r'(\d{4}-\d{2}-\d{2}|\d{2}[.\-/]\d{2}[.\-/]\d{4})')

# szum/nagłówki/stopki, które ignorujemy
SUMMARY_HINTS = (
    "Saldo początkowe", "Suma uznań", "Suma obciążeń", "Saldo końcowe",
    "Data wystawienia dokumentu", "Lista transakcji została", "Dokument nie jest wyciągiem",
    "Bank Millennium", "Opis", "Obciążenia", "Uznania", "Saldo",
    "Data", "transakcji", "księgowania"
)

# metki kolumn — nie wnoszą treści
LABELS = ("Rodzaj operacji", "Tytuł operacji", "Odbiorca", "Zleceniodawca", "Na rachunek",
          "Numer karty", "Posiadacz karty")

CARD_MASK_RE = re.compile(r'^\d{4}X+', re.I)  # np. "4874XXXXXXXX8777"

from pdfminer.layout import LTTextContainer, LTTextLineHorizontal

from pdfminer.layout import LTTextContainer, LTTextLineHorizontal

def extract_items_with_xy(pdf_file: UploadFile) -> list[tuple[int, float, float, str]]:
    """
    Zwraca listę (page_no, y0, x0, text) – potem grupujemy po y, sortujemy po x.
    """
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.file.read())
        tmp_path = tmp.name

    items: list[tuple[int, float, float, str]] = []
    for page_no, layout in enumerate(extract_pages(tmp_path), start=1):
        for element in layout:
            if isinstance(element, LTTextContainer):
                for child in element:
                    if isinstance(child, LTTextLineHorizontal):
                        t = (child.get_text() or "").strip()
                        if t:
                            items.append((page_no, float(child.y0), float(child.x0), t))
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return items



def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# -----------------------------
# Qdrant + OpenAI
# -----------------------------

def get_qdrant_client() -> QdrantClient:
    kwargs = dict(url=QDRANT_URL, timeout=QDRANT_HTTP_TIMEOUT)
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    try:
        # nowsze biblioteki wspierają prefer_grpc (szybsze, stabilniejsze przy dużych wsadach)
        return QdrantClient(prefer_grpc=True, **kwargs)
    except TypeError:
        # starsza wersja klienta bez prefer_grpc
        return QdrantClient(**kwargs)


def validate_sql(sql: str) -> str:
    # basic guardrails: single statement, SELECT only, no semicolon chains
    parsed = sqlparse.parse(sql)
    if len(parsed) != 1:
        raise ValueError("Only single statement allowed")
    stmt = parsed[0]
    if stmt.get_type() != "SELECT":
        raise ValueError("Only SELECT is allowed")
    s = str(stmt).strip()
    if ";" in s:
        raise ValueError("Semicolons are not allowed")
    banned = ["PRAGMA", "ATTACH", "DETACH", "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "VACUUM"]
    for b in banned:
        if b.lower() in s.lower():
            raise ValueError(f"Keyword not allowed: {b}")
    return s

def answer_via_rag(question: str, sid: str) -> dict:
    client = get_qdrant_client()
    qvec = embed_texts([question])[0]
    flt = Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=sid))])
    sr = client.search(collection_name=QDRANT_COLLECTION, query_vector=qvec,
                       limit=TOP_K, query_filter=flt, with_payload=True)
    snippets = []
    for rec in sr[:MAX_SNIPPETS]:
        pl = rec.payload or {}
        snippets.append({"filename": pl.get("filename",""), "page": pl.get("page",0),
                         "text": (pl.get("text","") or "")[:400]})
    context_lines = []
    for rec in sr:
        pl = rec.payload or {}
        context_lines.append(json.dumps({
            "file": pl.get("filename"), "page": pl.get("page"),
            "date": pl.get("tx_date"), "amount": pl.get("amount"),
            "currency": pl.get("currency"), "desc": pl.get("description"),
        }, ensure_ascii=True))
    context = "\n".join(context_lines[:100])
    prompt = (
        "Context (JSON lines of transactions):\n" + context +
        "\n\nQuestion: " + question +
        "\n\nInstructions: Answer concisely in the same language as the question. "
        "Use only the context. Include up to 3 citations like [filename p.page]. "
        "If insufficient data, say so."
    )
    return {"answer": llm_answer(prompt), "snippets": snippets}


def ensure_collection(client: QdrantClient, vector_size: int = 1536):
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        return
    params = VectorParams(size=vector_size, distance=Distance.COSINE)
    try:
        client.create_collection(collection_name=QDRANT_COLLECTION, vectors_config=params)  # nowsze
    except TypeError:
        client.create_collection(collection_name=QDRANT_COLLECTION, vectors=params)        # starsze


openai_client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY not set")
    # normalizacja + ucięcie do rozsądnej długości
    cleaned: List[str] = []
    for t in texts:
        if t is None:
            continue
        if isinstance(t, bytes):
            t = t.decode("utf-8", errors="ignore")
        s = str(t).strip()
        if not s:
            continue
        cleaned.append(s[:4000])  # tniemy bardzo długie linie

    if not cleaned:
        return []

    out: List[List[float]] = []
    batch_size = 100
    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i:i+batch_size]
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out



def llm_answer(prompt: str) -> str:
    chat = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a financial assistant. Answer ONLY using provided context. If missing, say you do not have enough data."
            )},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return chat.choices[0].message.content.strip()

import unicodedata
def strip_accents(s: str) -> str:
    if not s: return s
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

FOOD_KEYWORDS = [
    "biedronka","żabka","zabka","lidl","carrefour","auchan","kaufland","aldi",
    "stokrotka","piotr i pawel","piotr i paweł","delikatesy","spozyw","spożyw",
    "piekarnia","cukiernia","mcdonald","kfc","burger king","subway","pizza",
    "pizzeria","sushi","kebab","bistro","restauracja","bar mleczny","obiad","jedzenie","food"
]

def is_food(desc: str) -> bool:
    if not desc: return False
    low = strip_accents(desc.lower())
    return any(k in low for k in FOOD_KEYWORDS)

def month_from_date(tx_date: str | None) -> str | None:
    if not tx_date: return None
    m = re.match(r"^(\d{4})-(\d{2})-\d{2}$", tx_date)
    return f"{m.group(1)}-{m.group(2)}" if m else None


# -----------------------------
# PDF parsing
# -----------------------------
# daty: 2024-08-19, 19.08.2024, 19-08-24, 19/08/2024, 2024.08.19
DATE_PATTERNS = [
    r"(?P<date>\d{4}[.\-\/]\d{2}[.\-\/]\d{2})",
    r"(?P<date>\d{2}[.\-\/]\d{2}[.\-\/]\d{4})",
    r"(?P<date>\d{2}[.\-\/]\d{2}[.\-\/]\d{2})",
]

# kwoty: 1 234,56 PLN | -249,00 zł | +12.50 EUR | 300,00
AMOUNT_RE = re.compile(
    r"(?<!\w)"
    r"(?P<sign>[+\-]?)\s*"
    r"(?P<num>\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2}))"
    r"\s*(?P<curr>PLN|EUR|USD|GBP|CHF|zł|zl)?"
    r"(?!\w)"
)
# Kompilacje per-wzorzec (do wyszukiwania) i jeden unijny bez nazwanych grup (do usuwania)
DATE_REGEXES = [re.compile(p) for p in DATE_PATTERNS]

DATE_ANY = re.compile(
    r"(?:\d{4}[.\-\/]\d{2}[.\-\/]\d{2}|\d{2}[.\-\/]\d{2}[.\-\/]\d{4}|\d{2}[.\-\/]\d{2}[.\-\/]\d{2})"
)

def normalize_date(s: str) -> str:
    s = s.strip()
    # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
    if re.match(r"^\d{4}[.\-\/]\d{2}[.\-\/]\d{2}$", s):
        return s.replace("/", "-").replace(".", "-")
    # DD.MM.YYYY / DD-MM-YYYY / DD/MM/YYYY
    m = re.match(r"^(\d{2})[.\-\/](\d{2})[.\-\/](\d{4})$", s)
    if m:
        d, mth, y = m.groups()
        return f"{y}-{mth}-{d}"
    # DD.MM.YY / DD-MM-YY / DD/MM/YY -> 20YY
    m = re.match(r"^(\d{2})[.\-\/](\d{2})[.\-\/](\d{2})$", s)
    if m:
        d, mth, y2 = m.groups()
        y = f"20{y2}"
        return f"{y}-{mth}-{d}"
    return s

def parse_amount_from_text(s: str) -> tuple[float | None, str | None]:
    m = AMOUNT_RE.search(s.replace("\u00A0", " "))  # NBSP -> space
    if not m:
        return None, None
    sign = -1.0 if (m.group("sign") or "") == "-" else 1.0
    num = m.group("num").replace(" ", "").replace("\u00A0", "").replace(",", ".")
    try:
        val = float(num) * sign
    except Exception:
        return None, None
    curr = (m.group("curr") or "").upper()
    if curr in ("ZŁ", "ZL"):
        curr = "PLN"
    return val, (curr or None)

def try_parse_tx(line: str) -> Dict[str, Any]:
    date_val = None
    for pat in DATE_PATTERNS:
        d = re.search(pat, line)
        if d:
            date_val = normalize_date(d.group("date"))
            break
    amount_val, curr = parse_amount_from_text(line)
    # description: usuń znalezioną datę i kwotę
    desc = line
    # usuń daty bez nazwanych grup
    desc = DATE_ANY.sub(" ", desc)
    # usuń kwoty
    desc = AMOUNT_RE.sub(" ", desc)
    desc = re.sub(r"\s+", " ", desc).strip()
#     desc = re.sub(r"\s+", " ", desc).strip()
    return {
        "tx_date": date_val,
        "amount": amount_val,
        "currency": curr,
        "description": desc if desc else line,
    }

# helpery
# helpery u góry pliku (zostaw jeśli już masz)
# --- pomocnicze ---
AMOUNT_TOKEN = re.compile(r'^[+\-]?\d{1,3}(?:[ \u00A0]?\d{3})*(?:[.,]\d{2})$|^[+\-]?\d+[.,]\d{2}$')
SUMMARY_HINTS = (
    "Saldo początkowe","Suma uznań","Suma obciążeń","Saldo końcowe",
    "Data wystawienia dokumentu","Lista transakcji została",
    "Dokument nie jest wyciągiem","Bank Millennium",
)
LABEL_HINTS_SUBSTR = (
    "Rodzaj operacji","Tytuł operacji","Odbiorca","Zleceni",
    "Na rachunek","Z rachunku","Kwota","Saldo","Posiadacz karty","Numer karty"
)

def is_amount_token(s: str) -> bool:
    s = s.strip().replace("\u00A0"," ").replace(" ","")
    return bool(AMOUNT_TOKEN.match(s))

def cluster_two(xs: list[float]) -> tuple[float, float]:
    """prosty k-means 2-klastrowy po x, zwraca ~centra (kwota, saldo)"""
    c0, c1 = min(xs), max(xs)
    for _ in range(8):
        g0, g1 = [], []
        for x in xs:
            (g0 if abs(x-c0) <= abs(x-c1) else g1).append(x)
        if not g0 or not g1: break
        c0, c1 = sum(g0)/len(g0), sum(g1)/len(g1)
    if c0 > c1: c0, c1 = c1, c0
    return c0, c1

def parse_millennium_transactions(items: list[tuple[int,float,float,str]]) -> list[dict]:
    """
    1) grupujemy elementy w wiersze po y (tolerancja 4 px),
    2) per strona wykrywamy kolumny liczbowe (kwota vs saldo) po x,
    3) budujemy rekord transakcji z opisu (lewa strona) i kwoty (druga od prawej).
    Transakcję finalizujemy przy starcie kolejnego bloku ('Rodzaj operacji') lub przy zmianie daty.
    """
    from collections import defaultdict

    # grupowanie wierszy
    ROW_TOL = 4.0
    rows = defaultdict(list)  # (page, ybin) -> [(x, text)]
    for page, y0, x0, t in items:
        rows[(page, round(y0/ROW_TOL))].append((x0, t))

    pages = defaultdict(list)  # page -> [(ybin, [(x, text), ...])]
    for (page, yb), cells in rows.items():
        cells = sorted(cells, key=lambda z: z[0])
        if cells:
            pages[page].append((yb, cells))
    for p in pages:
        pages[p].sort(key=lambda r: r[0])

    txs: list[dict] = []
    last_date: str | None = None

    for page, rowlist in pages.items():
        # wykryj 2 prawe kolumny liczbowe na stronie
        num_xs = [x for _, cells in rowlist for (x, t) in cells if is_amount_token(t)]
        if len(num_xs) >= 4:
            amount_x, saldo_x = cluster_two(num_xs)
        else:
            amount_x, saldo_x = 390.0, 503.0  # rozsądne defaulty dla Millennium

        ctx_desc: list[str] = []
        ctx_amt: float | None = None
        ctx_page = page

        def flush():
            nonlocal ctx_desc, ctx_amt, ctx_page, last_date
            if ctx_amt is None or not ctx_desc:
                ctx_desc, ctx_amt = [], None
                return
            # czyszczenie opisu: out daty, numery, maski kart i same etykiety
            parts: list[str] = []
            for seg in ctx_desc:
                seg = seg.strip()
                if not seg: 
                    continue
                if any(h in seg for h in SUMMARY_HINTS): 
                    continue
                if any(lbl in seg for lbl in LABEL_HINTS_SUBSTR):
                    continue
                if re.fullmatch(r'\d{4}-\d{2}-\d{2}', seg) or re.fullmatch(r'\d{2}[.\-/]\d{2}[.\-/]\d{4}', seg):
                    continue
                if re.search(r'\d{4}X+', seg) or re.search(r'\d{8,}', seg):
                    continue
                # usuń ewentualne saldo/kwotę „doklejoną” do tekstu na końcu
                seg = re.sub(r'\s*[|]?\s*[+\-]?\d{1,3}(?:\s?\d{3})*(?:[.,]\d{2})\s*$', '', seg).strip()
                if seg:
                    parts.append(seg)
            desc = " | ".join(parts)
            desc = re.sub(r'\s*\|\s*(\|\s*)+', ' | ', desc).strip(" |")

            txs.append({
                "page": ctx_page,
                "tx_date": last_date,
                "amount": ctx_amt,
                "currency": "PLN",
                "description": desc[:800],
                "category": "food" if is_food(desc) else None,
                "tx_month": month_from_date(last_date),
            })
            ctx_desc, ctx_amt = [], None

        for _, cells in rowlist:
            # 1) daty w dowolnym polu
            for _, t in cells:
                m = re.search(r'(\d{4}-\d{2}-\d{2}|\d{2}[.\-/]\d{2}[.\-/]\d{4})', t)
                if m:
                    s = m.group(1)
                    if re.fullmatch(r'\d{2}[.\-/]\d{2}[.\-/]\d{4}', s):
                        d, mth, y = s[:2], s[3:5], s[6:10]
                        s = f"{y}-{mth}-{d}"
                    # jeśli nowa data i mamy rozpoczętą transakcję, to ją zamknij
                    if last_date is not None and s != last_date:
                        flush()
                    last_date = s
                    break

            # 2) start nowego bloku
            if any("Rodzaj operacji" in t for _, t in cells):
                flush()
                ctx_desc = []

            # 3) akumuluj opis – tylko lewa strona, bez etykiet i liczb
            for x, t in cells:
                ts = t.strip()
                if any(lbl in ts for lbl in LABEL_HINTS_SUBSTR):
                    continue
                if is_amount_token(ts):
                    continue
                if x >= (amount_x - 5):  # prawa część wiersza omijamy
                    continue
                ctx_desc.append(ts)
                ctx_page = page

            # 4) wyłap kwotę z wiersza (liczby < „pół” między kolumną kwoty a saldem)
            nums = [(x, t.strip()) for x, t in cells if is_amount_token(t)]
            if nums:
                nums.sort(key=lambda z: z[0])
                border = (amount_x + saldo_x) / 2.0
                cand = [t for (x, t) in nums if x < border] or [nums[0][1]]
                ctx_amt = float(cand[-1].replace("\u00A0"," ").replace(" ","").replace(",", "."))

        flush()

    return txs



def extract_lines_with_pages(pdf_file: UploadFile) -> List[Tuple[int, str]]:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.file.read())
        tmp_path = tmp.name
    lines: List[Tuple[int, str]] = []
    for page_no, layout in enumerate(extract_pages(tmp_path), start=1):
        for element in layout:
            if isinstance(element, LTTextContainer):
                for child in element:
                    if isinstance(child, LTTextLineHorizontal):
                        text = child.get_text().strip()
                        if text:
                            lines.append((page_no, text))
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return lines




def ensure_extra_columns():
    conn = get_db()
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(transactions)").fetchall()}
        with conn:
            if "category" not in cols:
                conn.execute("ALTER TABLE transactions ADD COLUMN category TEXT")
            if "tx_month" not in cols:
                conn.execute("ALTER TABLE transactions ADD COLUMN tx_month TEXT")
    finally:
        conn.close()

@app.on_event("startup")
def _startup():
    conn = get_db()
    with conn:
        conn.execute(SCHEMA_SQL)
    ensure_extra_columns()
    client = get_qdrant_client()
    ensure_collection(client, vector_size=1536)

# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
def _startup():
    conn = get_db()
    with conn: conn.execute(SCHEMA_SQL)
    client = get_qdrant_client()
    ensure_collection(client, vector_size=1536)

# -----------------------------
# Upload exactly 1 PDF
# -----------------------------
@app.post("/upload")
async def upload(request: Request, pdf: UploadFile = File(...)):
    sid = uuid.uuid4().hex

    conn = get_db()
    client = get_qdrant_client()

    fname = pdf.filename or "uploaded.pdf"
    items = extract_items_with_xy(pdf)
    tx_records = parse_millennium_transactions(items)


    if not tx_records:
        return JSONResponse({
            "message": "No transactions recognized (Millennium parser).",
            "session_id": sid
        }, status_code=200)

    texts_for_embed: List[str] = []
    payloads: List[Dict[str, Any]] = []
    points: List[PointStruct] = []

    total_tx = 0
    for rec in tx_records:
        with conn:
            conn.execute(
                "INSERT INTO transactions(session_id, filename, page, tx_date, amount, currency, description, raw_line, category, tx_month) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    sid,
                    fname,
                    rec.get("page"),
                    rec.get("tx_date"),
                    rec.get("amount"),
                    rec.get("currency") or "PLN",
                    rec.get("description"),
                    None,  # raw_line nie zapisujemy dla parsera blokowego
                    rec.get("category"),
                    rec.get("tx_month"),
                ),
            )
        total_tx += 1

        txt = (
            f"file:{fname} page:{rec.get('page')} "
            f"date:{rec.get('tx_date')} amount:{rec.get('amount')} {rec.get('currency') or 'PLN'} "
            f"desc:{rec.get('description') or ''} cat:{rec.get('category') or ''} month:{rec.get('tx_month') or ''}"
        )
        texts_for_embed.append(txt)
        payloads.append({
            "session_id": sid,
            "filename": fname,
            "page": rec.get("page"),
            "tx_date": rec.get("tx_date"),
            "amount": rec.get("amount"),
            "currency": rec.get("currency") or "PLN",
            "description": rec.get("description"),
            "category": rec.get("category"),
            "tx_month": rec.get("tx_month"),
            "text": txt,
        })

    # Embeddings i Qdrant
    vectors = embed_texts(texts_for_embed)
    if vectors:
        ensure_collection(client, vector_size=len(vectors[0]))
        for vec, pl in zip(vectors, payloads):
            points.append(PointStruct(id=int(uuid.uuid4().int % 10**16), vector=vec, payload=pl))
        # wsadowo, z wait=True
        for i in range(0, len(points), QDRANT_BATCH_UPSERT):
            chunk = points[i:i + QDRANT_BATCH_UPSERT]
            client.upsert(collection_name=QDRANT_COLLECTION, points=chunk, wait=True)

    return JSONResponse({
        "message": f"Indexed {len(points)} transactions from 1 PDF (Millennium parser). Parsed tx: {total_tx}.",
        "session_id": sid
    })


# -----------------------------
# Ask endpoint
# -----------------------------
@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    question = (body.get("question") or "").strip()
    sid = (body.get("session_id") or "").strip()
    if not question or not sid:
        return JSONResponse({"error": "Missing question or session_id"}, status_code=400)

    # --- FAZA 1: Poproś LLM o SQL (Text-to-SQL agent) ---
    try:
        schema_hint = (
            "Schema:\n"
            "transactions(session_id TEXT, filename TEXT, page INT, tx_date TEXT, amount REAL, currency TEXT,\n"
            "description TEXT, raw_line TEXT, category TEXT, tx_month TEXT)\n"
            f"Target session_id: {sid}\n"
        )
        plan = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SQL_SYSTEM_PROMPT},
                {"role": "user", "content": schema_hint + "\nQuestion: " + question},
            ],
            tools=[SQL_TOOL],
            tool_choice="auto",
            temperature=0.0,
        )
        sql = None
        for tc in (plan.choices[0].message.tool_calls or []):
            if tc.function.name == "run_sql":
                sql = json.loads(tc.function.arguments or "{}").get("sql")
                break
        if not sql:
            raise RuntimeError("No SQL from tool")

        safe_sql = validate_sql(sql)

        # wykonaj SQL
        conn = get_db()
        try:
            cur = conn.execute(safe_sql)
            cols = [d[0] for d in (cur.description or [])]
            rows = cur.fetchall()
        finally:
            conn.close()

        result_table = {
            "columns": cols,
            "rows": [[("" if v is None else v) for v in r] for r in rows[:100]]
        }

        # heurystyczne filtry do cytatów (opcjonalnie)
        month_filter = None
        cat_filter = None
        m = re.search(r"tx_month\s*=\s*'(\d{4}-\d{2})'", safe_sql)
        if m: month_filter = m.group(1)
        c = re.search(r"category\s*=\s*'([^']+)'", safe_sql)
        if c: cat_filter = c.group(1)

        must = [FieldCondition(key="session_id", match=MatchValue(value=sid))]
        if month_filter: must.append(FieldCondition(key="tx_month", match=MatchValue(value=month_filter)))
        if cat_filter:   must.append(FieldCondition(key="category", match=MatchValue(value=cat_filter)))

        client = get_qdrant_client()
        qvec = embed_texts([question])[0]
        sr = client.search(collection_name=QDRANT_COLLECTION, query_vector=qvec,
                           limit=TOP_K, query_filter=Filter(must=must), with_payload=True)

        snippets = []
        for rec in sr[:MAX_SNIPPETS]:
            pl = rec.payload or {}
            snippets.append({"filename": pl.get("filename",""), "page": pl.get("page",0),
                             "text": (pl.get("text","") or "")[:400]})

        final = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content":
                    "Answer strictly based on the provided table and snippets. "
                    "Be concise and reply in the user's language. "
                    "If a total is requested, compute it from the table. "
                    "Include up to 3 citations like [filename p.page]."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": "Context JSON follows."},
                {"role": "user", "content": json.dumps({
                    "sql_executed": safe_sql,
                    "result_table": result_table,
                    "snippets": snippets
                }, ensure_ascii=True)},
            ],
            temperature=0.2,
        )
        answer = final.choices[0].message.content.strip()
        return JSONResponse({"answer": answer, "snippets": snippets, "debug_sql": safe_sql})

    except Exception:
        # --- FAZA 2: Fallback do Twojego dotychczasowego RAG ---
        out = answer_via_rag(question, sid)
        return JSONResponse(out)

