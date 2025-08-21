


<!--# --------- /var/www/apintelligence/bank/index.php (frontend) ----------->
<!--# PHP + HTML + JS frontend. Save as: /var/www/apintelligence/bank/index.php-->
<!--# Talks to the Python API (mrag_app.py) running e.g. on http://127.0.0.1:8010-->

<?php
$API_BASE = 'https://apintelligence.tech:8011'; // adjust if uvicorn runs elsewhere
?>
<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mini RAG for Bank Statements</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 860px; margin: 24px auto; padding: 0 16px; }
    header { display: flex; justify-content: space-between; align-items: center; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 12px 0; }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    input[type=file] { padding: 8px; }
    button { padding: 8px 12px; cursor: pointer; }
    #messages { border: 1px solid #eee; border-radius: 8px; padding: 12px; min-height: 160px; }
    .msg-user { background: #eef6ff; padding: 8px; border-radius: 6px; margin: 6px 0; }
    .msg-ai { background: #f6f6f6; padding: 8px; border-radius: 6px; margin: 6px 0; }
    .snippet { font-family: ui-monospace, monospace; font-size: 12px; background: #fcfcfc; border-left: 3px solid #ddd; padding: 6px 8px; margin: 4px 0; white-space: pre-wrap; }
    .muted { color: #666; font-size: 12px; }
  </style>
</head>
<body>
  <header>
    <h1>Mini RAG for Bank Statements</h1>
    <div class="muted">Session: <span id="sid">-</span></div>
  </header>

  <div class="card">
    <h3>1) Upload exactly 1 PDF (history 3 months)</h3>
    <form id="uploadForm" enctype="multipart/form-data">
      <input id="pdf" type="file" name="pdf" accept="application/pdf" required />
      <button type="submit">Upload and Index</button>
    </form>
    <div id="uploadStatus" class="muted"></div>
  </div>

  <div class="card">
    <h3>2) Ask a question</h3>
    <div id="messages"></div>
    <div class="row" style="margin-top:8px;">
      <input id="q" type="text" placeholder="Type your question..." style="flex:1; padding:8px;" />
      <button id="askBtn">Ask</button>
    </div>
  </div>

  <script>
    const API_BASE = "<?= $API_BASE ?>";
    const SID_KEY = 'bank_session_id';

    function setSid(val){ localStorage.setItem(SID_KEY, val); document.getElementById('sid').textContent = (val||'').slice(0,8)+'...'; }
    function getSid(){ return localStorage.getItem(SID_KEY) || ''; }

    // UI helpers
    function addMsg(cls, html){
      const div = document.createElement('div');
      div.className = cls;
      div.innerHTML = html;
      document.getElementById('messages').appendChild(div);
    }

    // Upload handler: exactly 1 PDF
    document.getElementById('uploadForm').addEventListener('submit', async (e)=>{
      e.preventDefault();
      const input = document.getElementById('pdf');
      if (!input.files || input.files.length !== 1){
        alert('Please select exactly 1 PDF file.');
        return;
      }
      const fd = new FormData();
      fd.append('pdf', input.files[0]);
      // const res = await fetch(API_BASE + '/upload', { method:'POST', body: fd });
      const res = await fetch('proxy_upload.php', { method: 'POST', body: fd });

      const data = await res.json();
      if (data.session_id){ setSid(data.session_id); }
      document.getElementById('uploadStatus').textContent = data.message || JSON.stringify(data);
    });

    async function ask(){
      const q = document.getElementById('q').value.trim();
      const sid = getSid();
      if(!q){ return; }
      if(!sid){ alert('Please upload a PDF first.'); return; }
      addMsg('msg-user', q);
      document.getElementById('q').value = '';
      const res = await fetch('proxy_ask.php', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, session_id: getSid() })
      });

      const data = await res.json();
      let html = '';
      html += `<div>${data.answer || '(no answer)'}`;
      if (data.snippets && data.snippets.length){
        html += '<div class="muted" style="margin-top:6px;">Sources:</div>';
        for (const s of data.snippets){
          html += `<div class="snippet">[${s.filename} p.${s.page}] ${s.text}</div>`;
        }
      }
      html += '</div>';
      addMsg('msg-ai', html);
    }
    document.getElementById('askBtn').addEventListener('click', ask);
    document.getElementById('q').addEventListener('keydown', (e)=>{ if(e.key==='Enter') ask(); });

    // Init
    setSid(getSid());
  </script>
</body>
</html>
