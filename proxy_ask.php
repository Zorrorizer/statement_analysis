<?php
declare(strict_types=1);

// proxy_ask.php -> forwards {question, session_id} JSON to FastAPI /ask on localhost:8011
error_reporting(E_ALL);
ini_set('display_errors', '1');

$BACKEND_URL = 'http://127.0.0.1:8011/ask';
$TIMEOUT     = 60;

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    header('Content-Type: text/plain; charset=UTF-8');
    exit('POST only');
}

// accept JSON body or x-www-form-urlencoded fallback
$raw = file_get_contents('php://input') ?: '';
$payload = [];
if ($raw !== '' && str_starts_with($_SERVER['CONTENT_TYPE'] ?? '', 'application/json')) {
    $payload = json_decode($raw, true) ?: [];
} else {
    $payload = [
        'question'   => $_POST['question']   ?? '',
        'session_id' => $_POST['session_id'] ?? '',
    ];
}

$question   = trim((string)($payload['question']   ?? ''));
$session_id = trim((string)($payload['session_id'] ?? ''));

if ($question === '' || $session_id === '') {
    http_response_code(400);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'question and session_id required']);
    exit;
}

$jsonOut = json_encode(['question' => $question, 'session_id' => $session_id], JSON_UNESCAPED_SLASHES);

$ch = curl_init($BACKEND_URL);
curl_setopt_array($ch, [
    CURLOPT_POST           => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_TIMEOUT        => $TIMEOUT,
    CURLOPT_POSTFIELDS     => $jsonOut,
    CURLOPT_HTTPHEADER     => [
        'Content-Type: application/json',
        'Accept: application/json',
    ],
]);

$response = curl_exec($ch);
$code     = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$errno    = curl_errno($ch);
$errstr   = curl_error($ch);
curl_close($ch);

header('Content-Type: application/json');
http_response_code($code ?: 500);

if ($errno !== 0) {
    echo json_encode(['error' => 'curl-fail', 'errno' => $errno, 'errstr' => $errstr]);
} elseif ($response !== false && $response !== '') {
    echo $response; // pass-through JSON from FastAPI (answer + snippets)
} else {
    echo json_encode(['error' => 'empty-response', 'http_code' => $code]);
}
