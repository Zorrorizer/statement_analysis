<?php
declare(strict_types=1);

// proxy_upload.php -> forwards 1 PDF to FastAPI /upload on localhost:8011
error_reporting(E_ALL);
ini_set('display_errors', '1');

$BACKEND_URL = 'http://127.0.0.1:8011/upload';
$TIMEOUT     = 600;

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    header('Content-Type: text/plain; charset=UTF-8');
    exit('POST only');
}

if (empty($_FILES['pdf']['tmp_name'])) {
    http_response_code(400);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'pdf file required']);
    exit;
}

// optional light checks
$mime = $_FILES['pdf']['type'] ?? '';
if ($mime && stripos($mime, 'pdf') === false) {
    http_response_code(400);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'only PDF allowed']);
    exit;
}

$postFields = [
    'pdf' => new CURLFile(
        $_FILES['pdf']['tmp_name'],
        $_FILES['pdf']['type'] ?: 'application/pdf',
        $_FILES['pdf']['name'] ?: 'statement.pdf'
    ),
];
//save pdf to disk for debugging
//filename with hour minute second
$timestamp = date('Ymd_His');

$filename= 'uploaded_' . $timestamp . '.pdf';
file_put_contents('/var/www/apintelligence/'.$filename, file_get_contents($_FILES['pdf']['tmp_name']));

$ch = curl_init($BACKEND_URL);
curl_setopt_array($ch, [
    CURLOPT_POST           => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_TIMEOUT        => $TIMEOUT,
    CURLOPT_POSTFIELDS     => $postFields,
    CURLOPT_HTTPHEADER     => ['Accept: application/json', 'Expect:'] // disable 100-continue
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
    echo $response; // pass-through JSON from FastAPI (contains session_id)
} else {
    echo json_encode(['error' => 'empty-response', 'http_code' => $code]);
}
