import os
import shutil
import subprocess
import sys
import textwrap
import time

from pathlib import Path


ROOT = Path('/home/user/cua_gym_web_form')
STATE_PATH = ROOT / 'answer.txt'
SERVER_PATH = ROOT / 'server.py'
CHROME_LOG_PATH = ROOT / 'chrome.log'


ROOT.mkdir(parents=True, exist_ok=True)
STATE_PATH.write_text('', encoding='utf-8')
SERVER_PATH.write_text(
    textwrap.dedent(
        """\
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from pathlib import Path
        from urllib.parse import parse_qs


        STATE_PATH = Path('/home/user/cua_gym_web_form/answer.txt')
        PAGE = '''<!doctype html>
        <html>
          <head>
            <meta charset="utf-8">
            <title>CUA-Gym Web Form</title>
            <style>
              body {
                align-items: center;
                background: #f7f7f4;
                color: #202124;
                display: flex;
                font-family: Arial, sans-serif;
                height: 100vh;
                justify-content: center;
                margin: 0;
              }
              main {
                border: 1px solid #c9c9c2;
                padding: 32px;
                width: 460px;
              }
              label {
                display: block;
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 14px;
              }
              input {
                border: 2px solid #5f6368;
                box-sizing: border-box;
                font-size: 30px;
                height: 56px;
                padding: 8px 12px;
                width: 100%;
              }
            </style>
          </head>
          <body>
            <main id="answer-card">
              <label for="answer">Answer</label>
              <input id="answer" name="answer" autofocus autocomplete="off">
            </main>
            <script>
              const card = document.getElementById('answer-card');
              const input = document.getElementById('answer');
              function focusInput() {
                input.focus();
                input.select();
              }
              async function saveInput() {
                await fetch('/save', {
                  method: 'POST',
                  headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                  body: new URLSearchParams({answer: input.value})
                });
              }
              window.addEventListener('load', focusInput);
              setTimeout(focusInput, 100);
              setTimeout(focusInput, 500);
              setTimeout(focusInput, 1500);
              setTimeout(focusInput, 3000);
              card.addEventListener('click', (event) => {
                if (event.target !== input) {
                  input.focus();
                }
              });
              input.addEventListener('input', saveInput);
            </script>
          </body>
        </html>'''


        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(PAGE.encode('utf-8'))

            def do_POST(self):
                if self.path != '/save':
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers.get('Content-Length', '0') or '0')
                body = self.rfile.read(length).decode('utf-8')
                value = parse_qs(body).get('answer', [''])[0]
                STATE_PATH.write_text(value, encoding='utf-8')
                self.send_response(204)
                self.end_headers()

            def log_message(self, format, *args):
                pass


        ThreadingHTTPServer(('127.0.0.1', 18099), Handler).serve_forever()
        """
    ),
    encoding='utf-8',
)

server_process = subprocess.Popen(
    [sys.executable, str(SERVER_PATH)],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
time.sleep(1)

chrome = next(
    (
        binary
        for binary in ('google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser')
        if shutil.which(binary)
    ),
    None,
)
if chrome is None:
    server_process.terminate()
    raise RuntimeError('No Chrome or Chromium executable found')

env = os.environ.copy()
env.setdefault('DISPLAY', ':0')
env.setdefault('HOME', '/home/user')
env.setdefault('XDG_RUNTIME_DIR', '/run/user/1000')

chrome_log = CHROME_LOG_PATH.open('ab')
subprocess.Popen(
    [
        chrome,
        '--no-first-run',
        '--disable-fre',
        '--disable-default-apps',
        '--disable-sync',
        '--disable-gpu',
        '--disable-session-crashed-bubble',
        '--test-type',
        f'--user-data-dir={ROOT / "chrome-profile"}',
        '--app=http://127.0.0.1:18099',
    ],
    stdout=chrome_log,
    stderr=chrome_log,
    env=env,
    start_new_session=True,
)
for _ in range(20):
    time.sleep(0.5)
    check = subprocess.run(
        ['pgrep', '-f', 'chrome.*127.0.0.1:18099'],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check.returncode == 0:
        break
else:
    log = CHROME_LOG_PATH.read_text(encoding='utf-8', errors='replace') if CHROME_LOG_PATH.exists() else ''
    server_process.terminate()
    raise RuntimeError(f'Chrome did not start for CUA-Gym web form. Log: {log[-2000:]}')
