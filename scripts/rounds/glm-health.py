"""Is GLM-5.4 usable on AI Hub? Exit 0 when six small Responses requests on the zhipu passthrough all succeed and the
platform-wide success rate of the last hour (AI Hub reliability tool) is at least 90% where it is reported.
On 2026-11-28 the model failed almost every request with AI Hub 1011 (no available upstream account).

Usage: glm-health.py   (one JSON line on stdout)
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

URL = 'http://api.aihub.woa.com/api/v1/responses'
METRICS = '/root/.cursor/skills/aihub-skills/sub-skills/reliability-ops-skill/scripts/connect_mcp.py'


def probe() -> list:
    auth = 'Bearer ' + open('/root/.frontieram/aihub.key').read().strip() + '?provider=zhipu'
    body = json.dumps({'model': 'glm-5.4', 'input': 'Reply with OK.', 'max_output_tokens': 300, 'store': False,
                       'reasoning': {'effort': 'low'}}).encode()
    statuses = []
    for _ in range(6):
        request = urllib.request.Request(URL, data=body, method='POST',
                                         headers={'Authorization': auth, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                statuses.append(response.status)
        except urllib.error.HTTPError as error:
            statuses.append(error.code)
        except OSError:
            statuses.append(0)
        time.sleep(5)
    return statuses


def platform() -> dict | None:
    done = subprocess.run([sys.executable, METRICS, 'call', 'query_metrics_success_rate',
                           json.dumps({'model_id': 'glm-5.4', 'hours': 1})], capture_output=True, text=True, timeout=180)
    try:
        rows = json.loads(done.stdout)['data']['metrics']
    except (ValueError, KeyError, TypeError):
        return None
    return rows[0] if rows else None


statuses = probe()
metric = platform() if all(s == 200 for s in statuses) else None
rate = (metric or {}).get('success_rate')
healthy = all(s == 200 for s in statuses) and (rate is None or rate >= 90)
print(json.dumps({'time': time.strftime('%FT%T'), 'probe': statuses, 'platform_success_rate': rate,
                  'platform_requests': (metric or {}).get('request_count'), 'healthy': healthy}))
sys.exit(0 if healthy else 1)
