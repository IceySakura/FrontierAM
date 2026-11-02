from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from nexus.extensions.projects.hy_live.node_proxy_preload import PROXY_PRELOAD_SETUP


def test_workbuddy_node_children_keep_proxy_certificates_and_preload_without_changing_other_commands() -> None:
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required to execute the proxy preload')
    script = PROXY_PRELOAD_SETUP.split("<<'JS'\n", 1)[1].split('\nJS', 1)[0]
    harness = r"""
const vm = require('node:vm');
const calls = [];
const childProcess = {spawn(...args) { calls.push(args); return {pid: 1}; }};
const pty = {spawn(...args) { calls.push(args); return {pid: 2}; }};
const parentEnv = {
  HTTPS_PROXY: 'http://platform-proxy:8080',
  NO_PROXY: 'localhost,127.0.0.1',
  NODE_EXTRA_CA_CERTS: '/platform/ca.pem',
  SSL_CERT_FILE: '/platform/ca.pem',
  NODE_OPTIONS: '--require /tmp/nexus-node-proxy/preload.js',
  NEXUS_WORKBUDDY_PROXY_CHILDREN: '1',
};
const context = {
  require(name) {
    if (name === 'fs') return {realpathSync() { return '/runtime/node'; }};
    if (name === 'node:child_process') return childProcess;
    if (name === 'node:module') return {
      syncBuiltinESMExports() {},
      createRequire() { return () => pty; },
    };
    return {setGlobalDispatcher() {}, EnvHttpProxyAgent: class {}};
  },
  process: {
    execPath: '/runtime/node', env: parentEnv,
    argv: ['/runtime/node', '/wb/lib/sidecar-entry.js'], platform: 'linux',
  },
};
vm.runInNewContext(SCRIPT, context);
childProcess.spawn('/runtime/node', ['/wb/lib/sidecar-entry.js'], {
  env: {NODE_OPTIONS: '--max-old-space-size=1024'},
  detached: true,
});
pty.spawn('/runtime/node', ['/wb/cli/bin/codebuddy', '--serve'], {
  env: {NODE_OPTIONS: '', CODEBUDDY_API_KEY: 'search-key'},
});
childProcess.spawn('/runtime/node', ['/workspace/app.js'], {env: {NODE_OPTIONS: ''}});
childProcess.spawn('/bin/bash', ['-c', 'true'], {env: {}});
console.log(JSON.stringify(calls));
"""
    completed = subprocess.run(
        [node, '-e', harness.replace('SCRIPT', json.dumps(script))],
        check=True,
        capture_output=True,
        text=True,
    )
    calls = json.loads(completed.stdout)
    for call in calls[:2]:
        env = call[2]['env']
        assert env['HTTPS_PROXY'] == 'http://platform-proxy:8080'
        assert env['NODE_EXTRA_CA_CERTS'] == '/platform/ca.pem'
        assert env['NODE_OPTIONS'].startswith('--require /tmp/nexus-node-proxy/preload.js')
    assert '--max-old-space-size=1024' in calls[0][2]['env']['NODE_OPTIONS']
    assert calls[0][2]['detached']
    assert calls[1][2]['env']['CODEBUDDY_API_KEY'] == 'search-key'
    assert calls[2][2]['env'] == {'NODE_OPTIONS': ''}
    assert calls[3][2]['env'] == {}
