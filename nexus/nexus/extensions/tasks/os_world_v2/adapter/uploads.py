"""V2 multipart uploads through size-limited AGS gateways, preserving official upload semantics."""

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import shlex

from aiohttp import web


CHUNK_SIZE = 8 * 1024**2
MAX_SIZE = 1024**3
ASSEMBLE = r"""
import hashlib,http.client,json,os,shutil,base64
s=SPEC
assembled=s["directory"]+"/assembled"
digest=hashlib.sha256()
with open(assembled,"wb") as target:
 for index in range(s["count"]):
  path=s["directory"]+f"/part-{index:06d}"
  with open(path,"rb") as source:
   while block:=source.read(1024*1024):
    digest.update(block)
    target.write(block)
  os.unlink(path)
if os.path.getsize(assembled)!=s["size"] or digest.hexdigest()!=s["sha256"]:
 raise RuntimeError("V2 staged upload size/SHA256 mismatch")
boundary="nexus-v2-upload-"+s["nonce"]
prefix=(
 "--"+boundary+"\r\nContent-Disposition: form-data; name=\"file_path\"\r\n\r\n"+
 s["destination"]+"\r\n--"+boundary+
 "\r\nContent-Disposition: form-data; name=\"file_data\"; filename=\"upload.bin\""+
 "\r\nContent-Type: application/octet-stream\r\n\r\n"
).encode()
suffix=("\r\n--"+boundary+"--\r\n").encode()
c=http.client.HTTPConnection("127.0.0.1",5000,timeout=600)
c.putrequest("POST","/setup/upload")
c.putheader("Content-Type","multipart/form-data; boundary="+boundary)
c.putheader("Content-Length",str(len(prefix)+s["size"]+len(suffix)))
c.endheaders()
c.send(prefix)
with open(assembled,"rb") as source:
 while block:=source.read(1024*1024):
  c.send(block)
c.send(suffix)
r=c.getresponse()
body=r.read(65536)
print("NEXUS_UPLOAD_RESPONSE="+json.dumps({
 "status":r.status,"body":base64.b64encode(body).decode(),
 "content_type":r.getheader("Content-Type","text/plain")
}))
c.close()
"""


async def upload_large_multipart(runtime, request):
    """Bound host memory, verify bytes, then call the unmodified guest upload endpoint."""
    directory = '/tmp/nexus-v2-upload-' + secrets.token_hex(16)
    created = await runtime.run_command(command=f'umask 077; mkdir {directory}', timeout=20)
    if created.return_code != 0:
        raise web.HTTPBadGateway(text='Cannot create guest upload staging directory')
    digest = hashlib.sha256()
    count, size, destination = 0, 0, None
    seen_file = False

    async def upload_chunk(block):
        nonlocal count
        remote = directory + f'/part-{count:06d}'
        # Retrying an immutable chunk at a unique path is safe; do not retry final
        # submission to the official endpoint after an ambiguous transport error.
        for attempt in range(3):
            response = await runtime.setup_upload_bytes(remote, block, filename='part')
            if response.status_code == 200:
                count += 1
                return
            if response.status_code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                raise web.HTTPBadGateway(text=f'Guest upload chunk failed: HTTP {response.status_code}')
            await asyncio.sleep(attempt + 1)

    try:
        reader = await request.multipart()
        async for part in reader:
            if part.name == 'file_path' and destination is None:
                raw = await part.read_chunk(8192)
                if not part.at_eof() or len(raw) > 4096:
                    raise web.HTTPBadRequest(text='Invalid upload destination length')
                destination = raw.decode()
                if not destination.startswith('/') or any(c in destination for c in '\r\n\0'):
                    raise web.HTTPBadRequest(text='Invalid upload destination')
            elif part.name in {'file_data', 'file'} and not seen_file:
                seen_file = True
                pending = bytearray()
                while block := await part.read_chunk(256 * 1024):
                    size += len(block)
                    if size > MAX_SIZE:
                        raise web.HTTPRequestEntityTooLarge(max_size=MAX_SIZE, actual_size=size)
                    digest.update(block)
                    pending.extend(block)
                    while len(pending) >= CHUNK_SIZE:
                        await upload_chunk(bytes(pending[:CHUNK_SIZE]))
                        del pending[:CHUNK_SIZE]
                if pending or size == 0:
                    await upload_chunk(bytes(pending))
            else:
                raise web.HTTPBadRequest(text='Unsupported or duplicate upload form field')
        if destination is None or not seen_file:
            raise web.HTTPBadRequest(text='Upload requires file_path and file_data')
        spec = {
            'directory': directory,
            'count': count,
            'size': size,
            'sha256': digest.hexdigest(),
            'destination': destination,
            'nonce': secrets.token_hex(16),
        }
        # /execute may hold the guest HTTP server's execution lock. Calling
        # /setup/upload synchronously from inside /execute deadlocks that server.
        # Detach submission, return /execute, and only poll the bounded result.
        program = 'python3 -c ' + shlex.quote(ASSEMBLE.replace('SPEC', repr(spec)))
        result_path = directory + '/result'
        launched = await runtime.run_command(
            command=f'nohup sh -c {shlex.quote(program + " > " + result_path + "; touch " + directory + "/done")} '
            f'>/dev/null 2>{directory}/stderr </dev/null &',
            timeout=20,
        )
        if launched.return_code != 0:
            raise web.HTTPBadGateway(text='Cannot launch guest upload submission')
        deadline = asyncio.get_running_loop().time() + 650
        while True:
            result = await runtime.run_command(
                command=f'test -f {directory}/done && cat {result_path}',
                timeout=20,
            )
            if result.return_code == 0:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise web.HTTPGatewayTimeout(text='Guest upload submission did not finish')
            await asyncio.sleep(1)
        lines = [
            row.removeprefix('NEXUS_UPLOAD_RESPONSE=')
            for row in (result.stdout or '').splitlines()
            if row.startswith('NEXUS_UPLOAD_RESPONSE=')
        ]
        if result.return_code != 0 or len(lines) != 1:
            raise web.HTTPBadGateway(text=f'Guest upload assembly/submission failed (rc={result.return_code})')
        data = json.loads(lines[0])
        logging.getLogger(__name__).info(
            'V2 staged upload bytes=%s chunks=%s SHA256=%s guest_status=%s',
            size,
            count,
            digest.hexdigest(),
            data['status'],
        )
        return web.Response(
            body=base64.b64decode(data['body']),
            status=data['status'],
            headers={'Content-Type': data['content_type']},
        )
    finally:
        try:
            await runtime.run_command(command=f'rm -rf -- {directory}', timeout=20)
        except Exception:
            logging.getLogger(__name__).warning(
                'V2 temporary upload cleanup failed; sandbox release owns final cleanup'
            )
