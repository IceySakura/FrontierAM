"""Upload sandbox files to host-issued COS URLs using Python 3.7+ stdlib."""

import base64
import hashlib
import json
import os
import urllib.request


CHUNK_SIZE = 1024 * 1024


def inspect_file(path):
    digest = hashlib.md5()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b''):
            digest.update(chunk)
        size = source.tell()
    return {'size_bytes': size, 'content_md5': base64.b64encode(digest.digest()).decode('ascii')}


def upload_file(request):
    path = request['path']
    offset = request.get('offset', 0)
    length = request['length']
    if offset < 0 or length < 0 or offset + length > os.path.getsize(path):
        raise ValueError('Invalid upload file range')

    def chunks():
        with open(path, 'rb') as source:
            source.seek(offset)
            remaining = length
            while remaining:
                chunk = source.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    raise OSError('Recording file changed during upload')
                remaining -= len(chunk)
                yield chunk

    headers = dict(request.get('headers') or {})
    headers['Content-Length'] = str(length)
    upload = urllib.request.Request(
        request['url'], data=chunks(), headers=headers, method=request.get('method') or 'PUT'
    )
    with urllib.request.urlopen(upload, timeout=request['timeout']) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f'COS upload failed (HTTP {response.status})')
        etag = response.headers.get('ETag')
        if not etag:
            raise RuntimeError('COS upload returned no ETag')
        return {'etag': etag}


def main(payload):
    request = json.loads(payload)
    try:
        if request['action'] == 'inspect':
            result = inspect_file(request['path'])
        elif request['action'] == 'upload':
            result = {} if request.get('already_uploaded') else upload_file(request)
            if request.get('delete_source', True):
                os.remove(request['path'])
                result['source_deleted'] = True
        else:
            raise ValueError('Unknown artifact upload action: ' + request['action'])
        result['success'] = True
    except Exception as exc:
        result = {'success': False, 'error': str(exc)}
    print('NEXUS_ARTIFACT_UPLOAD_RESULT:' + json.dumps(result))
