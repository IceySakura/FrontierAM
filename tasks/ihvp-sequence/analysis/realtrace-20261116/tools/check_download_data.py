"""Tiny three-part assembly, validation failures, and existing-file protection."""
from pathlib import Path
import copy,hashlib,json,tempfile
from download_data import assemble,DownloadError,validate_manifest,ZIP_NAME,DEFAULT_REPO,DEFAULT_TAG

def rejected(call):
    try:call()
    except DownloadError:return
    raise AssertionError('invalid input accepted')

with tempfile.TemporaryDirectory(prefix='ihvp-download-check-') as temp:
    root=Path(temp);chunks=[b'PK\x03\x04abc',bytes(range(256)),b'end\x00\xff'];data=b''.join(chunks)
    manifest={'file':ZIP_NAME,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'repo':DEFAULT_REPO,'tag':DEFAULT_TAG,'parts':[]}
    for index,chunk in enumerate(chunks,1):
        name=f'data.part{index:03d}';(root/name).write_bytes(chunk)
        manifest['parts'].append({'name':name,'bytes':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()})
    validate_manifest(manifest,expected_parts=3)
    target=assemble(manifest,root);assert target.read_bytes()==data
    assert not (root/(ZIP_NAME+'.partial')).exists()
    assert assemble(manifest,root)==target
    target.write_bytes(b'user file');rejected(lambda:assemble(manifest,root));assert target.read_bytes()==b'user file';target.unlink()
    part=root/manifest['parts'][1]['name'];part.write_bytes(b'x'*len(chunks[1]));rejected(lambda:assemble(manifest,root));assert part.read_bytes()==b'x'*len(chunks[1]);part.write_bytes(chunks[1])
    bad=copy.deepcopy(manifest);bad['sha256']='0'*64;rejected(lambda:assemble(bad,root));assert not target.exists() and not (root/(ZIP_NAME+'.partial')).exists()
    partial=root/(ZIP_NAME+'.partial');partial.write_bytes(b'user partial');rejected(lambda:assemble(manifest,root));assert partial.read_bytes()==b'user partial';partial.unlink()
    # Production manifests require exactly eight parts; fixtures explicitly opt into three.
    rejected(lambda:validate_manifest(manifest))
    for change in ['path','duplicate','hash','size','repo','file','count']:
        bad=copy.deepcopy(manifest)
        if change=='path':bad['parts'][0]['name']='../escape'
        elif change=='duplicate':bad['parts'][1]['name']=bad['parts'][0]['name']
        elif change=='hash':bad['parts'][0]['sha256']='not a hash'
        elif change=='size':bad['bytes']+=1
        elif change=='repo':bad['repo']='unexpected/repository'
        elif change=='file':bad['file']='/tmp/unexpected.zip'
        else:bad['parts'].pop()
        rejected(lambda:validate_manifest(bad,expected_parts=3))
print('PASS: three-part byte roundtrip, hashes/sizes, ordered assembly, manifest boundaries and no overwrite of existing files')
