"""Download the private release's data parts, verify, and assemble without extracting.

Requires an authenticated gh CLI for downloading. Existing verified parts are reused;
an existing incorrect file is never overwritten. --assemble-only works offline.

Receive: python3 tools/download_data.py --out /path/to/evaluator-data
Offline: python3 tools/download_data.py --out /path/to/evaluator-data --assemble-only
Keep IHVP_DATA_PARTS.json and all eight parts in that directory for offline assembly.
Allow about twice the ZIP size in free space; the tool retains the downloaded parts.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,sys,tempfile

DEFAULT_REPO='IceySakura/FrontierAM'
DEFAULT_TAG='ihvp-sequence-statement-20261117'
MANIFEST_NAME='IHVP_DATA_PARTS.json'
ZIP_NAME='EVALUATOR_ONLY_ihvp_balanced_15public_30private_statement_20261117.zip'
EXPECTED_PARTS=8
CHUNK_BYTES=8*1024*1024

class DownloadError(Exception):
    pass

def _safe_name(value):
    return isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,240}',value) is not None

def _positive_int(value):
    return type(value) is int and value>0

def _sha256(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None

def validate_manifest(payload,*,repo=DEFAULT_REPO,tag=DEFAULT_TAG,expected_parts=EXPECTED_PARTS):
    if not isinstance(payload,dict):raise DownloadError('The data manifest must be a JSON object.')
    if payload.get('repo')!=repo or payload.get('tag')!=tag:
        raise DownloadError('The data manifest belongs to a different repository or release.')
    if payload.get('file')!=ZIP_NAME or not _safe_name(payload.get('file')):
        raise DownloadError('The data manifest contains an unexpected ZIP filename.')
    if not _positive_int(payload.get('bytes')) or not _sha256(payload.get('sha256')):
        raise DownloadError('The data manifest contains an invalid ZIP size or SHA256.')
    parts=payload.get('parts')
    if not isinstance(parts,list) or len(parts)!=expected_parts:
        raise DownloadError(f'The data manifest must contain exactly {expected_parts} parts.')
    names=set();total=0
    forbidden={MANIFEST_NAME,ZIP_NAME,ZIP_NAME+'.partial'}
    for part in parts:
        if not isinstance(part,dict):raise DownloadError('Invalid data part entry.')
        name=part.get('name')
        if not _safe_name(name) or name in forbidden or name in names:
            raise DownloadError('The manifest contains an unsafe or duplicate part filename.')
        if not _positive_int(part.get('bytes')) or not _sha256(part.get('sha256')):
            raise DownloadError('A data part has an invalid size or SHA256.')
        names.add(name);total+=part['bytes']
    if total!=payload['bytes']:raise DownloadError('Part sizes do not add up to the ZIP size.')
    return payload

def load_manifest(path,**kwargs):
    path=Path(path)
    try:
        if path.stat().st_size>1024*1024:raise DownloadError('The data manifest is unexpectedly large.')
        payload=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,UnicodeError,json.JSONDecodeError) as exc:
        raise DownloadError('Cannot read a valid data manifest: '+str(path)) from exc
    return validate_manifest(payload,**kwargs)

def verify_file(path,size,sha256):
    path=Path(path)
    if not path.is_file() or path.stat().st_size!=size:
        raise DownloadError(f'Wrong size or missing file: {path.name}. Existing files are not overwritten; move the file aside or use a new --out directory.')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(CHUNK_BYTES),b''):digest.update(chunk)
    if digest.hexdigest()!=sha256:
        raise DownloadError(f'SHA256 mismatch: {path.name}. Existing files are not overwritten; move the file aside or use a new --out directory.')

def _publish_no_clobber(source,target):
    # Atomic publication without the overwrite race of os.replace/Unix rename.
    # Both files are on the same filesystem; no incomplete final filename appears.
    try:os.link(source,target)
    except FileExistsError as exc:raise DownloadError(f'Refusing to overwrite existing file: {Path(target).name}') from exc
    Path(source).unlink()

def _gh_download(repo,tag,name,directory):
    gh=shutil.which('gh')
    if gh is None:
        raise DownloadError('GitHub CLI (gh) is missing. Install gh and run gh auth login with an account that can read this private repository, or use --assemble-only with existing files.')
    command=[gh,'release','download',tag,'--repo',repo,'--pattern',name,'--dir',str(directory)]
    completed=subprocess.run(command,capture_output=True,text=True)
    if completed.returncode:
        raise DownloadError(f'GitHub download failed for {name}. Run gh auth status and confirm access to the private repository {repo} and release {tag}. No existing file was overwritten.')
    candidate=Path(directory)/name
    if not candidate.is_file():raise DownloadError(f'GitHub did not return the expected asset: {name}')
    return candidate

def assemble(manifest,out):
    """Stream verified parts into a private partial file, then publish atomically."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True,mode=0o700)
    target=out/manifest['file']
    if target.exists():
        verify_file(target,manifest['bytes'],manifest['sha256']);return target
    for part in manifest['parts']:verify_file(out/part['name'],part['bytes'],part['sha256'])
    partial=out/(manifest['file']+'.partial')
    if partial.exists():raise DownloadError(f'Refusing to overwrite an existing partial ZIP: {partial.name}. Move it aside or use another --out directory.')
    created=False
    try:
        with partial.open('xb') as destination:
            created=True;digest=hashlib.sha256();total=0
            for part in manifest['parts']:
                with (out/part['name']).open('rb') as source:
                    for chunk in iter(lambda:source.read(CHUNK_BYTES),b''):
                        destination.write(chunk);digest.update(chunk);total+=len(chunk)
            destination.flush();os.fsync(destination.fileno())
        if total!=manifest['bytes'] or digest.hexdigest()!=manifest['sha256']:
            raise DownloadError('The assembled ZIP failed its size/SHA256 check; no final ZIP was published.')
        _publish_no_clobber(partial,target);created=False
    finally:
        if created:partial.unlink(missing_ok=True)
    return target

def receive(out,*,repo=DEFAULT_REPO,tag=DEFAULT_TAG,manifest_path=None,assemble_only=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True,mode=0o700)
    path=Path(manifest_path) if manifest_path is not None else out/MANIFEST_NAME
    if not path.exists():
        if assemble_only:raise DownloadError('Missing local IHVP_DATA_PARTS.json; --assemble-only never downloads files.')
        if manifest_path is not None:raise DownloadError('The requested local manifest does not exist.')
        with tempfile.TemporaryDirectory(prefix='.ihvp-manifest-',dir=out) as temp:
            candidate=_gh_download(repo,tag,MANIFEST_NAME,temp)
            load_manifest(candidate,repo=repo,tag=tag)
            _publish_no_clobber(candidate,path)
    manifest=load_manifest(path,repo=repo,tag=tag)
    target=out/manifest['file']
    if target.exists():
        verify_file(target,manifest['bytes'],manifest['sha256'])
        print('Existing ZIP passed size and SHA256 checks.');return target
    for index,part in enumerate(manifest['parts'],1):
        local=out/part['name']
        if local.exists():
            verify_file(local,part['bytes'],part['sha256']);print(f'[{index}/{len(manifest["parts"])}] Reusing verified part: {local.name}')
        elif assemble_only:raise DownloadError('Missing local data part: '+local.name)
        else:
            with tempfile.TemporaryDirectory(prefix='.ihvp-part-',dir=out) as temp:
                candidate=_gh_download(repo,tag,part['name'],temp)
                verify_file(candidate,part['bytes'],part['sha256']);_publish_no_clobber(candidate,local)
            print(f'[{index}/{len(manifest["parts"])}] Downloaded and verified: {local.name}',flush=True)
    print('Assembling ZIP in manifest order; this requires space for both parts and ZIP.',flush=True)
    return assemble(manifest,out)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path.cwd()/'ihvp-data')
    parser.add_argument('--repo',default=DEFAULT_REPO)
    parser.add_argument('--tag',default=DEFAULT_TAG)
    parser.add_argument('--manifest',type=Path,help='use an existing local manifest instead of downloading it')
    parser.add_argument('--assemble-only',action='store_true',help='verify and assemble existing files without GitHub access')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',args.repo) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag):
        parser.error('invalid repository or release tag')
    try:
        target=receive(args.out,repo=args.repo,tag=args.tag,manifest_path=args.manifest,assemble_only=args.assemble_only)
    except (DownloadError,OSError) as exc:
        print('Error: '+str(exc),file=sys.stderr);return 1
    print('Verified complete ZIP: '+str(target))
    print('Evaluator-only: this tool does not extract data or place files in a model workspace.')
    return 0

if __name__=='__main__':raise SystemExit(main())
