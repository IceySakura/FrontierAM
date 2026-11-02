"""Publishing checks must still reject malformed artifacts under python -O."""

import subprocess
import sys
import textwrap


def test_optimized_python_rejects_modified_publishable_archives(tmp_path, request):
    script = request.config.rootpath / 'docs/examples/configs/cb-insiders/package_npm.py'
    code = textwrap.dedent("""
        import hashlib, importlib.util, io, json, sys, tarfile
        from pathlib import Path

        path, directory = Path(sys.argv[1]), Path(sys.argv[2])
        spec = importlib.util.spec_from_file_location('cb_package_check', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        archive = directory / 'fixture.deb'
        archive.write_bytes(b'fixture upstream bytes')
        module.DEB_SHA256['x64'] = hashlib.sha256(archive.read_bytes()).hexdigest()
        real_open = tarfile.open
        module.build_package(archive, 'x64', directory / 'valid', 'https://registry.fixture/')

        for corrupt in ('manifest', 'payload', 'members'):
            def modified_archive(name, mode='r', **kwargs):
                tar = real_open(name, mode, **kwargs)
                if mode == 'r':
                    extract = tar.extractfile
                    members = tar.getmembers
                    def modified_extract(member, *args, **kw):
                        if corrupt == 'manifest' and member == 'package/package.json':
                            return io.BytesIO(b'{"name":"wrong-package"}')
                        if corrupt == 'payload' and member == 'package/payload/codebuddy-insiders.deb':
                            return io.BytesIO(b'changed bytes')
                        return extract(member, *args, **kw)
                    tar.extractfile = modified_extract
                    if corrupt == 'members':
                        tar.getmembers = lambda: members() + [members()[0]]
                return tar
            module.tarfile.open = modified_archive
            try:
                module.build_package(archive, 'x64', directory / corrupt, 'https://registry.fixture/')
            except ValueError:
                print(corrupt + ': rejected')
            else:
                raise RuntimeError(corrupt + ' was accepted in optimized mode')
            finally:
                module.tarfile.open = real_open
        """)
    completed = subprocess.run(
        [sys.executable, '-O', '-c', code, str(script), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ['manifest: rejected', 'payload: rejected', 'members: rejected']
