from __future__ import annotations

import subprocess
import sys


def test_importing_cli_does_not_require_tdsql_driver() -> None:
    script = """
import sys


class BlockDatabaseDrivers:
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname in {'asyncmy', 'aiomysql'}:
            raise ModuleNotFoundError(fullname)
        return None


sys.meta_path.insert(0, BlockDatabaseDrivers())
import nexus.cli.main

assert 'nexus.runner.trace.tdsql._driver' not in sys.modules
assert 'nexus.runner.trace.tdsql.writer' not in sys.modules
"""

    subprocess.run([sys.executable, '-c', script], check=True)
