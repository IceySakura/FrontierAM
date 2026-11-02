from __future__ import annotations

import sys

from tests.integration.distributed.run_k8s_predefined_cluster import main as run_predefined_cluster


def main(argv: list[str] | None = None) -> int:
    return run_predefined_cluster(['--tdsql-batch-e2e', '--skip-opus48', *(argv if argv is not None else sys.argv[1:])])


if __name__ == '__main__':
    raise SystemExit(main())
