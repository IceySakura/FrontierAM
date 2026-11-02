"""Export V1-compatible platform rows from the actual published V2 dataset index."""

import argparse
import json

from pathlib import Path

from nexus.extensions.tasks.os_world_v2.data.index import export_platform_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--exclude-missing-dependencies', action='store_true')
    args = parser.parse_args()
    count = export_platform_jsonl(args.output, exclude_missing_dependencies=args.exclude_missing_dependencies)
    print(json.dumps({'output': str(args.output), 'rows': count}))


if __name__ == '__main__':
    main()
