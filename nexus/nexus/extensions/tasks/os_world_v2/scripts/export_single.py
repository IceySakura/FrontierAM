"""Export complete V2 YAMLs without using the V1 generator."""

import argparse

from pathlib import Path

import yaml

from nexus.extensions.tasks.os_world_v2.generators.generator_native import SUPPORTED, single_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-modes', action='store_true')
    parser.add_argument('--model-family')
    parser.add_argument('--harness')
    parser.add_argument('--task-id', required=True)
    parser.add_argument('-o', '--output', type=Path, required=True)
    args = parser.parse_args()
    modes = sorted(SUPPORTED) if args.all_modes else [(args.model_family, args.harness)]
    outputs = [
        (family, harness, single_config(task_id=args.task_id, model_family=family, harness=harness))
        for family, harness in modes
    ]
    for family, harness, config in outputs:
        path = args.output / f'osworld_v2_{family}_{harness}.yaml' if args.all_modes else args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config.model_dump(mode='json', exclude_none=True), sort_keys=False))


if __name__ == '__main__':
    main()
