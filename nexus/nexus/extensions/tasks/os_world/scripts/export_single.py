"""Write a complete RunInput YAML using the same builder as run-batch."""

from __future__ import annotations

import argparse

from pathlib import Path

import yaml

from nexus.extensions.tasks.os_world.generators.generator_native import SUPPORTED, single_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-family', choices=('gpt', 'claude', 'hunyuan'))
    parser.add_argument('--harness', choices=('bare', 'cc', 'codex'))
    parser.add_argument('--all-modes', action='store_true', help='Write six complete YAMLs into the output directory')
    parser.add_argument('--task-id', required=True)
    parser.add_argument('-o', '--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.all_modes:
        if args.model_family or args.harness:
            parser.error('--all-modes cannot be combined with --model-family/--harness')
        modes = sorted(SUPPORTED)
    else:
        if not args.model_family or not args.harness:
            parser.error('Provide --model-family and --harness, or --all-modes')
        modes = [(args.model_family, args.harness)]
    # Build all before writing: a missing credential cannot leave half the modes updated.
    outputs = []
    for family, harness in modes:
        config = single_config(task_id=args.task_id, model_family=family, harness=harness)
        target = args.output / f'osworld_{family}_{harness}.yaml' if args.all_modes else args.output
        outputs.append((target, config))
    for target, config in outputs:
        # Do not resolve credentials. The run command resolves their references.
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            '# Generated complete RunInput; regenerate after changing task/dataset/provisioning settings.\n'
            + yaml.safe_dump(config.model_dump(mode='json', exclude_none=True), sort_keys=False)
        )
        print(f'Wrote complete RunInput YAML: {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
