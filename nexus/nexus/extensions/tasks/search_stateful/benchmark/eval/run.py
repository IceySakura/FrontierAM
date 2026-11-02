"""One-click HLE evaluation runner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess

from collections import defaultdict
from pathlib import Path


class EvalRunner:
    def __init__(
        self,
        dataset: str,
        dataset_format: str,
        out_path: str,
        database: str | None,
        max_concurrent: int,
        config_path: str | None,
    ) -> None:
        self.dataset = Path(dataset)
        self.dataset_format = dataset_format
        self.out_path = Path(out_path)
        self.database = Path(database) if database else self._default_database_path(dataset_format)
        self.max_concurrent = max_concurrent
        self.generator_path = Path(__file__).resolve().parent / 'generator.py'
        self.config_dir = Path(__file__).resolve().parent / 'config'
        self.config_path = Path(config_path) if config_path else (self.config_dir / 'base.yaml')

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='One-click HLE evaluation runner')
        parser.add_argument('--dataset', required=True, help='Dataset path (HLE parquet dir/file or naive jsonl)')
        parser.add_argument('--format', default='hle', choices=['hle', 'naive'])
        parser.add_argument('--out', default='outputs/hle_results.jsonl')
        parser.add_argument('--database', default=None)
        parser.add_argument('--max-concurrent', type=int, default=4)
        parser.add_argument(
            '--config', default=None, help='Path to base config yaml (defaults to eval/config/base.yaml)'
        )
        return parser.parse_args()

    def run(self) -> None:
        if not self.dataset.exists():
            raise FileNotFoundError(f'Dataset not found: {self.dataset}')
        if not self.config_path.exists():
            raise FileNotFoundError(f'Config not found: {self.config_path}')

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env['SCIENCE_INPUT_PATH'] = str(self.dataset)
        env['HLE_DATASET_FORMAT'] = self.dataset_format
        env['SCIENCE_CONFIG'] = str(self.config_path)
        self._validate_code_interpreter(env)

        cmd = [
            'uv',
            'run',
            '--extra',
            'extensions',
            *self._with_packages(),
            'nexus',
            'run-batch',
            '--generator',
            str(self.generator_path),
            '--database',
            str(self.database),
            '--export-path',
            str(self.out_path),
            '--export-format',
            'jsonl',
            '--max-concurrent',
            str(self.max_concurrent),
        ]
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f'[eval] subprocess exited with code {e.returncode}')
        self.summarize_results(self.out_path)

    def _with_packages(self) -> list[str]:
        requirements_path = self.config_dir.parents[1] / 'docker' / 'uv-requirements.txt'
        if not requirements_path.exists():
            return []
        specs: list[str] = []
        for line in requirements_path.read_text(encoding='utf-8').splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            specs.extend(['--with', stripped])
        return specs

    @staticmethod
    def _default_database_path(dataset_format: str) -> Path:
        safe_format = dataset_format.replace('/', '_')
        return Path('outputs') / f'hle_results_{safe_format}.db'

    @staticmethod
    def _validate_code_interpreter(env: dict[str, str]) -> None:
        if env.get('CODE_INTERPRETER') is None:
            raise ValueError('CODE_INTERPRETER is required. Set to "none", "python", or "jupyter".')

    @staticmethod
    def summarize_results(path: Path) -> None:
        if not path.exists():
            print(f'[eval] results not found: {path}')
            return

        total = success = failed = 0
        scored_samples = 0
        score_sum = 0.0
        prompt_sum: dict[str, float] = defaultdict(float)
        prompt_count: dict[str, int] = defaultdict(int)

        with path.open('r', encoding='utf-8', errors='replace') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total += 1
                status = record.get('status')
                success += int(status == 'success')
                failed += int(status == 'failed')

                task_output = record.get('task_output') or {}
                prompt_id = task_output.get('task_id') or record.get('run_id')
                if not isinstance(prompt_id, str) or not prompt_id:
                    continue

                score = task_output.get('score')
                if isinstance(score, str):
                    try:
                        score = float(score)
                    except ValueError:
                        continue
                if not isinstance(score, (int, float)):
                    continue

                score = float(score)
                scored_samples += 1
                score_sum += score
                prompt_sum[prompt_id] += score
                prompt_count[prompt_id] += 1

        micro_avg_score = (score_sum / scored_samples) if scored_samples else 0.0
        scored_sample_coverage = (scored_samples / total) if total else 0.0
        scored_prompts = len(prompt_count)
        macro_avg_score = (
            sum(prompt_sum[k] / prompt_count[k] for k in prompt_count) / scored_prompts if scored_prompts else 0.0
        )
        avg_samples_per_prompt = (scored_samples / scored_prompts) if scored_prompts else 0.0

        print('[eval] summary')
        print(f'  total: {total}')
        print(f'  success: {success}')
        print(f'  failed: {failed}')
        print(f'  prompts: {scored_prompts}')
        print(f'  scored_samples: {scored_samples} ({scored_sample_coverage:.2%})')
        print(f'  scored_prompts: {scored_prompts} (avg_samples_per_prompt={avg_samples_per_prompt:.2f})')
        print(f'  avg_score_micro: {micro_avg_score:.6f} (avg over samples)')
        print(f'  avg_score_macro: {macro_avg_score:.6f} (avg over prompts)')


def main() -> None:
    args = EvalRunner.parse_args()
    runner = EvalRunner(
        dataset=args.dataset,
        dataset_format=args.format,
        out_path=args.out,
        database=args.database,
        max_concurrent=args.max_concurrent,
        config_path=args.config,
    )
    runner.run()


if __name__ == '__main__':
    main()
