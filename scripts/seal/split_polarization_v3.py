"""Held-out public feedback for polarization-demosaicking-v3 (user decision 2026-11-26, results/OPEN_ITEMS.md).

The feedback used to be the five Wen5 scenes shipped to the model, so it measured the samples a submission could fit
(a regressor fit on them scored 43 on feedback and 0 on the private Qiu scenes). Here 16 of the 132 formal scenes
become the feedback set: within each dataset the scenes are ordered by the strong reference's RMSE (then name) and
every eighth from the fifth is taken (Qiu 4 of 32, Wen 12 of 100); the private score uses the other 116.

Usage: split_polarization_v3.py   -> tasks/polarization-demosaicking/versions/v3/data/private/feedback-split.json
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / 'var/stores/store-v63/baselines/polarization-demosaicking-v3'
OUT = ROOT / 'tasks/polarization-demosaicking/versions/v3/data/private/feedback-split.json'


def main() -> None:
    active = (BASELINES / 'ACTIVE').read_text().strip()
    strong = json.loads((BASELINES / active / 'baseline.json').read_text())['metrics']['parts']['strong']['scene_rmse']
    feedback = []
    for dataset in ('qiu', 'wen'):
        scenes = sorted((s for s in strong if s.startswith(dataset + '/')), key=lambda s: (strong[s], s))
        feedback += scenes[4::8]
    scored = sorted(s for s in strong if s not in feedback)
    OUT.write_text(json.dumps({
        'rule': 'within each dataset, scenes ordered by the strong reference RMSE (then name); every eighth from the fifth',
        'source_baseline': active, 'feedback': sorted(feedback), 'scored': scored}, indent=1) + '\n')
    print(json.dumps({'feedback': len(feedback), 'scored': len(scored), 'qiu_feedback': sum(s.startswith('qiu/') for s in feedback)}))


if __name__ == '__main__':
    main()
