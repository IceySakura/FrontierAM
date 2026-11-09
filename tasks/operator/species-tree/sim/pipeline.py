"""Simulate species-tree instances in the S100 style (rebuilt by FrontierAM; results/SPECIES_TREE_SCALE.md).

One instance: SimPhy draws a species tree (birth process, n ingroup species plus one outgroup) and k gene trees under the
multispecies coalescent; AliSim (IQ-TREE 2) evolves a sequence of the given length along each gene tree under a GTR+G
model with per-gene parameters; IQ-TREE estimates each gene tree with approximate Bayes (aBayes) supports. Species get
the task's labels: outgroup 0, ingroup 1..n in random order independent of the phylogeny.

Writes genes.tre (estimated gene trees, one per line), truth.tre (species tree) and stats.json (AD: mean normalised RF
between true gene trees and the species tree; GTEE: mean normalised RF between estimated and true gene trees).

Usage: pipeline.py OUT --taxa N --genes K --seed S [--height G] [--pop N] [--rate R] [--length L] [--threads 1]
"""

import argparse
import json
import random
import re
import shutil
import subprocess
import time

from pathlib import Path

TOOLS = Path('/data/tools/phylo')
SIMPHY = TOOLS / 'SimPhy_1.0.2/bin/simphy_lnx64'
IQTREE = TOOLS / 'iqtree-2.3.6-Linux-intel/bin/iqtree2'
LEAF = re.compile(r'([(,])([^(),:;]+)(?=:)')


def splits(newick: str) -> set[frozenset]:
    """Non-trivial bipartitions of an unrooted tree, each as the side without the smallest label."""
    bare = re.sub(r'\)[^,();]*', ')', re.sub(r':[^,();]+', '', newick.strip().rstrip(';')))
    stack, clades, labels, token = [[]], [], [], ''
    for c in bare + ',':
        if c == '(':
            stack.append([])
        elif c in ',)':
            if token:
                stack[-1].append(token)
                labels.append(token)
                token = ''
            if c == ')':
                clade = stack.pop()
                clades.append(frozenset(clade))
                stack[-1].extend(clade)
        elif not c.isspace():
            token += c
    everyone = frozenset(labels)
    anchor = min(everyone)
    result = set()
    for clade in clades:
        side = clade if anchor not in clade else everyone - clade
        if 1 < len(side) < len(everyone) - 1:
            result.add(side)
    return result


def nrf(a: str, b: str) -> float:
    sa, sb = splits(a), splits(b)
    n = len(set(re.findall(r'[(,]([^(),:;]+)', a)))
    return len(sa - sb) / (n - 3)


def relabel(newick: str, mapping: dict[str, str]) -> str:
    return LEAF.sub(lambda m: m.group(1) + mapping[m.group(2).split('_')[0]], newick)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--taxa', type=int, default=100, help='ingroup species')
    parser.add_argument('--genes', type=int, default=50)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--height', type=float, default=2e6, help='species tree height in generations')
    parser.add_argument('--pop', type=float, default=2e5, help='effective population size')
    parser.add_argument('--rate', type=float, default=4e-8, help='mean substitution rate per site per generation')
    parser.add_argument('--length', type=int, default=200)
    parser.add_argument('--fast', action='store_true', help='IQ-TREE --fast tree search')
    parser.add_argument('--ad-only', action='store_true', help='simulate true trees only and report AD (calibration)')
    args = parser.parse_args()
    out = args.out
    work = out / 'work'
    shutil.rmtree(out, ignore_errors=True)
    work.mkdir(parents=True)
    rng = random.Random(args.seed)
    started = time.time()
    subprocess.run([str(SIMPHY), '-rs', '1', '-rl', f'f:{args.genes}', '-sl', f'f:{args.taxa}', '-so', 'f:1',
                    '-sb', 'f:0.0000001', '-st', f'f:{args.height:g}', '-sp', f'f:{args.pop:g}',
                    '-su', f'ln:{__import__("math").log(args.rate):.6f},0.1', '-hh', 'f:1', '-hs', 'ln:1.5,1',
                    '-hl', 'ln:1.2,1', '-hg', 'f:200', '-cs', str(args.seed), '-v', '0', '-o', 'sim', '-ot', '0',
                    '-op', '1'], cwd=work, check=True, capture_output=True)
    run = work / 'sim/1'
    species = (run / 's_tree.trees').read_text().strip()
    names = sorted(set(re.findall(r'[(,]([^(),:;]+):', species)), key=lambda s: int(s) if s.isdigit() else -1)
    outgroup = '0'
    if outgroup not in names:
        raise SystemExit(f'SimPhy outgroup label not found among {names[:5]}')
    ingroup = [s for s in names if s != outgroup]
    labels = list(range(1, len(ingroup) + 1))
    rng.shuffle(labels)
    mapping = {outgroup: '0', **{s: str(v) for s, v in zip(ingroup, labels)}}
    truth = relabel(species, mapping)
    (out / 'truth.tre').write_text(truth + '\n')
    true_genes, estimated = [], []
    for g in range(1, args.genes + 1):
        gene = relabel((run / f'g_trees{g:0{len(str(args.genes))}d}.trees').read_text().strip(), mapping)
        true_genes.append(gene)
        if args.ad_only:
            continue
        tree = work / f'g{g}.tre'
        tree.write_text(gene + '\n')
        rates = [round(rng.gammavariate(2, 1), 4) for _ in range(5)]
        freqs = [rng.gammavariate(10, 1) for _ in range(4)]
        freqs = [round(f / sum(freqs), 4) for f in freqs]
        freqs[-1] = round(1 - sum(freqs[:-1]), 4)
        alpha = round(rng.lognormvariate(-0.47, 0.27), 4)
        model = f'GTR{{{",".join(map(str, rates))}}}+F{{{",".join(map(str, freqs))}}}+G4{{{alpha}}}'
        subprocess.run([str(IQTREE), '--alisim', str(work / f'a{g}'), '-t', str(tree), '-m', model, '--length',
                        str(args.length), '--seed', str(args.seed * 1000 + g), '-af', 'phy', '-quiet'],
                       check=True, capture_output=True)
        command = [str(IQTREE), '-s', str(work / f'a{g}.phy'), '-m', 'GTR+G4', '--abayes', '-nt', '1',
                   '--seed', str(args.seed * 1000 + g), '-pre', str(work / f'e{g}'), '-quiet', '-redo']
        if args.fast:
            command.append('--fast')
        subprocess.run(command, check=True, capture_output=True)
        # IQ-TREE 2.3 writes an aBayes-only support as '/0.99'; the task's inputs carry the bare value.
        estimated.append(re.sub(r'\)/', ')', (work / f'e{g}.treefile').read_text().strip()))
    ad = sum(nrf(truth, t) for t in true_genes) / len(true_genes)
    if args.ad_only:
        print(json.dumps({'taxa': args.taxa, 'seed': args.seed, 'height': args.height, 'pop': args.pop, 'AD': ad}))
        shutil.rmtree(out)
        return
    (out / 'genes.tre').write_text('\n'.join(estimated) + '\n')
    gtee = sum(nrf(t, e) for t, e in zip(true_genes, estimated)) / len(true_genes)
    est_vs_species = sum(nrf(truth, e) for e in estimated) / len(estimated)
    stats = {'taxa': args.taxa, 'genes': args.genes, 'seed': args.seed, 'height': args.height, 'pop': args.pop,
             'rate': args.rate, 'length': args.length, 'fast': args.fast, 'AD': ad, 'GTEE': gtee,
             'estimated_vs_species': est_vs_species, 'seconds': round(time.time() - started, 1)}
    (out / 'stats.json').write_text(json.dumps(stats, indent=1) + '\n')
    shutil.rmtree(work)
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
