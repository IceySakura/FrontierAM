#!/usr/bin/env python3
"""Baseline submission pipeline: support-weighted gene-tree contraction, NJ/consensus starts,
NNI hill-climbing on outgroup-rooted triplet (quartet) support."""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib2 import (parse, leaves, unrooted_adj, gene_info, nj, compatible, orient,
                  greedy, rf)
from trip import GeneForest, SpeciesTree
import numpy as np
from collections import defaultdict

def main(inp, out):
    random.seed(1)
    lines=[l.strip() for l in open(inp, encoding='utf-8') if l.strip()]
    if not lines: raise ValueError('empty input')
    names=leaves(parse(lines[0]))
    names=['0']+[x for x in names if x!='0'] if '0' in names else names
    n=len(names)
    G=[gene_info(unrooted_adj(parse(l),names),n,names) for l in lines]
    Dp=np.mean([g[3] for g in G],axis=0); Dt=np.mean([g[4] for g in G],axis=0)
    njp=[m for m,_ in nj(Dp)]; njt=[m for m,_ in nj(Dt)]
    F=GeneForest(lines,names,n,sup_thr=0.5,sup_pow=1.0,weight_mode='edge')
    freq=defaultdict(float)
    for g in G:
        for m,L,s in g[2]: freq[m]+= (s if s is not None else 0.333)
    supg=greedy(list(freq.items()),n)
    best=None; bestscore=-1
    for start,extra in [([],njt),([],njp),(supg,njt)]:
        T=SpeciesTree(F,start,extra)
        T.nni_pass()
        if T.score>bestscore: bestscore=T.score; best=T.newick(names)
    # iterated local search from best tree with perturbations
    T=SpeciesTree(F,[],njt)
    open(out,'w',encoding='utf-8').write(best+'\n')

if __name__=='__main__':
    main(sys.argv[1],sys.argv[2])
