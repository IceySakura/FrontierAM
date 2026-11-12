"""Independent dense mathematics, fresh split design, and v4 privacy contracts.

Synthetic tiny arrays only. Run outside timing experiments.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
VERSION=ROOT/'tasks/ihvp-sequence/versions/v4'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m);return m


@pytest.fixture(scope='module')
def modules():
    model=load('v4_test_model',VERSION/'curator/model.py')
    with patch.dict(sys.modules,{'model':model}):
        generator=load('v4_test_generator',VERSION/'curator/generate.py')
    independent=load('v4_test_independent',ROOT/'scripts/analyze/ihvp_routed_diagnostics.py')
    return model,generator,independent


def rotation(n,p,angles):
    matrix=np.eye(n)
    for (i,j),theta in zip(np.asarray(p).reshape(-1,2),angles):
        c,s=np.cos(theta),np.sin(theta)
        matrix[i,i],matrix[i,j],matrix[j,i],matrix[j,j]=c,-s,s,c
    return matrix


@pytest.mark.parametrize('architecture',['routed_only','routed_dct2','routed_dct3'])
def test_routed_factors_against_explicit_dense_construction(modules,architecture):
    model,generator,independent=modules;n=64
    config={'id':'tiny','split':'selftest','seed':9891,'profile':'unit',
            'condition_target':1e5,'settings':{**generator.FAMILIES[0],'architecture':architecture}}
    a,_=generator.generate_arrays(config,dimension=n,steps=4)
    base=np.eye(n)
    for p,angles in zip(a['base_routing'],a['base_angles']):
        base=rotation(n,p,angles)@base
    dct=np.sqrt(2/n)*np.cos(np.pi/n*np.arange(n)[:,None]*(np.arange(n)[None,:]+.5));dct[0]/=np.sqrt(2)
    for p,signs in zip(a['base_permutations'],a['base_signs']):
        base=dct@np.diag(signs)@np.eye(n)[p]@base
    base=np.diag(a['post_signs'])@base
    case=model.HardCase(a);other=independent.IndependentHardCase(a)
    for t in (0,1,3):
        q=base@rotation(n,a['temporal_permutations'][1],a['temporal_angles'][t,1])@rotation(n,a['temporal_permutations'][0],a['temporal_angles'][t,0])
        np.testing.assert_allclose(q.T@q,np.eye(n),atol=1e-12)
        block=np.zeros((n,n))
        for (i,j),(u,v,w) in zip(a['block_permutation'].reshape(-1,2),a['block_coefficients'][t]):
            block[i,i],block[i,j],block[j,i],block[j,j]=u,v,v,w
        f=block@np.diag(np.sqrt(a['diagonal'][t]))@q.T;h=f.T@f
        np.testing.assert_allclose(case.hvp(t,np.eye(n)),h,rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(other.hvp(t,np.eye(n)),h,rtol=1e-10,atol=1e-8)
        exact=np.linalg.solve(h,a['b'][t]);error=exact-case.reference(t)
        assert error@h@error/(a['whitened_rhs'][t]@a['whitened_rhs'][t])<1e-16
        assert np.linalg.eigvalsh(h)[0]>=case.mu*(1-1e-10)
        assert np.linalg.eigvalsh(h)[-1]<=case.L*(1+1e-10)
        assert case.score(t,case.reference(t))['energy_loss']==0
        assert abs(case.score(t,np.zeros(n))['energy_loss']-1)<1e-12


def test_fresh_15_30_and_no_fixed_family_architecture_mapping(modules,tmp_path):
    _,g,_=modules
    receipt={'status':'FROZEN','design_sha256':g.design_hash(),
             'model_sha256':g.digest(VERSION/'curator/model.py'),
             'generator_sha256':g.digest(VERSION/'curator/generate.py'),
             'public_qualification_sha256':'synthetic-test-only','benchmark_sha256':'synthetic-test-only'}
    frozen=tmp_path/'freeze.json';frozen.write_text(json.dumps(receipt))
    public=g.frozen_plan(tmp_path/'public','public',16384,50)
    private=g.frozen_plan(tmp_path/'private','private',16384,50,frozen)
    assert len(public['cases'])==15 and len(private['cases'])==30
    assert not {x['seed'] for x in public['cases']} & {x['seed'] for x in private['cases']}
    for cohort in (public,private):
        for family in g.FAMILIES:
            for replicate in {x['replicate'] for x in cohort['cases']}:
                cells=[x for x in cohort['cases'] if x['settings']['name']==family['name'] and x['replicate']==replicate]
                assert len(cells)==3
                assert {x['settings']['architecture'] for x in cells}==set(g.ARCHITECTURES)
    # No receipt, including a resumed private plan, must not bypass the gate.
    with pytest.raises(ValueError,match='requires'):
        g.frozen_plan(tmp_path/'private','private',16384,50)


def test_learner_zip_is_exact_six_file_allowlist(tmp_path):
    build=load('v4_test_learner_builder',ROOT/'scripts/seal/build_ihvp_v4_model_visible.py')
    import zipfile
    out=tmp_path/'learner.zip';result=build.build(out)
    assert result['file_count']==6
    with zipfile.ZipFile(out) as archive:
        assert set(archive.namelist())==set(build.ALLOWLIST)
        assert not any(name.endswith('.npz') or 'curator' in name or '.git' in name for name in archive.namelist())
        assert json.loads(archive.read('samples/smoke.json'))


def test_version_specific_feedback_is_aggregate_only():
    from arena.public_v63 import config
    assert config('ihvp-sequence-v4')['category'] is None
    assert config('ihvp-sequence-v4')['private_feedback']=='none_before_final_freeze'
    assert config('ihvp-sequence-v3')['category']=='family'
    spec=json.loads((VERSION/'task.json').read_text())
    assert spec['status']=='draft' and spec['data']['retained_scoring_sequences']==0


def test_sample_selection_needs_public_only_and_fixed_endpoints():
    prepare=load('v4_test_prepare',ROOT/'scripts/seal/prepare_ihvp_hard_v4.py')
    public={'cases':[{'id':f'public_{i}','sha256':str(i)*64} for i in range(15)]}
    sample=prepare.samples(public)
    assert [x['source_public_feedback_id'] for x in sample['cases']]==['public_0','public_14']
    assert sample['split']=='samples' and all('path' not in x for x in sample['cases'])


def test_v4_driver_rejects_legacy_data_and_stays_unmeasured(tmp_path,monkeypatch):
    from arena.operator_drivers.ihvp_hard_v4 import IHVPHardV4
    spec=json.loads((VERSION/'task.json').read_text());task=IHVPHardV4(spec)
    assert task.grader_version=='v4'
    assert task.score({}, {}, {})['status']=='unmeasured'
    with pytest.raises(ValueError):task.data_dir('public_all')
    with pytest.raises(ValueError):task.data_dir('samples')


def test_importer_rejects_path_traversal_and_symlink(tmp_path):
    prepare=load('v4_test_prepare_paths',ROOT/'scripts/seal/prepare_ihvp_hard_v4.py')
    root=tmp_path/'data';root.mkdir()
    with pytest.raises(ValueError):prepare.contained(root,root/'..'/'private')
    outside=tmp_path/'outside';outside.mkdir();(root/'alias').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):prepare.contained(root,root/'alias'/'teacher.npz')


def test_private_freeze_refuses_incomplete_public_evidence(tmp_path):
    # Import in an isolated model namespace; no actual dataset/seed is read.
    model=load('v4_test_freeze_model',VERSION/'curator/model.py')
    with patch.dict(sys.modules,{'model':model}):
        original=list(sys.path)
        try:
            module=load('v4_test_freeze',ROOT/'scripts/seal/freeze_ihvp_v4.py')
        finally:sys.path[:]=original
    qualification=tmp_path/'qualification.json';qualification.write_text(json.dumps({'status':'PASS','split':'private'}))
    benchmark=tmp_path/'aggregate.json';benchmark.write_text('{}')
    (tmp_path/'run_manifest.json').write_text('{}')
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='public'):
        module.freeze(SimpleNamespace(qualification=qualification,benchmark=benchmark,out=tmp_path/'freeze.json'))
    assert not (tmp_path/'freeze.json').exists()


def test_target_sealer_public_tar_has_only_whitelist(tmp_path):
    original=list(sys.path)
    try:
        sys.path.insert(0,str(ROOT/'scripts/seal'))
        seal=load('v4_test_target_seal',ROOT/'scripts/seal/seal_ihvp_v4.py')
    finally:sys.path[:]=original
    import tarfile
    tar=tmp_path/'public.tar';seal.public_material(tar)
    with tarfile.open(tar) as archive:
        assert set(archive.getnames())==set(seal.ALLOWLIST)
        assert len(archive.getnames())==6
        assert all(m.isfile() and m.mode==0o444 for m in archive.getmembers())
