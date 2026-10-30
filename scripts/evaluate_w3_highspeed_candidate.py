"""Full causal pipeline comparison against the incumbent on session holdouts."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import run_w3_next_u_ablation as core
import fit_w3_kinematic_compensation as fit
from factr2_next.data_collection.quality import sha256,write_json
from factr2_next.inference.kinematic_compensation import Compensator,ResidualModel


def replay(t,x,y,mu,artifact,envelope):
    pipeline=Compensator(ResidualModel(artifact));bounds=np.deg2rad(np.asarray(envelope))
    history=[];final=[];support=[]
    for stamp,xx,tau,free in zip(t,x,y,mu):
        history.append(xx[:7]);history=history[-50:]
        active=bool(np.all((np.asarray(history)>=bounds[:,0])&(np.asarray(history)<=bounds[:,1])))
        r=pipeline.update(int(stamp),xx[:7],xx[7:14],xx[:7]+xx[14:],tau,free,allow_compensation=active)
        final.append(r['final']);support.append(active and len(history)==50)
    return np.asarray(final),np.asarray(support)


def evaluate(experiment,sigma_dir,base,compensation,output):
    experiment,sigma_dir,base,compensation,output=map(lambda p:Path(p).resolve(),
        (experiment,sigma_dir,base,compensation,output))
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);cfg,manifest=core.verify(experiment)
    root=Path('/home/venom/factr2');oldbase=root/'models/w3/right/frozen_sigma_observer_20261023'
    oldcomp=root/'data/right/kinematic_compensation_20261025_v1'
    old_art=json.loads((oldcomp/'model.json').read_text());new_art=json.loads((compensation/'model.json').read_text())
    old_bounds=json.loads((oldbase/'metadata.json').read_text())['operating_envelope']['position_bounds_deg']
    new_bounds=json.loads((base/'metadata.json').read_text())['operating_envelope']['position_bounds_deg']
    if old_art['base_metadata_sha256']!=sha256(oldbase/'metadata.json') or new_art['base_metadata_sha256']!=sha256(base/'metadata.json'):
        raise ValueError('Mean and residual identities differ')
    results={};waveforms=[];parity=0.
    for role in ('val','test'):
        ds=core.dataset(manifest,cfg,role)
        with np.load(experiment/'evaluation'/f'seed_0_{role}_INCUMBENT.npz') as a:old_mu=a['mean_free_nm']
        with np.load(sigma_dir/'cache'/f'{role}.npz') as a:new_mu=a['mean_free_nm']
        accum={False:dict(old=[],new=[],old_raw=[],new_raw=[],support=[]),True:dict(old=[],new=[],old_raw=[],new_raw=[],support=[])}
        offset=0
        for (x,y),info in zip(ds.episodes,ds.episode_info):
            x=x[49:];y=y[49:];n=len(x);high=str(info.get('speed','')).startswith('continuous_v')
            with h5py.File(info['path']) as h:t=h[info['episode']+'/joint_pos/timestamps'][49:]
            before=old_mu[offset:offset+n];after=new_mu[offset:offset+n];offset+=n
            old_final,_=replay(t,x,y,before,old_art,old_bounds)
            new_final,support=replay(t,x,y,after,new_art,new_bounds)
            # Independent vectorized offline recipe versus step-by-step online
            # class, without envelope gating, on the same full-precision inputs.
            if high and role=='test':
                item=fit.series(t,x,y,after,np.full(n,info['path']))
                _,variants,_,_=fit.metrics(item,np.ones(n,bool),new_art)
                pure,_=replay(t,x,y,after,new_art,[[-10000,10000]]*7)
                parity=max(parity,float(abs(pure-variants['final']).max()))
                np.testing.assert_allclose(pure,variants['final'],atol=1e-10,rtol=1e-10)
                waveforms.append(dict(name=info['speed'],t=(t-t[0])*1e-9,old=old_final,new=new_final,cmd=np.rad2deg(x[:,:7]+x[:,14:]),q=np.rad2deg(x[:,:7])))
            # Both filters/history are initialized independently per episode;
            # exclude a fixed one-second prefix from both schemes.
            keep=(t>=t[0]+1_000_000_000)
            a=accum[high]
            for k,v in dict(old=old_final,new=new_final,old_raw=y-before,new_raw=y-after,support=support).items():a[k].append(v[keep])
        if offset!=len(new_mu) or offset!=len(old_mu):raise ValueError('Evaluation cache alignment')
        results[role]={}
        for high,arrays in accum.items():
            combined={k:np.concatenate(v) for k,v in arrays.items()}
            results[role]['high' if high else 'old']=dict(
                metrics={k:core.error_metrics(v) for k,v in combined.items() if k!='support'},
                position_support_fraction=float(combined['support'].mean()))
    c=json.loads((experiment/'deployment_criteria.json').read_text());test=results['test'];passed=True;per_joint={}
    for scope in ('old','high'):
        before=test[scope]['metrics']['old'];after=test[scope]['metrics']['new']
        limit=(1-c['minimum_final_highspeed_test_mse_improvement_fraction']) if scope=='high' else c['maximum_final_old_test_mse_ratio']
        passed &= after['physical_mse_nm2']<=limit*before['physical_mse_nm2']
        b,a=map(np.asarray,(before['per_joint_rmse_nm'],after['per_joint_rmse_nm']))
        per_joint[scope]=(a<=b+np.maximum(.2*b,.02)).tolist();passed &= bool(np.all(per_joint[scope]))
    passed &= test['high']['position_support_fraction']>=c['minimum_highspeed_test_position_support_fraction']
    report=dict(passed=bool(passed),splits=results,per_joint_pass=per_joint,
        stepwise_vs_vectorized_max_error_nm=parity,criteria=c,
        interpretation='Reported no-load joint torque residual; not independently measured external torque',
        source_models=dict(old_mean=sha256(oldbase/'metadata.json'),new_mean=sha256(base/'metadata.json'),
                           old_compensation=sha256(oldcomp/'model.json'),new_compensation=sha256(compensation/'model.json')))
    write_json(output/'comparison.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(7,len(waveforms),figsize=(16,17),sharex='col',squeeze=False)
    for col,w in enumerate(waveforms):
        for j in range(7):
            ax=axes[j,col];ax.plot(w['t'],w['old'][:,j],lw=.6,label='incumbent final')
            ax.plot(w['t'],w['new'][:,j],lw=.6,label='new final');ax.set_ylabel(f'J{j+1} (Nm)');ax.grid(alpha=.2)
        axes[0,col].set_title(w['name']);axes[-1,col].set_xlabel('seconds');axes[0,col].legend()
    fig.tight_layout();fig.savefig(output/'heldout_waveforms.png',dpi=140);plt.close(fig)
    lines=['# 高速独立会话比较','','|数据|关节|旧窗口输出 RMS (Nm)|新窗口输出 RMS (Nm)|','|---|---|---:|---:|']
    for scope in ('old','high'):
        for j in range(7):lines.append(f'|{scope}|J{j+1}|{test[scope]["metrics"]["old"]["per_joint_rmse_nm"][j]:.5f}|{test[scope]["metrics"]["new"]["per_joint_rmse_nm"][j]:.5f}|')
    lines+=['',f'预先固定的部署检查：{bool(passed)}；在线逐帧类与离线向量计算最大差 {parity:.3g} Nm。',
            '仅评价空载报告力矩残差，未测真实接触力矩。输出滤波会延迟并削弱短脉冲。测试未参与拟合或选模。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('experiment','sigma-dir','base','compensation','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=evaluate(a.experiment,a.sigma_dir,a.base,a.compensation,a.output)
    print(json.dumps(dict(passed=r['passed'])),flush=True)
    if not r['passed']:raise SystemExit('Keep incumbent: full-pipeline gate failed; do not tune on this test')
