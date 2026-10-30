"""Same-record heldout mean comparison, no correction and no fitting."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import run_w3_next_u_ablation as core
from factr2_next.data_collection.quality import sha256,write_json


def analyze(experiment,base,output):
    experiment,base,output=map(lambda p:Path(p).resolve(),(experiment,base,output))
    output.mkdir(exist_ok=False)
    cfg,manifest=core.verify(experiment);ds=core.dataset(manifest,cfg,'test')
    with np.load(experiment/'evaluation/seed_0_test_INCUMBENT.npz') as a:before=a['error_nm']
    with np.load(experiment/'evaluation/seed_0_test_NEXT_MSE.npz') as a:after=a['error_nm']
    groups={k:dict(before=[],after=[]) for k in ('historical_low','previous_high','new_wide_high')}
    offset=0;waves=[]
    for (x,_),info in zip(ds.episodes,ds.episode_info):
        n=len(x)-49;a,b=before[offset:offset+n],after[offset:offset+n];offset+=n
        high=str(info.get('speed','')).startswith('continuous_v')
        key='new_wide_high' if Path(info['path']).parent==experiment else ('previous_high' if high else 'historical_low')
        groups[key]['before'].append(a);groups[key]['after'].append(b)
        if key=='new_wide_high':
            with h5py.File(info['path']) as h:t=h[info['episode']+'/joint_pos/timestamps'][49:]
            waves.append(dict(name=info['speed'],t=(t-t[0])*1e-9,before=a,after=b))
    if offset!=len(before) or offset!=len(after):raise ValueError('Cache/window alignment mismatch')
    result={}
    for key,arrays in groups.items():
        metrics={k:core.error_metrics(np.concatenate(v)) for k,v in arrays.items()}
        r=metrics['after']['physical_mse_nm2']/metrics['before']['physical_mse_nm2']
        result[key]=dict(metrics=metrics,rms_reduction_fraction=1-float(np.sqrt(r)))
    report=dict(extra_compensation_enabled=False,test_used_for_selection=False,
        same_record_comparison=True,scopes=result,
        new_metadata_sha256=sha256(base/'metadata.json'),
        limitations='No-load model residual, not independent contact torque ground truth; raw mean errors before display filtering')
    write_json(output/'comparison.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(7,len(waves),figsize=(15,15),sharex='col',squeeze=False)
    for col,w in enumerate(waves):
        for j in range(7):
            a=axes[j,col];a.plot(w['t'],w['before'][:,j],lw=.65,color='#999999',label='Previous NEXT mean (no extra correction)')
            a.plot(w['t'],w['after'][:,j],lw=.65,color='#1670ca',label='New NEXT mean (no extra correction)')
            a.axhline(0,color='black',lw=.4);a.set_ylabel(f'J{j+1} Nm');a.grid(alpha=.2)
        axes[0,col].set_title(w['name']);axes[0,col].legend(fontsize=7);axes[-1,col].set_xlabel('s')
    fig.suptitle('Independent new wide-pose test | identical recordings | raw external joint torque residual')
    fig.tight_layout();fig.savefig(output/'heldout_waveforms.png',dpi=140);plt.close(fig)
    lines=['# 同一独立测试记录上的新旧模型比较','',
        '全部关闭额外速度/加速度补偿，使用完整精度原始残差；两模型使用完全相同的独立测试输入，不把不同实机轨迹的波形直接相互比较。',
        '|测试范围|旧模型整体 RMS N·m|新模型整体 RMS N·m|RMS 降幅|','|---|---:|---:|---:|']
    for key,v in result.items():
        before=np.sqrt(v['metrics']['before']['physical_mse_nm2']);after=np.sqrt(v['metrics']['after']['physical_mse_nm2'])
        lines.append(f'|{key}|{before:.5f}|{after:.5f}|{100*v["rms_reduction_fraction"]:.2f}%|')
    lines+=['','本表为本轮选模结束后的独立评价，不用于继续调整本轮模型。空载残差不是有独立传感器验证的真实接触精度。',
        '',f'![独立记录原始残差对比]({output/"heldout_waveforms.png"})']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('experiment','base','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(analyze(a.experiment,a.base,a.output),indent=2))
