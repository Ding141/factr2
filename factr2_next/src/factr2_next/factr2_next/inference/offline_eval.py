"""Whole-episode physical-unit evaluation; no implicit 1000-row truncation."""
import argparse
import csv
from pathlib import Path
import time
import h5py
import numpy as np
import torch
import yaml
from factr2_next.data_collection.quality import require, validate_manifest, write_json, sha256
from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.inference.history_buffer import HistoryBuffer
from factr2_next.w3_config import SIGNALS


def read_episode(h5_path, episode, arm=None, max_steps=None, keys=None):
    keys=keys or {k:(f'{arm}_'+k if arm else k) for k in SIGNALS}
    names={k:v.format(arm=arm) for k,v in keys.items()}
    with h5py.File(h5_path,'r') as h:
        ep=h[episode]
        out={k:np.asarray(ep[name]['data'],dtype=np.float32) for k,name in names.items()}
    n=len(out['joint_pos'])
    if any(a.ndim!=2 or len(a)!=n or a.shape!=out['joint_pos'].shape or not np.isfinite(a).all() for a in out.values()):
        raise ValueError('evaluation_shape_finite_lengths')
    if max_steps is not None:
        out={k:v[:max_steps] for k,v in out.items()}
    return out


def predict_sequence(loaded, arrays):
    buf=HistoryBuffer(loaded.history);pred=[];measured=[];norm=loaded.normalization
    with torch.inference_mode():
        for q,v,c,tau in zip(*(arrays[k] for k in SIGNALS)):
            buf.append(q,v,c)
            if not buf.ready:continue
            x=(buf.array()-norm['x_mean'])/norm['x_std']
            y=loaded.model(torch.from_numpy(x).unsqueeze(0).to(loaded.device)).cpu().numpy()[0]
            pred.append((y*norm['y_std']+norm['y_mean']).astype(np.float32));measured.append(tau)
    if not pred:raise ValueError('no_predictions_history')
    return np.stack(pred),np.stack(measured)


def physical_metrics(measured, pred):
    residual=np.asarray(measured,dtype=np.float64)-np.asarray(pred,dtype=np.float64)
    if residual.ndim!=2 or not len(residual) or not np.isfinite(residual).all():raise ValueError('metrics_input')
    return dict(samples=len(residual),overall_rmse_nm=float(np.sqrt(np.mean(residual**2))),
        per_joint_mse_nm2=np.mean(residual**2,axis=0).tolist(),
        per_joint_rmse_nm=np.sqrt(np.mean(residual**2,axis=0)).tolist(),
        signed_bias_nm=residual.mean(axis=0).tolist(),residual_std_nm=residual.std(axis=0).tolist(),
        abs_residual_p95_nm=np.percentile(abs(residual),95,axis=0).tolist(),
        abs_residual_p99_nm=np.percentile(abs(residual),99,axis=0).tolist())


def plots(out, label, residual, timestamps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    t=(timestamps-timestamps[0])*1e-9
    fig,axes=plt.subplots(2,1,figsize=(10,7))
    for i in range(residual.shape[1]):
        axes[0].plot(t,residual[:,i],label=f'joint_{i}')
    axes[0].set(xlabel='Time (s)',ylabel='Measured - predicted (Nm)');axes[0].legend(ncol=4)
    # Each call is a single accepted continuous episode. Never concatenate spectra.
    dt=np.mean(np.diff(t)) if len(t)>1 else .02
    freq=np.fft.rfftfreq(len(t),dt)
    amp=abs(np.fft.rfft(residual-residual.mean(axis=0),axis=0))*2/len(t)
    for i in range(residual.shape[1]):axes[1].plot(freq,amp[:,i])
    axes[1].set(xlabel='Frequency (Hz)',ylabel='Residual amplitude (Nm)',xlim=(0,25))
    fig.tight_layout();fig.savefig(out/(label+'.png'));plt.close(fig)


def acceptance(report, residual, cfg, train_std):
    required=('per_joint_rmse_nm','per_joint_abs_bias_nm','baseline_improvement','sigma_floor_nm',
              'free_contact_false_positive_time_rate','contact_high','contact_low','contact_scale_nm','filter_alpha')
    def incomplete(v):
        return v is None or (isinstance(v,list) and (len(v)!=7 or any(x is None for x in v)))
    if not cfg.get('version') or any(incomplete(cfg.get(k)) for k in required):
        return {'result':'UNCONFIGURED','reason':'Freeze all physical thresholds before final test; null cannot PASS'}
    for k in ('per_joint_rmse_nm','per_joint_abs_bias_nm','sigma_floor_nm'):
        if len(cfg[k])!=7 or any(not np.isfinite(x) or x<0 for x in cfg[k]):raise ValueError('acceptance_threshold:'+k)
    if (not 0<cfg['filter_alpha']<=1 or not 0<=cfg['contact_low']<=cfg['contact_high']
            or cfg['contact_scale_nm']<=0):raise ValueError('acceptance_contact_config')
    y=None;contact=False;hits=0
    # Caller passes episode list; reset filters and hysteresis at each boundary.
    count=0
    for episode_residual in residual:
        y=None;contact=False
        for row in episode_residual:
            y=row.copy() if y is None else cfg['filter_alpha']*row+(1-cfg['filter_alpha'])*y
            magnitude=float(np.sum(abs(y))/cfg['contact_scale_nm'])
            contact=(magnitude>cfg['contact_low']) if contact else (magnitude>=cfg['contact_high'])
            hits+=int(contact);count+=1
    fp=hits/max(count,1)
    model=report['model'];base=report['baseline']
    improvement=1-model['overall_rmse_nm']/max(base['overall_rmse_nm'],1e-12)
    passed=(np.all(np.asarray(model['per_joint_rmse_nm'])<=cfg['per_joint_rmse_nm']) and
        np.all(abs(np.asarray(model['signed_bias_nm']))<=cfg['per_joint_abs_bias_nm']) and
        improvement>=cfg['baseline_improvement'] and fp<=cfg['free_contact_false_positive_time_rate'])
    return dict(result='PASS' if passed else 'FAIL',version=cfg['version'],baseline_improvement=improvement,
        free_contact_false_positive_time_rate=fp,
        normalization_sigma_nm=np.maximum(train_std,cfg['sigma_floor_nm']).tolist(),
        sigma_floor_applies_to='diagnostic sigma reporting only; model normalization stays frozen')


def evaluate(loaded, entries, output, split='test', max_steps=None, arm=None, make_plots=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    all_m,all_p,all_b=[],[],[];groups={};episodes=[];residuals=[];times=[]
    csv_rows=[];started=time.monotonic()
    for entry in entries:
        meta={}
        if loaded.metadata:
            expected={k:loaded.metadata[k] for k in ('contract_version','side','joint_order','tool','gripper','load','calibration')}
            checked=require(entry['path'],entry['episodes'],expected);meta=checked['metadata']
        for ep in entry['episodes']:
            arrays=read_episode(entry['path'],ep,arm,max_steps,loaded.config['data']['keys'])
            pred,measured=predict_sequence(loaded,arrays)
            baseline=np.broadcast_to(loaded.normalization['y_mean'],pred.shape)
            residual=measured-pred;residuals.append(residual)
            with h5py.File(entry['path'],'r') as h:
                key=loaded.config['data']['keys']['joint_pos'].format(arm=arm)
                if 'timestamps' in h[ep][key]:ts=h[ep][key]['timestamps'][:len(arrays['joint_pos'])][loaded.history-1:]
                else:ts=np.arange(len(pred),dtype=np.int64)*20_000_000
            duration=(int(ts[-1])-int(ts[0]))*1e-9 if len(ts)>1 else 0
            info=dict(path=str(entry['path']),episode=ep,samples=len(pred),duration_seconds=duration,
                model=physical_metrics(measured,pred),baseline=physical_metrics(measured,baseline))
            episodes.append(info);times.append(duration)
            masks={'session:'+meta.get('session_id',Path(entry['path']).stem):np.ones(len(pred),dtype=bool),
                   'trajectory:'+str(meta.get('trajectory_id','unknown')):np.ones(len(pred),dtype=bool)}
            speed=np.linalg.norm(arrays['joint_vel'][loaded.history-1:],axis=1)
            pose=np.linalg.norm(arrays['joint_pos'][loaded.history-1:],axis=1)
            for key,value,threshold in [('speed',speed,.5),('pose',pose,.5)]:
                masks[key+':lt_0.5']=value< threshold;masks[key+':ge_0.5']=value>=threshold
            for label,mask in masks.items():
                if mask.any():groups.setdefault(label,[]).append((measured[mask],pred[mask],baseline[mask]))
            all_m.append(measured);all_p.append(pred);all_b.append(baseline)
            for i in range(len(pred)):
                csv_rows.append([str(entry['path']),ep,int(ts[i])]+measured[i].tolist()+pred[i].tolist()+residual[i].tolist())
            if make_plots:plots(output/'plots',f'{Path(entry["path"]).stem}_{ep}',residual,ts)
    if not all_m:raise ValueError('evaluation_empty')
    report=dict(schema='w3_evaluation_v1',split=split,source=loaded.metadata.get('source','generic'),
        units='torque Nm, MSE Nm^2, timestamps int64 ns',residual_sign='measured-predicted',
        model=physical_metrics(np.concatenate(all_m),np.concatenate(all_p)),
        baseline=physical_metrics(np.concatenate(all_m),np.concatenate(all_b)),episodes=episodes,
        duration_seconds=sum(times),evaluation_seconds=time.monotonic()-started,
        grouping='speed norm rad/s and pose norm rad, fixed 0.5 boundary',groups={})
    for label,parts in groups.items():
        m,p,b=[np.concatenate([x[i] for x in parts]) for i in range(3)]
        report['groups'][label]={'model':physical_metrics(m,p),'baseline':physical_metrics(m,b)}
    with open(output/'samples.csv','w') as f:
        writer=csv.writer(f);n=loaded.output_size
        writer.writerow(['path','episode','timestamp_ns']+[f'{s}_joint_{i}_nm' for s in ('measured','predicted','residual') for i in range(n)])
        writer.writerows(csv_rows)
    with open(output/'metrics.csv','w') as f:
        writer=csv.writer(f);writer.writerow(['predictor','joint','mse_nm2','rmse_nm','signed_bias_nm','std_nm','p95_nm','p99_nm'])
        for label in ('model','baseline'):
            v=report[label]
            for i in range(loaded.output_size):writer.writerow([label,i]+[v[k][i] for k in
                ('per_joint_mse_nm2','per_joint_rmse_nm','signed_bias_nm','residual_std_nm','abs_residual_p95_nm','abs_residual_p99_nm')])
    write_json(output/'report.json',report)
    return report,residuals


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True);p.add_argument('--h5-path',action='append')
    p.add_argument('--manifest');p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--arm',choices=['left','right']);p.add_argument('--episode',action='append')
    p.add_argument('--max-steps',type=int);p.add_argument('--device',default='cpu');p.add_argument('--cpu-threads',type=int,default=1)
    p.add_argument('--output',required=True);p.add_argument('--no-plots',action='store_true');p.add_argument('--acceptance')
    a=p.parse_args();torch.set_num_threads(a.cpu_threads)
    loaded=load_checkpoint(a.run_dir,a.device)
    if a.h5_path:
        entries=[]
        for path in a.h5_path:
            with h5py.File(path,'r') as h:eps=a.episode or sorted(h)
            entries.append({'path':path,'episodes':eps})
    else:
        path=a.manifest or loaded.config['data'].get('manifest')
        manifest,_=validate_manifest(path,{k:loaded.metadata[k] for k in ('side','joint_order','contract_version')})
        if loaded.metadata and sha256(path)!=loaded.metadata['manifest_sha256']:raise ValueError('evaluation_manifest_changed')
        entries=manifest['splits'][a.split]
    report,residual=evaluate(loaded,entries,a.output,'explicit_h5' if a.h5_path else a.split,a.max_steps,a.arm,not a.no_plots)
    report['acceptance']=acceptance(report,residual,yaml.safe_load(Path(a.acceptance).read_text()),loaded.normalization['y_std']) if a.acceptance else {'result':'UNCONFIGURED'}
    write_json(Path(a.output)/'report.json',report)
    print(f"predictions: {report['model']['samples']} RMSE Nm: {report['model']['overall_rmse_nm']:.6f}; acceptance {report['acceptance']['result']}")
    if report['acceptance']['result']!='PASS' and a.acceptance:raise SystemExit(2)


if __name__=='__main__':main()
