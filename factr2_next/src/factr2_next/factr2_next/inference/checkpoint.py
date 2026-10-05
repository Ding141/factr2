"""Load own trusted artifacts; reject incompatible W3 deployment profiles."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import yaml
from factr2_next.data_collection.quality import sha256
from factr2_next.training.models import build_model
from factr2_next.training.w3_train import FEATURE_ORDER
from factr2_next.w3_config import validate_w3_config
import json


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    normalization: dict
    config: dict
    history: int
    input_size: int
    output_size: int
    device: torch.device
    metadata: dict


def load_checkpoint(run_dir, device='cpu', expected=None):
    run_dir=Path(run_dir);device=torch.device(device)
    for f in ('model.pt','config.yaml','normalization.npz','metrics.json'):
        if not (run_dir/f).is_file():
            raise ValueError('checkpoint_missing:'+f)
    config=yaml.safe_load((run_dir/'config.yaml').read_text())
    w3='contract_version' in config
    meta={}
    if w3 or expected:
        if not (run_dir/'metadata.json').is_file():raise ValueError('checkpoint_missing:metadata.json')
        meta=json.loads((run_dir/'metadata.json').read_text())
        validate_w3_config(config,'train',require_paths=False)
        required=dict(schema='w3_checkpoint_v1',contract_version=config['contract_version'],side=config['side'],
            joint_order=config['joint_names'],feature_order=FEATURE_ORDER,history=50,input_size=21,output_size=7,sample_hz=50)
        required.update(expected or {})
        for k,v in required.items():
            if meta.get(k)!=v:raise ValueError('checkpoint_profile:'+k)
        for k in ('dataset_id','manifest_sha256','software_versions','software_commits','tool','load','calibration','seed','resolved_device'):
            if k not in meta:raise ValueError('checkpoint_metadata:'+k)
        for f in ('model.pt','config.yaml','normalization.npz','metrics.json'):
            if meta.get('artifact_sha256',{}).get(f)!=sha256(run_dir/f):raise ValueError('checkpoint_hash:'+f)
        if meta['seed']!=config['train']['seed'] or meta['resolved_device']!=config['train']['resolved_device']:
            raise ValueError('checkpoint_resolved_config')
    ckpt=torch.load(run_dir/'model.pt',map_location=device,weights_only=True)
    if w3 and (ckpt['input_size'],ckpt['output_size'],ckpt['history'])!=(21,7,50):
        raise ValueError('checkpoint_dimensions')
    if w3 and ckpt.get('model')!=config['model']:raise ValueError('checkpoint_model_config')
    model=build_model(ckpt['model'],ckpt['input_size'],ckpt['output_size'],ckpt['history']).to(device)
    try:model.load_state_dict(ckpt['model_state_dict'],strict=True)
    except RuntimeError as e:raise ValueError('checkpoint_weights_shape:'+str(e)) from e
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('checkpoint_weights_finite')
    model.eval()
    with np.load(run_dir/'normalization.npz',allow_pickle=False) as norms:
        norm={k:norms[k].astype(np.float32) for k in norms.files}
    for k,size in [('x_mean',ckpt['input_size']),('x_std',ckpt['input_size']),('y_mean',ckpt['output_size']),('y_std',ckpt['output_size'])]:
        if k not in norm or norm[k].shape!=(size,) or not np.isfinite(norm[k]).all():raise ValueError('checkpoint_normalization:'+k)
        if k.endswith('std') and not (norm[k]>0).all():raise ValueError('checkpoint_std_positive:'+k)
    return LoadedCheckpoint(model,norm,config,int(ckpt['history']),int(ckpt['input_size']),int(ckpt['output_size']),device,meta)
