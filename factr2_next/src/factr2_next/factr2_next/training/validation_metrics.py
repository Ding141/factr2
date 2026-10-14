"""Physical validation diagnostics without changing the paper MSE objective."""
import numpy as np
import re
import torch
from factr2_next.training.coverage_audit import group_name


def validation_epoch(model, loader, device, normalization, loss_units):
    ds = loader.dataset
    model.eval()
    groups, episodes, speeds = {}, {}, {}
    total_loss = 0.
    seen = 0
    scale = torch.as_tensor(normalization['y_std'], device=device)
    with torch.inference_mode():
        for x, y in loader:
            error = model(x.to(device))-y.to(device)
            physical = error*scale
            loss_error = physical if loss_units == 'physical_nm' else error
            total_loss += float((loss_error**2).mean())*len(x)
            residual = -physical.cpu().numpy().astype(float)
            # Non-shuffled sequential validation; group by final sample's episode.
            indices = np.searchsorted(ds.ends, np.arange(seen, seen+len(x)), side='right')
            for index in np.unique(indices):
                info = ds.episode_info[index]
                values = residual[indices == index]
                for collection, key in ((groups, group_name(info)), (episodes, int(index)),
                        (speeds, group_name(info)+':'+info.get('speed', 'unknown'))):
                    item = collection.setdefault(key, dict(count=0, sum=np.zeros(7), squared=np.zeros(7)))
                    item['count'] += len(values)
                    item['sum'] += values.sum(0)
                    item['squared'] += (values**2).sum(0)
            seen += len(x)
    if seen != len(ds):
        raise ValueError('Validation diagnostics require an ordered full dataset')
    def finish(item):
        return dict(samples=item['count'], signed_bias_nm=(item['sum']/item['count']).tolist(),
                    per_joint_rmse_nm=np.sqrt(item['squared']/item['count']).tolist())
    statics = [dict(ds.episode_info[i], **finish(item)) for i,item in episodes.items()
               if group_name(ds.episode_info[i]) == 'static']
    paired = {}
    for item in statics:
        match = re.fullmatch(r'static_pose(\d+)_from_(low|high)', item['kind'])
        if match:
            paired.setdefault((item['path'], match[1]), {})[match[2]] = item
    returns = []
    for (path, pose), pair in paired.items():
        if set(pair) == {'low', 'high'}:
            gap = abs(np.asarray(pair['low']['signed_bias_nm'])-pair['high']['signed_bias_nm'])
            returns.append(dict(path=path, pose_id=pose, baseline_gap_nm=gap.tolist()))
    return total_loss/seen, dict(groups={k:finish(v) for k,v in groups.items()}, static_episodes=statics,
        groups_by_speed={k:finish(v) for k,v in speeds.items()}, matched_return_baselines=returns)
