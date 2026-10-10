"""Validate explicit W3 profiles before upstream defaults can be used."""
from pathlib import Path

SIGNALS = {
    'joint_pos': ('joint_pos', 'position'),
    'joint_vel': ('joint_vel', 'velocity'),
    'joint_cmd': ('joint_cmd', 'position'),
    'measured_joint_torque': ('joint_effort', 'effort'),
}


def validate_w3_config(cfg, purpose, require_paths=True):
    if 'contract_version' not in cfg:
        # Preserve the upstream Piper profiles.
        if cfg.get('side') or str(cfg.get('robot_topic_root', '')).startswith('/factr2/'):
            raise ValueError('W3 profile requires contract_version')
        return
    if cfg['contract_version'] != 'w3_next_v1' or cfg.get('side') not in ('left', 'right'):
        raise ValueError('Unsupported W3 contract/side')
    side = cfg['side']
    if cfg.get('joint_names') != [f'{side}_joint_{i}' for i in range(7)]:
        raise ValueError('W3 joint order must be side_joint_0..6')
    if purpose in ('record', 'inference'):
        if cfg.get('robot_topic_root') != f'/factr2/{side}':
            raise ValueError('Wrong W3 input root')
        if set(cfg.get('topics', {})) != set(SIGNALS):
            raise ValueError('W3 requires exactly four input keys')
        for key, (suffix, field) in SIGNALS.items():
            spec = cfg['topics'][key]
            topic = spec['topic'].format(robot_topic_root=cfg['robot_topic_root'])
            if topic != f'/factr2/{side}/{suffix}' or spec.get('field') != field:
                raise ValueError(f'Wrong W3 topic/field for {key}')
    if purpose in ('inference', 'visualize'):
        if cfg.get('next_topic_root') != f'/next/{side}':
            raise ValueError('Wrong NEXT output root')
        for topic in cfg['outputs'].values():
            if not topic.format(next_topic_root=cfg['next_topic_root']).startswith(f'/next/{side}/'):
                raise ValueError('Output must stay in side NEXT namespace')
    if purpose == 'train':
        data = cfg['data']
        if data.get('sample_hz',50) not in (50,100):
            raise ValueError('W3 sample frequency must be 50 or 100 Hz')
        if data.get('arm_mode') != 'single' or data.get('history') != 50:
            raise ValueError('W3 baseline requires single arm and history=50')
        if data.get('keys') != {k: k for k in SIGNALS}:
            raise ValueError('W3 H5 keys must have no arm prefix')
        model = cfg['model']
        baseline = dict(type='lstm', state_mode='stateless', bidirectional=False,
                        hidden_size=128, num_layers=2, head_hidden=256, head_layers=2, dropout=0.1)
        if any(model.get(k) != v for k, v in baseline.items()):
            raise ValueError('W3 v1 baseline model differs from contract')
        if data.get('val_split') == 'random':
            raise ValueError('W3 random window split forbidden')
        if not data.get('manifest'):
            raise ValueError('W3 accepted train/val/test manifest required')
        if require_paths and not Path(data['manifest']).is_file():
            raise ValueError('Set an existing W3 split manifest')
    if purpose == 'inference' and cfg.get('require_adapter_status', True) is not True:
        raise ValueError('W3 inference requires adapter diagnostics')
    if require_paths and purpose == 'inference':
        for name in ('model.pt', 'config.yaml', 'normalization.npz'):
            if not (Path(cfg['checkpoint_dir']) / name).is_file():
                raise ValueError(f'Set a complete W3 checkpoint directory: missing {name}')
    if require_paths and purpose == 'record':
        if 'SESSION_ID' in cfg['output_dir'] or 'SESSION_ID' in cfg['session_name']:
            raise ValueError('Set W3 output_dir and session_name before recording')
