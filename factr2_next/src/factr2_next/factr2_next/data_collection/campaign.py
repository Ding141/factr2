"""Independent-session and unchanged-control checks for new NEXT campaigns."""
import hashlib
import json
from pathlib import Path
import yaml


def control_profile(metadata):
    control = metadata['control']
    paths = dict(position=control['gains']['position_params'],
                 gravity=control['feedforward']['gravity_params'],
                 friction=control['feedforward']['friction_model'])
    values = {k:yaml.safe_load(Path(p).read_text()) for k,p in paths.items()}
    # Paths of identical frozen friction models can differ between sessions.
    def normalize(value):
        if isinstance(value, dict):
            return {k:normalize(v) for k,v in value.items() if k != 'friction_model_yaml'}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value
    encoded = json.dumps(normalize(values), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_independent_manifest(manifest, reports):
    roles = {}; hashes = {}; profiles = set()
    for split in ('train', 'val', 'test'):
        for report in reports[split]:
            meta = report['metadata']
            if meta['source'] != 'real' or meta['sample_hz'] != 100:
                raise ValueError('Independent campaign requires real 100 Hz recordings')
            derivation = meta.get('derivation', {})
            identity = derivation.get('parent_session_id', meta['session_id'])
            digest = derivation.get('parent_h5_sha256', report['h5_sha256'])
            for key, table in ((identity, roles), (digest, hashes)):
                if key in table and table[key] != split:
                    raise ValueError('independent_session_overlap:' + key)
                table[key] = split
            profile = control_profile(meta)
            if meta.get('control_profile_sha256') != profile:
                raise ValueError('control_profile_hash_mismatch')
            profiles.add(profile)
    if len(profiles) != 1:
        raise ValueError('control_profile_mismatch')
    if manifest.get('evaluation_scope') != 'independent_recording_sessions':
        raise ValueError('independent_evaluation_scope_required')
    return dict(session_roles=roles, control_profile_sha256=next(iter(profiles)))
