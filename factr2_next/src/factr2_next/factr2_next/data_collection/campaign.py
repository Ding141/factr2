"""Independent-session and unchanged-control checks for new NEXT campaigns."""
import hashlib
import json
from pathlib import Path
import yaml
import xml.etree.ElementTree as ET


def geometry_sha256(description):
    """Hash all mechanical geometry/inertia, independently of loaded arm interfaces."""
    root = ET.fromstring(description)
    for child in list(root):
        if child.tag == 'ros2_control':
            root.remove(child)
    return hashlib.sha256(ET.canonicalize(ET.tostring(root, encoding='unicode')).encode()).hexdigest()


def arm_control_profile(metadata):
    """Compare the recorded arm's actual physics, gains, interfaces and calibration.

    A second stationary arm may be loaded into the same controller. Its gains
    do not change this arm's label mapping. Full original snapshots remain saved.
    """
    names = metadata['joint_order']
    side = metadata['side']
    if names != [f'{side}_joint_{i}' for i in range(7)]:
        raise ValueError('Arm profile requires exact seven-joint order')
    control = metadata['control']
    paths = dict(position=control['gains']['position_params'],
                 gravity=control['feedforward']['gravity_params'])
    values = {}
    for label, path in paths.items():
        raw = yaml.safe_load(Path(path).read_text())
        params = next(iter(raw.values()))['ros__parameters']
        joints = params['joints']
        if len(joints) != len(set(joints)) or not set(names) <= set(joints):
            raise ValueError('Missing/duplicate controller joints')
        indices = [joints.index(n) for n in names]
        selected = {}
        for key, value in params.items():
            if key == 'friction_model_yaml':
                continue  # Contents are hashed below; per-session paths differ.
            if key == 'robot_description':
                model = ET.fromstring(value)
                for interface in model.findall('ros2_control'):
                    for joint in list(interface.findall('joint')):
                        if joint.get('name') not in names:
                            interface.remove(joint)
                    for param in interface.findall('./hardware/param'):
                        if param.get('name') == 'offsets_yaml':
                            interface.find('hardware').remove(param)
                value = ET.canonicalize(ET.tostring(model, encoding='unicode'))
            elif isinstance(value, list) and len(value) == len(joints):
                value = [value[i] for i in indices]
            selected[key] = value
        values[label] = selected
    calibration = yaml.safe_load(Path(metadata['calibration']['path']).read_text())
    offsets = {x['name']:x for x in calibration['offsets']}
    values['calibration'] = []
    for name in names:
        entry = offsets[name]
        values['calibration'].append(dict(name=name, axis_sign=float(entry['axis_sign']),
            zero_offset=float(entry['zero_offset']), motor_type=entry['motor_type'],
            slot=int(entry['slot']), channel=int(entry.get('channel', calibration.get('channel')))))
    values['friction'] = yaml.safe_load(Path(control['feedforward']['friction_model']).read_text())
    encoded = json.dumps(values, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def control_profile(metadata):
    if metadata.get('control_profile_scope') == 'arm_v2':
        return arm_control_profile(metadata)
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
