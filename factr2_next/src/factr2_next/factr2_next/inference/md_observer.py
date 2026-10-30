"""Explicit opt-in, read-only NEXT+MD observation using the frozen MD recipe."""
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from factr2_next.data_collection.quality import sha256
from factr2_next.inference.checkpoint import LoadedCheckpoint, load_checkpoint
from factr2_next.inference.history_buffer import HistoryBuffer
from factr2_next.inference.inference_node import InferenceNode
from factr2_next.training.models import build_model
from factr2_next.training.motion_discriminator import MotionDiscriminator


def load_md_checkpoint(path, recipe_path, device='cpu', expected=None):
    """Verify MD artifacts and their original NEXT provenance before observing."""
    path, recipe_path = Path(path), Path(recipe_path)
    meta = json.loads((path / 'metadata.json').read_text())
    required = dict(schema='next_md_ablation_v1', input_size=126, output_size=7,
                    history=50, sample_hz=100, side='right',
                    joint_order=[f'right_joint_{i}' for i in range(7)],
                    feature_order=['q', 'qdot', 'q_cmd-q', 'MD_threshold_major'])
    required.update(expected or {})
    for key, value in required.items():
        if meta.get(key) != value:
            raise ValueError('md_checkpoint_profile:' + key)
    for name in ('model.pt', 'normalization.npz', 'config.yaml', 'metrics.json'):
        if meta.get('artifact_sha256', {}).get(name) != sha256(path / name):
            raise ValueError('md_checkpoint_hash:' + name)
    if sha256(recipe_path) != meta.get('recipe_sha256'):
        raise ValueError('md_recipe_hash')
    recipe = json.loads(recipe_path.read_text())
    if (recipe.get('schema') != 'next_md_recipe_v1'
            or recipe.get('flatten_order') != 'threshold-major then right_joint_0..6'
            or recipe.get('max_gap_ns') != 40_000_000
            or np.asarray(recipe.get('thresholds_rad_s')).shape != (15, 7)):
        raise ValueError('md_recipe_profile')
    MotionDiscriminator(recipe['thresholds_rad_s'], recipe['max_gap_ns'])
    cfg = yaml.safe_load((path / 'config.yaml').read_text())
    baseline_path = Path(meta['baseline_run'])
    baseline = load_checkpoint(baseline_path, device=device, expected=expected)
    comparison = cfg['comparison']
    if (Path(comparison['baseline_run']) != baseline_path
            or comparison['baseline_model_sha256'] != sha256(baseline_path / 'model.pt')
            or meta['manifest_sha256'] != baseline.metadata['manifest_sha256']
            or cfg['model'] != baseline.config['model']
            or cfg['side'] != meta['side'] or cfg['joint_names'] != meta['joint_order']
            or cfg['data']['history'] != 50 or cfg['data']['sample_hz'] != 100):
        raise ValueError('md_baseline_provenance')
    device = torch.device(device)
    checkpoint = torch.load(path / 'model.pt', map_location=device, weights_only=True)
    if (checkpoint.get('input_size'), checkpoint.get('output_size'), checkpoint.get('history')) != (126, 7, 50):
        raise ValueError('md_checkpoint_dimensions')
    if checkpoint['model'] != cfg['model']:
        raise ValueError('md_checkpoint_model_config')
    model = build_model(checkpoint['model'], 126, 7, 50).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('md_checkpoint_weights_finite')
    model.eval()
    with np.load(path / 'normalization.npz', allow_pickle=False) as archive:
        norm = {key: archive[key].astype(np.float32) for key in archive.files}
    for key, size in [('x_mean', 126), ('x_std', 126), ('y_mean', 7), ('y_std', 7)]:
        if key not in norm or norm[key].shape != (size,) or not np.isfinite(norm[key]).all():
            raise ValueError('md_normalization:' + key)
        if key.endswith('std') and np.any(norm[key] <= 0):
            raise ValueError('md_normalization_std:' + key)
        original = norm[key][:21] if key.startswith('x_') else norm[key]
        if not np.array_equal(original, baseline.normalization[key]):
            raise ValueError('md_baseline_normalization:' + key)
    # Envelope and physical-control provenance belong to the same frozen data.
    metadata = dict(meta)
    for key in ('operating_envelope', 'control_profile_sha256', 'calibration'):
        metadata[key] = baseline.metadata[key]
    return LoadedCheckpoint(model, norm, cfg, 50, 126, 7, device, metadata), recipe


class MDHistoryBuffer(HistoryBuffer):
    def __init__(self, history, recipe):
        super().__init__(history)
        self.md = MotionDiscriminator(recipe['thresholds_rad_s'], recipe['max_gap_ns'])

    def append(self, joint_pos, joint_vel, joint_cmd, stamp):
        arrays = [np.asarray(v, dtype=np.float32) for v in (joint_pos, joint_vel, joint_cmd)]
        if any(v.shape != (7,) or not np.isfinite(v).all() for v in arrays):
            raise ValueError('md_history_shape_finite')
        if self.md.last_stamp is not None and stamp - self.md.last_stamp > self.md.max_gap_ns:
            self.clear()
        memory = self.md.step(arrays[1], stamp)
        super().append(*arrays)
        self.rows[-1] = np.concatenate([self.rows[-1], memory])

    def clear(self):
        super().clear()
        self.md.reset()


class MDInferenceNode(InferenceNode):
    def _load_checkpoint(self, path, **kwargs):
        expected_root = '/next/right/md/'
        for name, topic in self.cfg['outputs'].items():
            actual = self._format_topic(topic)
            if not actual.startswith(expected_root):
                raise ValueError('md_output_namespace:' + name)
        if 'status' not in self.cfg['outputs']:
            raise ValueError('md_status_topic_required')
        loaded, self.md_recipe = load_md_checkpoint(path, self.cfg['md_recipe'], **kwargs)
        return loaded

    def _create_history_buffer(self):
        return MDHistoryBuffer(self.loaded.history, self.md_recipe)

    def _append_history(self, joint_pos, joint_vel, joint_cmd):
        self.buffer.append(joint_pos, joint_vel, joint_cmd, self.last_stamp)


def main():
    import rclpy
    rclpy.init()
    node = None
    try:
        node = MDInferenceNode()
        node.get_logger().info('NEXT+MD observation only; zero initial MD means unknown earlier motion history')
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
