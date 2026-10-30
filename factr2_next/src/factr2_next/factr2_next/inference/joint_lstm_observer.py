"""Pinned scalar-LSTM mean observer; subscribes to state, never actuates."""
import json
from pathlib import Path
import time

from .checkpoint import load_checkpoint
from .inference_node import InferenceNode, TorqueFilter
from factr2_next.data_collection.quality import sha256

SCHEMA = 'w3_joint_lstm_observer_v1'
ROOT = '/next/right/joint_lstm'


class ResidualCascade:
    """LP12(tau - mu) then LP5; linear equivalent of paired torque filters."""
    mode, sample_hz = 'lowpass_12_then_5', 100.

    def __init__(self):
        self.input = TorqueFilter(dict(mode='lowpass', sample_hz=100., cutoff_hz=12.))
        self.output = TorqueFilter(dict(mode='lowpass', sample_hz=100., cutoff_hz=5.))

    def update(self, values):
        return self.output.update(self.input.update(values))

    def reset(self):
        self.input.reset(); self.output.reset()


def load_bundle(path):
    path = Path(path).resolve()
    meta = json.loads((path/'bundle.json').read_text())
    if (meta.get('schema') != SCHEMA or meta.get('architecture') != 'independent_joint_lstm_v1'
            or meta.get('controller_changes') is not False or meta.get('compensation_enabled') is not False):
        raise ValueError('joint_lstm_bundle_profile')
    expected = {'inference.yaml', 'web.yaml'} | {'mean_checkpoint/'+name for name in
        ('model.pt', 'config.yaml', 'metadata.json', 'metrics.json', 'normalization.npz')}
    if set(meta['files']) != expected:
        raise ValueError('joint_lstm_bundle_files')
    for name, digest in meta['files'].items():
        if sha256(path/name) != digest:
            raise ValueError('joint_lstm_bundle_hash:'+name)
    loaded = load_checkpoint(path/'mean_checkpoint', expected=dict(side='right', sample_hz=100,
        architecture='independent_joint_lstm_v1'))
    loaded.metadata['operating_envelope'] = meta['operating_envelope']
    loaded.model.requires_grad_(False)
    return loaded, meta


class JointLSTMInferenceNode(InferenceNode):
    def __init__(self, bundle, **kwargs):
        self.bundle = Path(bundle).resolve()
        self.bundle_metadata_hash = sha256(self.bundle/'bundle.json')
        self.identity_ok = True
        self.last_identity_check = 0.
        super().__init__(**kwargs)
        self.torque_filter = ResidualCascade()

    def _load_checkpoint(self, path, **kwargs):
        if Path(path).resolve() != self.bundle/'mean_checkpoint':
            raise ValueError('joint_lstm_checkpoint_location')
        loaded, meta = load_bundle(self.bundle)
        if self.cfg['next_topic_root'] != ROOT or self.cfg['contact']['enabled'] is not False:
            raise ValueError('joint_lstm_readonly_namespace')
        if self.cfg.get('operating_envelope') != meta['operating_envelope']:
            raise ValueError('joint_lstm_envelope_mismatch')
        for key, expected in kwargs.get('expected', {}).items():
            if loaded.metadata.get(key) != expected:
                raise ValueError('joint_lstm_checkpoint_expected:'+key)
        return loaded

    def _watchdog(self):
        now = time.monotonic()
        if now-self.last_identity_check >= 1.:
            self.last_identity_check = now
            try:
                self.identity_ok &= sha256(self.bundle/'bundle.json') == self.bundle_metadata_hash
            except OSError:
                self.identity_ok = False
        super()._watchdog()
        if not self.identity_ok:
            self._invalidate('joint_lstm_bundle_identity_changed')

    def _callback(self, *msgs):
        if not self.identity_ok:
            self._invalidate('joint_lstm_bundle_identity_changed')
            return
        return super()._callback(*msgs)
