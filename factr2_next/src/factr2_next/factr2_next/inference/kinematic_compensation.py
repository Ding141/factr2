"""Causal, frozen free-motion residual correction; no robot control or online fit.

The pretrained NEXT receives its original inputs. This branch smooths motion
signals before learning *additional* errors. Torque and free prediction receive
identical input filtering; a separate output filter has explicitly measured lag.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

SCHEMA = 'w3_kinematic_residual_v1'
ROOT = '/next/right/kinematic_compensated'
DEFAULT_SIGNAL = dict(input_torque_hz=12., output_hz=5., motor_velocity_hz=8.,
                      position_noise_rad=0.00038, jerk_density=1.,
                      max_gap_seconds=.04, warmup_seconds=.5)
POSE_MODES = ('position_velocity_acceleration', 'position_dynamic_coupling')
PAIR_I, PAIR_J = np.triu_indices(7, 1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LowPass:
    """Exact first-order RC pole, variable dt, unity DC gain, no look-ahead."""
    def __init__(self, hz):
        if not np.isfinite(hz) or not 0 < hz < 50:
            raise ValueError('cutoff must be in (0,50) Hz')
        self.hz = float(hz)
        self.y = None

    def update(self, x, dt):
        x = np.asarray(x, float)
        if not np.isfinite(x).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError('nonfinite input or invalid dt')
        self.y = x.copy() if self.y is None else self.y + (-np.expm1(-2*np.pi*self.hz*dt))*(x-self.y)
        return self.y.copy()


class MotionState:
    """Position-only constant-acceleration KF, shared covariance, seven joints.

    The quantized velocity feedback is intentionally not fused as a calibrated
    measurement. All derivatives are causal estimates, never ground truth.
    """
    def __init__(self, noise, jerk):
        if not np.isfinite([noise, jerk]).all() or min(noise, jerk) <= 0:
            raise ValueError('positive KF covariance required')
        self.r, self.jerk = float(noise)**2, float(jerk)
        self.x = None
        self.p = np.diag([self.r, .01, 1.])

    def update(self, q, dt):
        q = np.asarray(q, float)
        if self.x is None:
            self.x = np.stack([q, np.zeros(7), np.zeros(7)])
            return self.x.copy()
        f = np.array([[1, dt, dt*dt/2], [0, 1, dt], [0, 0, 1.]])
        noise = self.jerk*np.array([[dt**5/20, dt**4/8, dt**3/6],
                                   [dt**4/8, dt**3/3, dt**2/2],
                                   [dt**3/6, dt**2/2, dt]])
        x = f@self.x
        p = f@self.p@f.T + noise
        k = p[:, 0]/(p[0, 0]+self.r)
        self.x = x + k[:, None]*(q-x[0])[None, :]
        # Joseph covariance update avoids loss of positive semidefiniteness.
        a = np.eye(3); a[:, 0] -= k
        self.p = a@p@a.T + np.outer(k, k)*self.r
        return self.x.copy()


def feature_names(mode):
    groups = ['v', 'command_v', 'friction_v_slow', 'friction_command_slow',
              'friction_v_fast', 'friction_command_fast', 'abs_v', 'abs_command_v']
    if mode == 'velocity_acceleration' or mode in POSE_MODES:
        groups += ['a', 'command_a']
    elif mode != 'velocity':
        raise ValueError('unknown compensation feature mode')
    names = [f'{g}_j{j+1}' for g in groups for j in range(7)]
    if mode in POSE_MODES:
        names += [f'{g}_j{j+1}' for g in ('sin_q', 'one_minus_cos_q',
                  'sin_chain_q', 'one_minus_cos_chain_q') for j in range(7)]
        names += [f'{g}_j{i+1}_j{j+1}' for g in ('sin_pair_q', 'one_minus_cos_pair_q')
                  for i, j in zip(PAIR_I, PAIR_J)]
    if mode == 'position_dynamic_coupling':
        names += [f'{g}_j{j+1}' for g in ('v_abs_v', 'cv_abs_cv',
                  'v_sin_q', 'cv_sin_q', 'v_cos_q', 'cv_cos_q',
                  'a_sin_q', 'ca_sin_q', 'a_cos_q', 'ca_cos_q') for j in range(7)]
        names += ['motion_gain_'+s for s in feature_names('velocity_acceleration')]
    return names


def augment_position_features(base, q, mode):
    """Same nonlinear basis for batched fitting and single causal frames.

    Contains kinematics only. No measured torque, residual, clock/trajectory
    phase, or online bias adaptation can enter this frozen correction.
    """
    base, q = np.asarray(base, float), np.asarray(q, float)
    if mode not in POSE_MODES or base.shape[-1] != 70 or q.shape != base.shape[:-1]+(7,):
        raise ValueError('invalid pose feature dimensions/mode')
    chain = np.cumsum(q, axis=-1)
    pairs = q[..., PAIR_I]+q[..., PAIR_J]
    values = [base, np.sin(q), 1-np.cos(q), np.sin(chain), 1-np.cos(chain),
              np.sin(pairs), 1-np.cos(pairs)]
    if mode == 'position_dynamic_coupling':
        v, cv, a, ca = (base[..., s:s+7] for s in (0, 7, 56, 63))
        # Command-derived continuous speed gate is a feature, never an output
        # deadband. Including both bases lets train/validation distinguish
        # low-speed friction from moving dynamics without a hard transition.
        speed2 = np.mean(cv*cv, axis=-1, keepdims=True)
        gain = speed2/(speed2+np.deg2rad(1.)**2)
        values += [v*abs(v), cv*abs(cv), v*np.sin(q), cv*np.sin(q),
                   v*np.cos(q), cv*np.cos(q), a*np.sin(q), ca*np.sin(q),
                   a*np.cos(q), ca*np.cos(q), gain*base]
    result = np.concatenate(values, axis=-1)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite pose features')
    return result


def features(actual, command, mode):
    v, cv = actual[1], command[1]
    values = [v, cv, np.tanh(v/np.deg2rad(.2)), np.tanh(cv/np.deg2rad(.2)),
              np.tanh(v/np.deg2rad(1.)), np.tanh(cv/np.deg2rad(1.)), abs(v), abs(cv)]
    if mode == 'velocity_acceleration' or mode in POSE_MODES:
        values += [actual[2], command[2]]
    result = np.concatenate(values)
    if mode in POSE_MODES:
        result = augment_position_features(result, actual[0], mode)
    if len(result) != len(feature_names(mode)) or not np.isfinite(result).all():
        raise ValueError('invalid kinematic features')
    return result


class SignalProcessor:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULT_SIGNAL if cfg is None else cfg)
        if (set(self.cfg) != set(DEFAULT_SIGNAL) or not np.isfinite(list(self.cfg.values())).all()
                or self.cfg['max_gap_seconds'] != .04 or self.cfg['warmup_seconds'] < .5):
            raise ValueError('invalid signal recipe')
        self.reset()

    def reset(self):
        c = self.cfg
        self.actual = MotionState(c['position_noise_rad'], c['jerk_density'])
        self.command = MotionState(c['position_noise_rad'], c['jerk_density'])
        self.motor_velocity = LowPass(c['motor_velocity_hz'])
        self.measured = LowPass(c['input_torque_hz'])
        self.free = LowPass(c['input_torque_hz'])
        self.last = self.start = None

    def update(self, stamp, q, v, cmd, measured, free):
        arrays = [np.asarray(a, float) for a in (q, v, cmd, measured, free)]
        if type(stamp) not in (int, np.int64) or stamp <= 0 or any(a.shape != (7,) or not np.isfinite(a).all() for a in arrays):
            self.reset()
            raise ValueError('invalid input frame')
        if self.last is not None and stamp <= self.last:
            self.reset()
            raise ValueError('source stamp not advancing')
        reset = self.last is not None and stamp-self.last > self.cfg['max_gap_seconds']*1e9
        if reset:
            self.reset()
        dt = .01 if self.last is None else (stamp-self.last)*1e-9
        if self.start is None:
            self.start = stamp
        self.last = stamp
        q, v, cmd, tau, mu = arrays
        actual = self.actual.update(q, dt)
        command = self.command.update(cmd, dt)
        return dict(stamp_ns=stamp, dt=dt, actual=actual, command=command,
                    motor_velocity=self.motor_velocity.update(v, dt),
                    measured=self.measured.update(tau, dt), free=self.free.update(mu, dt),
                    ready=stamp-self.start >= self.cfg['warmup_seconds']*1e9, reset=reset)


class ResidualModel:
    def __init__(self, artifact):
        self.artifact = artifact
        if artifact.get('schema') != SCHEMA or artifact.get('uses_torque_as_feature') is not False or artifact.get('online_learning') is not False:
            raise ValueError('wrong residual model contract')
        self.mode = artifact['feature_mode']
        if artifact['feature_names'] != feature_names(self.mode):
            raise ValueError('feature order mismatch')
        self.scale = np.asarray(artifact['feature_scale'], float)
        self.weights = np.asarray(artifact['weights'], float)
        self.low = np.asarray(artifact['feature_min'], float)
        self.high = np.asarray(artifact['feature_max'], float)
        self.caps = np.asarray(artifact['correction_cap_nm'], float)
        self.enabled = np.asarray(artifact['enabled_joints'], bool)
        self.active = np.asarray(artifact.get('active_features', np.ones(len(self.scale), bool)), bool)
        self.position_bounds = np.asarray(artifact['position_bounds_rad'], float)
        n = len(self.scale)
        if (self.weights.shape != (n, 7) or self.low.shape != (n,) or self.high.shape != (n,)
                or self.caps.shape != (7,) or self.enabled.shape != (7,) or self.position_bounds.shape != (7, 2)
                or self.active.shape != (n,) or n != len(feature_names(self.mode)) or np.any(self.scale <= 0)
                or np.any(self.caps <= 0) or np.any(self.high < self.low)
                or any(not np.isfinite(a).all() for a in (self.scale, self.weights, self.low, self.high, self.caps, self.position_bounds))
                or np.any(self.position_bounds[:, 1] < self.position_bounds[:, 0])):
            raise ValueError('malformed residual model')
        if np.any(abs(self.weights[~self.active]) > 1e-12):
            raise ValueError('unexcited features must have zero weight')
        self.network = artifact.get('nonlinear_residual')
        if self.network is not None:
            net = self.network
            self.net_indices = np.asarray(net['input_indices'], int)
            self.net_center = np.asarray(net['input_center'], float)
            self.net_scale = np.asarray(net['input_scale'], float)
            m = len(self.net_indices)
            if (net.get('activation') != 'tanh' or m < 1 or m > n or
                    len(set(self.net_indices.tolist())) != m or
                    np.any(self.net_indices < 0) or np.any(self.net_indices >= n) or
                    self.net_center.shape != (m,) or self.net_scale.shape != (m,) or
                    np.any(self.net_scale <= 0) or
                    not np.isfinite(self.net_center).all() or not np.isfinite(self.net_scale).all()):
                raise ValueError('malformed nonlinear input normalization')
            self.net_layers = []
            for layer in net['layers']:
                w, b = np.asarray(layer['weight'],float), np.asarray(layer['bias'],float)
                if (w.ndim != 2 or w.shape[0] != m or w.shape[1] > 128 or
                        b.shape != (w.shape[1],) or not np.isfinite(w).all() or not np.isfinite(b).all()):
                    raise ValueError('malformed nonlinear layer')
                self.net_layers.append((w,b));m=w.shape[1]
            if not 1 <= len(self.net_layers) <= 4 or m != 7:
                raise ValueError('nonlinear output must contain seven joints')
            if np.any(~self.active[self.net_indices]):
                raise ValueError('nonlinear inputs must be within the supported active basis')
        self.speed_gate = np.asarray(artifact.get('motion_gate_deg_s', 0.),float)
        if (self.speed_gate.shape not in ((),(7,)) or not np.isfinite(self.speed_gate).all()
                or np.any(self.speed_gate < 0)):
            raise ValueError('invalid correction speed gate')
        self.blend = artifact.get('speed_blend')
        if self.blend is not None:
            if any('speed_blend' in self.blend[k] for k in ('slow','fast')):
                raise ValueError('nested speed blends are not allowed')
            self.blend_models = tuple(ResidualModel(self.blend[k]) for k in ('slow','fast'))
            self.blend_speed = np.asarray(self.blend['transition_deg_s'],float)
            if (self.blend_speed.shape not in ((),(7,)) or not np.isfinite(self.blend_speed).all()
                    or np.any(self.blend_speed <= 0)):
                raise ValueError('invalid blend transition')
            for model in self.blend_models:
                if (model.mode != self.mode or model.artifact['base_metadata_sha256'] != artifact['base_metadata_sha256']
                        or model.artifact['signal'] != artifact['signal'] or
                        not np.array_equal(model.caps,self.caps) or not np.array_equal(model.enabled,self.enabled)):
                    raise ValueError('blend components must share mean, signals, features, caps and joint mask')
        self.fallback = None
        if artifact.get('support_fallback') is not None:
            fallback = artifact['support_fallback']
            if 'support_fallback' in fallback or 'speed_blend' in fallback:
                raise ValueError('fallback must be a single frozen model')
            self.fallback = ResidualModel(fallback)
            if (self.fallback.mode != self.mode or fallback['base_metadata_sha256'] != artifact['base_metadata_sha256']
                    or fallback['signal'] != artifact['signal'] or
                    not np.array_equal(self.fallback.caps,self.caps) or
                    not np.array_equal(self.fallback.enabled,self.enabled)):
                raise ValueError('fallback identity/signals/caps/mask mismatch')

    @classmethod
    def load(cls, path, source_metadata_sha256):
        artifact = json.loads(Path(path).read_text())
        if artifact['base_metadata_sha256'] != source_metadata_sha256:
            raise ValueError('compensator and NEXT model identity differ')
        return cls(artifact)

    def correction_from_features(self, x):
        """Kinematics-only numerical core, shared by batched and live paths."""
        x = np.asarray(x,float)
        if self.blend is not None:
            slow,fast=(model.correction_from_features(x) for model in self.blend_models)
            speed2=np.mean(x[...,7:14]**2,axis=-1,keepdims=True)
            gain=speed2/(speed2+np.deg2rad(self.blend_speed)**2)
            return (1-gain)*slow+gain*fast
        value = (x/self.scale)@self.weights
        if self.network is not None:
            z = (x[..., self.net_indices]-self.net_center)/self.net_scale
            for index, (w,b) in enumerate(self.net_layers):
                z = z@w+b
                if index+1 < len(self.net_layers):z=np.tanh(z)
            value += z
        value = np.clip(value, -self.caps, self.caps)*self.enabled
        if np.any(self.speed_gate):
            speed2 = np.mean(x[..., 7:14]**2, axis=-1, keepdims=True)
            denominator = speed2+np.deg2rad(self.speed_gate)**2
            value *= np.divide(speed2, denominator, out=np.ones_like(value),where=denominator>0)
        return value

    def predict_features(self, x, q, ready):
        """Shared support and fallback semantics for live and batched replay."""
        x,q=np.asarray(x,float),np.asarray(q,float)
        margin = np.maximum(.2*(self.high-self.low), .2*self.scale)
        supported = (np.all(((x >= self.low-margin) & (x <= self.high+margin))[..., self.active],axis=-1)
            & np.all((q >= self.position_bounds[:, 0]) & (q <= self.position_bounds[:, 1]),axis=-1)
            & np.asarray(ready,bool))
        value = self.correction_from_features(x)
        value = np.where(supported[...,None],value,0.)
        if self.fallback is not None:
            other,other_support=self.fallback.predict_features(x,q,ready)
            value=np.where(supported[...,None],value,other)
            supported |= other_support
        return value,supported

    def predict(self, frame):
        x = features(frame['actual'], frame['command'], self.mode)
        value,supported=self.predict_features(x,frame['actual'][0],frame['ready'])
        return value,bool(supported)


class Compensator:
    def __init__(self, model):
        self.model = model
        self.signal = SignalProcessor(model.artifact['signal'])
        self.output = LowPass(model.artifact['signal']['output_hz'])

    def reset(self):
        self.signal.reset()
        self.output = LowPass(self.model.artifact['signal']['output_hz'])

    def update(self, stamp, q, v, cmd, measured, free, *, allow_compensation=True):
        try:
            frame = self.signal.update(stamp, q, v, cmd, measured, free)
            if frame['reset']:
                self.output = LowPass(self.model.artifact['signal']['output_hz'])
            correction, supported = self.model.predict(frame)
            if not allow_compensation:
                correction = np.zeros(7)
            raw = frame['measured']-frame['free']-correction
            final = self.output.update(raw, frame['dt'])
            return dict(frame, correction=correction, supported=supported,
                        raw=raw, final=final, free_total=frame['free']+correction,
                        baseline_raw=np.asarray(measured)-np.asarray(free))
        except (ValueError, FloatingPointError):
            self.reset()
            raise
