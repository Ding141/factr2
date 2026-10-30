"""Synthetic identifiability/sign and held-out no-retuning checks."""
import unittest
import numpy as np
from w3_signal_audit import SETTINGS, fit_lags, validate_lags, regular_runs, noise


class SignalAuditTests(unittest.TestCase):
    def test_known_lag_and_independent_mismatch_is_not_refitted(self):
        cfg = dict(SETTINGS, max_lag_ms=80, bootstrap_repeats=5)
        t = np.arange(0, 15, .01)
        def wave(t): return .2*np.sin(2*np.pi*.8*t) + .1*np.sin(2*np.pi*1.7*t)
        def run(lag):
            x = np.repeat(wave(t)[:, None], 7, axis=1)
            y = np.repeat(wave(t-lag)[:, None], 7, axis=1)
            return dict(t=t, q_dot=x, v_smooth=y, c_dot=x, kind='moving', event='synth')
        fitted = fit_lags([run(.04)], cfg)
        self.assertEqual([p['lag_ms'] for p in fitted['velocity_vs_position']], [40]*7)
        frozen = dict(settings=cfg, fits=fitted)
        result = validate_lags([run(-.04)], frozen)
        for p in result['velocity_vs_position']:
            self.assertEqual(p['frozen_lag_ms'], 40)
            self.assertGreater(p['shifted']['rmse'], p['baseline']['rmse'])

    def test_gap_is_not_interpolated_across(self):
        t = np.r_[np.arange(100)*10_000_000, 3_000_000_000+np.arange(100)*10_000_000]
        ep = dict(t=t, kind='moving', event='gap',
                  **{k:np.zeros((200,7)) for k in ('q','v','c','tau')})
        runs = regular_runs([ep], SETTINGS)
        self.assertEqual(len(runs), 2)
        self.assertTrue(all(r['t'][-1] < 1.01 for r in runs))

    def test_static_gate_does_not_require_zero_velocity_code(self):
        ep = dict(t=np.arange(500,dtype=np.int64)*10_000_000,
                  kind='static_hold', event='quantized_zero',
                  q=np.zeros((500,7)), c=np.zeros((500,7)),
                  v=np.ones((500,7))*.01221, tau=np.ones((500,7)))
        result = noise([ep], SETTINGS)
        self.assertGreater(result['rows'], 100)
        self.assertTrue(np.allclose(result['segments'][0]['velocity_mean_deg_s'], .01221*180/np.pi))


if __name__ == '__main__': unittest.main()
