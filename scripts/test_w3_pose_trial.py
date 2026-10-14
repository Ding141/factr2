"""Verify whole-pose isolation, purging and explicit sample-boundary allowance."""
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from prepare_w3_pose_trial import event_role, grouped_bounds, trial_event_blocks


class PoseTrialTests(unittest.TestCase):
    def test_whole_pose_holdout_includes_both_return_directions(self):
        for pose, role in ((5, 'train'), (6, 'val'), (7, 'test')):
            for kind in (f'static_pose{pose}_from_low', f'static_pose{pose}_from_high',
                         f'pose{pose}_line_x_leg0', f'pose{pose}_coupled1_leg2'):
                self.assertEqual(event_role(kind), role)
        self.assertEqual(event_role('sweep_j6'), 'train')
        with self.assertRaises(ValueError):
            event_role('pose8_coupled0_leg0')

    def test_two_second_separation_at_role_change(self):
        events = [dict(name=str(i), kind=f'static_pose{pose}_from_low',
                       utc=f'1970-01-01T00:00:{i*10:02d}+00:00',
                       completed_utc=f'1970-01-01T00:00:{(i+1)*10:02d}+00:00')
                  for i, pose in enumerate((5, 6, 7))]
        bounds = grouped_bounds(events)
        for left, right in (('0', '1'), ('1', '2')):
            self.assertEqual(bounds[right][1]-bounds[left][2], 2_000_000_000)

    def test_sample_clock_quantization_does_not_allow_missing_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'raw.h5'
            with h5py.File(path, 'w') as raw:
                raw.create_dataset('ep_0000/joint_pos/timestamps',
                                   data=np.arange(51, dtype=np.int64)*10_000_000)
            event = dict(name='leg', kind='pose0_line_x_leg0', phase='cartesian', speed='slow',
                         utc='1970-01-01T00:00:00+00:00',
                         completed_utc='1970-01-01T00:00:00.518+00:00')
            blocks, skipped = trial_event_blocks(path, {'events':[event]})
            self.assertEqual(blocks[0]['source_row_stop_exclusive'], 51)
            self.assertFalse(skipped)
            event['completed_utc'] = '1970-01-01T00:00:00.550+00:00'
            with self.assertRaisesRegex(ValueError, 'missing more than two sample ticks'):
                trial_event_blocks(path, {'events':[event]})
            event['completed_utc'] = '1970-01-01T00:00:00.100+00:00'
            blocks, skipped = trial_event_blocks(path, {'events':[event]})
            self.assertFalse(blocks)
            self.assertEqual(skipped[0]['reason'], 'fewer than 50 real samples')


if __name__ == '__main__':
    unittest.main()
