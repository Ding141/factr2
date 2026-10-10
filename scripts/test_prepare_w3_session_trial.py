"""Offline checks for role isolation and the actual prepared trial provenance."""
import json
from pathlib import Path
import unittest
import numpy as np

from prepare_w3_session_trial import role, guarded_range, completed_events


class TrialSplitTests(unittest.TestCase):
    def test_audited_prefix_excludes_unfinished_repeat_and_rejects_other_faults(self):
        run={'status':'failed','error':'Continuous block endpoint tracking error [1.55]',
             'events':[{'name':'valid','status':'completed'},
                       {'name':'unfinished','status':'sending','phase':'single_joint','repetition':1}]}
        with self.assertRaises(ValueError):completed_events(run)
        with self.assertRaises(ValueError):completed_events(run,True,{'accepted':False})
        events,excluded=completed_events(run,True,{'accepted':True})
        self.assertEqual([e['name'] for e in events],['valid']);self.assertEqual(excluded,['unfinished'])
        run['error']='Configured tracking error limit exceeded'
        with self.assertRaises(ValueError):completed_events(run,True,{'accepted':True})

    def test_first_repetition_always_training(self):
        for phase in ('single_joint','cartesian'):
            for speed in ('slow','fast'):
                self.assertEqual(role({'phase':phase,'speed':speed,'repetition':1}), 'train')

    def test_second_repetition_complementary_speed_holdouts(self):
        for joint in range(7):
            roles={role({'phase':'single_joint','joint':joint,'repetition':2,'speed':speed}) for speed in ('slow','fast')}
            self.assertEqual(roles,{'val','test'})
        for kind in ('line_x_leg1','line_y_leg2','line_z_leg3','circle_xy_leg1'):
            roles={role({'phase':'cartesian','kind':kind,'repetition':2,'speed':speed}) for speed in ('slow','fast')}
            self.assertEqual(roles,{'val','test'})

    def test_reset_stays_with_its_path_and_unknown_repeat_rejected(self):
        for split in ('train','val','test'):
            self.assertEqual(role({'phase':'transition','kind':'reset_home'},split),split)
        with self.assertRaises(ValueError):
            role({'phase':'single_joint','joint':0,'repetition':3,'speed':'slow'})

    def test_short_transition_skipped_without_relaxing_excitation_guard(self):
        stamps=np.arange(1001,dtype=np.int64)*10_000_000
        group={'start_utc':'1970-01-01T00:00:00+00:00',
               'end_utc':'1970-01-01T00:00:02+00:00','phases':['transition']}
        self.assertIsNone(guarded_range(stamps,group,100))
        group['phases']=['cartesian','transition']
        with self.assertRaisesRegex(ValueError,'too short'):
            guarded_range(stamps,group,100)
        group['end_utc']='1970-01-01T00:00:10+00:00'
        self.assertEqual(guarded_range(stamps,group,100),(100,901))


if __name__=='__main__':unittest.main()
