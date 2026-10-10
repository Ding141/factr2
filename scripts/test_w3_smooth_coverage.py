"""Geometry, interpolation and frequency regressions without robot operations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import yaml
from test_w3_coverage_motion import fixture_description
from w3_coverage_plan import ArmModel, build_plan, validate_plan
from w3_coverage_motion import execute_continuous

ROOT=Path(__file__).resolve().parents[1]

class SmoothTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=yaml.safe_load((ROOT/'config/w3/motions/right_coverage_smooth.yaml').read_text())
        cls.config['repeats']=1
        cls.plan=build_plan(cls.config,fixture_description(),[0]*7)

    def test_ranges_both_speeds_no_internal_stops(self):
        for s in self.plan['steps']:
            self.assertEqual(s['hold_seconds'],0)
            if s['phase']!='single_joint':continue
            q=np.asarray(s['positions_deg'])[:,s['joint']]
            np.testing.assert_allclose([q.min(),q.max()],self.config['working_bounds_deg'][s['joint']],atol=.002)
            v=np.diff(q)/.01
            # Smooth reversals; velocity stays nonzero through the center crossing.
            mid=len(v)//2
            self.assertGreater(abs(v[mid]),2 if s['speed']=='slow' else 8)
            self.assertLess(abs(v[0]),.001);self.assertLess(abs(v[-1]),.001)

    def test_joined_controller_linear_interpolation_acceleration_and_jerk(self):
        q=np.vstack([np.asarray(s['positions_deg'])[1:] for s in self.plan['steps']])
        v=np.diff(q,axis=0)/.01;a=np.diff(v,axis=0)/.01;j=np.diff(a,axis=0)/.01
        self.assertLess(np.abs(a).max(),35.001)
        self.assertLess(np.abs(j).max(),120.001)

    def test_cartesian_shape_from_generated_joint_curve(self):
        m=ArmModel(self.plan['robot_description'],'right','right_attachment_point')
        center=m.fk(np.deg2rad(self.config['home_deg']))[:3,3]
        for s in self.plan['steps']:
            if s['phase']!='cartesian':continue
            xyz=np.array([m.fk(np.deg2rad(q))[:3,3] for q in s['positions_deg'][::10]])
            if s['kind'].startswith('line_'):
                axis='xyz'.index(s['kind'][-1]);self.assertLess(np.abs(np.delete(xyz-center,axis,axis=1)).max(),.0001)
                self.assertGreater(np.ptp(xyz[:,axis]),.0158)
            else:
                c=center.copy();c[0]-=.008
                self.assertLess(np.abs(np.linalg.norm((xyz-c)[:,:2],axis=1)-.008).max(),.0001)

    def test_no_reinitialization_at_block_boundary_and_endpoint_fault_aborts(self):
        plan=copy.deepcopy(self.plan); calls=[]
        class Fake:
            state={'q':np.deg2rad(plan['steps'][0]['positions_deg'][-1])}
            def play(self,combined,guard,on_tick):
                calls.append(combined);on_tick(0)
                t=0.
                for step in plan['steps']:
                    t+=step['duration_seconds'];self.state['q']=np.deg2rad(step['positions_deg'][-1]);on_tick(t+1e-8)
        report={'events':[]};execute_continuous(plan,Fake(),lambda:None,report,lambda:None)
        self.assertEqual(len(calls),1);self.assertEqual(report['status'],'completed')
        self.assertEqual(len(report['events']),len(plan['steps']))
        class Bad(Fake):
            def play(self,combined,guard,on_tick):
                on_tick(0);self.state['q']=np.ones(7);on_tick(plan['steps'][0]['duration_seconds'])
        with self.assertRaisesRegex(RuntimeError,'tracking'):execute_continuous(plan,Bad(),lambda:None,{'events':[]},lambda:None)

    def test_moving_boundary_uses_tracking_cap_without_relaxing_final_arrival(self):
        plan=copy.deepcopy(self.plan)
        class Moving:
            state={}
            def play(self,combined,guard,on_tick):
                self.state={'q':np.deg2rad(plan['steps'][0]['positions_deg'][0])}
                on_tick(0);t=0.
                for step in plan['steps']:
                    t+=step['duration_seconds']
                    goal=np.asarray(step['positions_deg'][-1]);self.state['cmd']=np.deg2rad(goal)
                    self.state['q']=np.deg2rad(goal+2.)
                    on_tick(t+1e-8)
                # This models CoverageClient.play's final stopped-arrival check.
                if np.max(np.abs(np.rad2deg(self.state['q']-self.state['cmd'])))>plan['config']['arrival_tolerance_deg']:
                    raise RuntimeError('final stopped arrival error')
        report={'events':[]}
        with self.assertRaisesRegex(RuntimeError,'final stopped arrival'):
            execute_continuous(plan,Moving(),lambda:None,report,lambda:None)
        self.assertEqual(len(report['events']),len(plan['steps']))
        self.assertTrue(all(e['status']=='completed' for e in report['events']))
        self.assertNotIn('status',report)

    def test_continuous_repetitions_cross_home_without_stopping(self):
        config=copy.deepcopy(self.config);config['repeats']=2
        config['continuous_repetitions']=True;config['finish_at_home']=True
        config['home_deg'][0]=3. # asymmetric home must still reach both bounds
        plan=build_plan(config,fixture_description(),[0]*7)
        from w3_coverage_motion import trajectory_derivatives
        q=np.vstack([np.asarray(s['positions_deg'])[1:] if i else np.asarray(s['positions_deg']) for i,s in enumerate(plan['steps'])])
        t=np.arange(len(q))*.01;v,a=trajectory_derivatives(t,q)
        offset=0
        for i,step in enumerate(plan['steps']):
            qpart=np.asarray(step['positions_deg'])
            if step['phase']=='single_joint' and step['repetition']==1:
                axis=step['joint'];np.testing.assert_allclose([qpart[:,axis].min(),qpart[:,axis].max()],config['working_bounds_deg'][axis],atol=.003)
                boundary=offset+len(qpart)-1
                self.assertGreater(abs(v[boundary,axis]),.3)
                self.assertLess(abs(v[boundary,axis]-v[boundary-1,axis]),.02)
            offset+=len(qpart)-1
        np.testing.assert_allclose(q[-1],config['home_deg'],atol=1e-7)
        self.assertTrue(np.isfinite(v).all() and np.isfinite(a).all())

    def test_modified_trajectory_jerk_is_rejected(self):
        p=copy.deepcopy(self.plan);p['steps'][1]['positions_deg'][500][0]+=.1
        with self.assertRaises(ValueError):validate_plan(p)

if __name__=='__main__':unittest.main()
