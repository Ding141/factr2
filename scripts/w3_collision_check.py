#!/usr/bin/env python3
"""Read-only mesh checks of a frozen coverage plan. Never publishes ROS commands.

Fixed bodies and directly adjacent moving bodies contain intended CAD overlap.
A disabled arm's pose is supplied explicitly; this is not a live measurement.
External supports and cables are absent from the URDF and need site clearance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pinocchio as pin
import yaml


def check(plan_path, workspace, left_pose_deg, report_path, step_deg=.25):
    raw=plan_path.read_bytes();plan=yaml.load(raw,Loader=getattr(yaml,'CSafeLoader',yaml.SafeLoader))
    if plan['side']!='right':raise ValueError('This checker is for the right arm with a static left arm')
    description=plan['robot_description'];model=pin.buildModelFromXML(description)
    with tempfile.TemporaryDirectory(prefix='w3-collision-') as temporary:
        path=Path(temporary)/'robot.urdf';path.write_text(description)
        geometry=pin.buildGeomFromUrdf(model,str(path),pin.GeometryType.COLLISION,
            package_dirs=[str(workspace/'src/description')])
    excluded=[]
    for first,a in enumerate(geometry.geometryObjects):
        for second,b in enumerate(geometry.geometryObjects[:first]):
            adjacent=a.parentJoint==b.parentJoint
            if a.parentJoint and b.parentJoint:
                adjacent|=(model.parents[a.parentJoint]==b.parentJoint or model.parents[b.parentJoint]==a.parentJoint)
            for moving,fixed in ((a,b),(b,a)):
                for side in ('left','right'):
                    adjacent|=(moving.name==f'{side}_link_1_0' and fixed.name in
                        (f'{side}_base_link_0',f'{side}_arm_attachment_point_0'))
            left_only=a.name.startswith('left_') and b.name.startswith('left_')
            if adjacent or left_only:
                excluded.append({'pair':[a.name,b.name],'reason':'rigid/adjacent body' if adjacent else 'static left-arm pair'})
                continue
            geometry.addCollisionPair(pin.CollisionPair(first,second))
    data=model.createData();geometry_data=geometry.createData()
    q=pin.neutral(model)
    for index,angle in enumerate(left_pose_deg):q[model.joints[model.getJointId(f'left_joint_{index}')].idx_q]=np.deg2rad(angle)
    right_indices=[model.joints[model.getJointId(name)].idx_q for name in plan['joint_names']]
    checked=0;distance_checked=0;last=None;minimum=float('inf');closest=None;failure=None;t0=time.monotonic()
    for step in plan['steps']:
        positions=np.asarray(step['positions_deg'])
        for index,position in enumerate(positions):
            if last is not None and index not in (0,len(positions)-1) and np.abs(position-last).sum()<step_deg:continue
            q[right_indices]=np.deg2rad(position)
            pin.computeCollisions(model,data,geometry,geometry_data,q,False);checked+=1
            for pair,result in zip(geometry.collisionPairs,geometry_data.collisionResults):
                if result.isCollision():
                    failure={'step':step['name'],'time_in_step':step['times_seconds'][index],
                        'position_deg':position.tolist(),'pair':[geometry.geometryObjects[pair.first].name,geometry.geometryObjects[pair.second].name]}
                    break
            if failure:break
            # Mesh distance queries are much slower than collision queries.
            # Collision is checked at every selected sample; clearance is sampled separately.
            if checked%1000==1 or index in (0,len(positions)-1):
                distance_checked+=1
                pin.computeDistances(model,data,geometry,geometry_data,q)
                distances=[result.min_distance for result in geometry_data.distanceResults]
                nearest=int(np.argmin(distances))
                if distances[nearest]<minimum:
                    minimum=float(distances[nearest]);pair=geometry.collisionPairs[nearest]
                    closest={'pair':[geometry.geometryObjects[pair.first].name,geometry.geometryObjects[pair.second].name],
                        'step':step['name'],'position_deg':position.tolist()}
            last=position.copy()
        print(step['name'],checked,'min clearance m',round(minimum,6),flush=True)
        if failure:break
    report={'accepted':failure is None,'plan_sha256':hashlib.sha256(raw).hexdigest(),
        'robot_description_sha256':hashlib.sha256(description.encode()).hexdigest(),
        'left_pose_deg':left_pose_deg,'left_pose_source':'operator_confirmed_near_zero; modeled at zero',
        'max_sample_joint_travel_deg':step_deg,'checked_configurations':checked,
        'minimum_sampled_distance_m':minimum,'distance_configurations_checked':distance_checked,'closest':closest,'failure':failure,'excluded_pairs':excluded,
        'elapsed_seconds':time.monotonic()-t0,
        'limitations':['Discrete URDF mesh check, not continuous collision certification',
            'External support fixtures, cables, and pose uncertainty are not modeled',
            'Samples describe command targets; measured tracking must be monitored during capture']}
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    if failure:raise RuntimeError(f'Collision: {failure}')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path);parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--left-pose-deg',nargs=7,type=float,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    check(args.plan,args.workspace,args.left_pose_deg,args.report)

if __name__=='__main__':main()
