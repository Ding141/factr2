#!/usr/bin/env python3
"""Run the frozen-mean sigma observer and dashboard; no robot control APIs."""
import argparse
import json
import os
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
import yaml

from factr2_next.data_collection.quality import sha256, write_json
from factr2_next.inference.frozen_sigma_observer import ROOT, FrozenSigmaInferenceNode, load_observer
from factr2_next.visualization.frozen_sigma_web import FrozenSigmaWeb
from factr2_next.w3_config import validate_w3_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--port',type=int)
    parser.add_argument('--state-file',type=Path)
    parser.add_argument('--deployment-note',help='Visible observation limitation or experiment mode; never changes inference')
    parser.add_argument('--model-label',help='Dashboard title for the current observation experiment')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    bundle=args.bundle.resolve()
    loaded=load_observer(bundle)
    cfg=yaml.safe_load((bundle/'inference.yaml').read_text())
    web_cfg=yaml.safe_load((bundle/'web.yaml').read_text())
    if args.deployment_note:web_cfg['model_notice']=args.deployment_note+' '+web_cfg.get('model_notice','')
    if args.model_label:web_cfg['model_label']=args.model_label
    if args.port is not None:web_cfg['web']['port']=args.port
    if (web_cfg['web']['host']!='127.0.0.1' or not 1<=web_cfg['web']['port']<=65535
            or cfg['next_topic_root']!=ROOT or web_cfg['next_topic_root']!=ROOT
            or Path(cfg['checkpoint_dir']).resolve()!=bundle
            or cfg['operating_envelope']!=loaded.metadata['operating_envelope']):
        raise ValueError('frozen_sigma_observation_config_mismatch')
    validate_w3_config(cfg,'inference');validate_w3_config(web_cfg,'visualize')
    if args.check_only:
        print(json.dumps(dict(status='checked',schema=loaded.metadata['schema'],next_topic_root=ROOT,
            deployment_mode=loaded.metadata['deployment_mode'],calibration='uncalibrated',
            url=f"http://127.0.0.1:{web_cfg['web']['port']}/",deployment_ready=False),indent=2))
        return
    rclpy.init()
    observer=web=None
    executor=SingleThreadedExecutor()
    try:
        observer=FrozenSigmaInferenceNode(parameter_overrides=[Parameter('config_file',value=str(bundle/'inference.yaml'))],
            cli_args=['--ros-args','-r','__node:=next_frozen_sigma_right'],use_global_arguments=False)
        web=FrozenSigmaWeb(web_cfg,node_name='right_frozen_sigma_dashboard',use_global_arguments=False)
        executor.add_node(observer);executor.add_node(web)
        def status():
            value=dict(service='running',pid=os.getpid(),deployment_ready=False,calibration='uncalibrated',
                state=observer.state,reason=observer.reason,outputs=observer.counts['outputs'],
                next_topic_root=ROOT,url=f"http://127.0.0.1:{web_cfg['web']['port']}/",
                bundle=str(bundle),metadata_sha256=sha256(bundle/'metadata.json'))
            if args.state_file:write_json(args.state_file,value)
        web.create_timer(1.,status)
        status()
        executor.spin()
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        if web is not None:web.destroy_node()
        if observer is not None:observer.destroy_node()
        rclpy.try_shutdown()


if __name__=='__main__':main()
