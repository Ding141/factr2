"""Run the active independent-joint LSTM observer and torque dashboard."""
import argparse
import json
import os
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
import yaml

from factr2_next.data_collection.quality import sha256, write_json
from factr2_next.inference.joint_lstm_observer import load_bundle, JointLSTMInferenceNode, ROOT
from factr2_next.visualization.web_node import WebNode

PROFILE = Path(__file__).resolve().parents[1]/'config/w3/right/observer_active.json'


def resolve_bundle(explicit=None, check_only=False):
    if explicit is not None:
        return Path(explicit).resolve()
    profile = json.loads(PROFILE.read_text())
    if profile.get('schema') != 'w3_observer_profile_v1' or profile.get('kind') != 'joint_lstm':
        raise ValueError('joint_lstm_active_profile')
    if not check_only and os.environ.get('ROS_DOMAIN_ID') != str(profile['ros_domain_id']):
        raise ValueError('active observer requires ROS_DOMAIN_ID='+str(profile['ros_domain_id']))
    bundle = Path(profile['bundle']).resolve()
    if sha256(bundle/'bundle.json') != profile['bundle_sha256']:
        raise ValueError('joint_lstm_active_hash')
    return bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path); p.add_argument('--port', type=int, default=8083)
    p.add_argument('--also-port', type=int); p.add_argument('--state-file', type=Path)
    p.add_argument('--model-label'); p.add_argument('--deployment-note')
    p.add_argument('--check-only', action='store_true'); a = p.parse_args()
    bundle = resolve_bundle(a.bundle, a.check_only)
    profile=json.loads(PROFILE.read_text()) if a.bundle is None else {}
    residual=profile.get('residual_compensation')
    if residual and residual.get('enabled'):
        path=Path(residual['model'])
        if sha256(path)!=residual['model_sha256']:raise ValueError('active_residual_hash')
        if not a.check_only:
            if a.port!=8083 or a.also_port not in (None,8082):
                p.error('Active residual comparison uses ports 8082 and 8083')
            from observe_w3_model_comparison import main as comparison_main
            state=a.state_file or path.parent.parent/'service_state.json'
            return comparison_main(['--state-file',str(state),'--residual-model',str(path),'--cpu-threads','1'])
    loaded, meta = load_bundle(bundle)
    if any(not 1 <= port <= 65535 for port in (a.port, a.also_port) if port is not None):
        p.error('invalid port')
    if a.also_port == a.port: p.error('duplicate dashboard port')
    if a.check_only:
        print(json.dumps(dict(status='checked',bundle=str(bundle),architecture=meta['architecture'],
            model_sha256=sha256(bundle/'mean_checkpoint/model.pt'),compensation_enabled=bool(residual and residual.get('enabled')),
            residual_model=residual,
            controller_changes=False,next_topic_root=ROOT,deployment_ready=False),indent=2)); return
    rclpy.init(); executor = SingleThreadedExecutor(); node = None; webs = []
    try:
        node = JointLSTMInferenceNode(bundle, parameter_overrides=[
            Parameter('config_file', value=str(bundle/'inference.yaml'))],
            cli_args=['--ros-args', '-r', '__node:=next_joint_lstm_right'], use_global_arguments=False)
        executor.add_node(node)
        for port in (a.port, a.also_port):
            if port is None: continue
            cfg = yaml.safe_load((bundle/'web.yaml').read_text()); cfg['web']['port'] = port
            if a.model_label: cfg['model_label'] = a.model_label
            if a.deployment_note: cfg['model_notice'] = a.deployment_note+' '+cfg['model_notice']
            web = WebNode(cfg, node_name='right_joint_lstm_dashboard_'+str(port), use_global_arguments=False)
            webs.append(web); executor.add_node(web)
        def status():
            if a.state_file:
                write_json(a.state_file,dict(service='running',pid=os.getpid(),bundle=str(bundle),
                    bundle_sha256=node.bundle_metadata_hash,model_sha256=sha256(bundle/'mean_checkpoint/model.pt'),
                    state=node.state,reason=node.reason,outputs=node.counts['outputs'],next_topic_root=ROOT,
                    url=f'http://127.0.0.1:{a.port}/',compensation_enabled=False,controller_changes=False,
                    architecture=meta['architecture'],deployment_ready=False))
        webs[0].create_timer(1.,status); status()
        executor.spin()
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException): pass
    finally:
        executor.shutdown()
        for web in reversed(webs): web.destroy_node()
        if node is not None: node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__': main()
