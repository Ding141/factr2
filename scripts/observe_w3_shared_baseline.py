"""Preserved shared-LSTM mean, without residual compensation, for comparison."""
import argparse
import copy
import json
import os
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
import torch
import yaml

from factr2_next.data_collection.quality import sha256,write_json
from factr2_next.inference.inference_node import InferenceNode
from factr2_next.inference.frozen_sigma_observer import load_observer
from factr2_next.inference.joint_lstm_observer import ResidualCascade
from factr2_next.visualization.web_node import WebNode

REPO=Path(__file__).resolve().parents[1]
DEFAULT=REPO/'models/w3/right/wide_highspeed_observer_20261026_v1'


class SharedBaselineInferenceNode(InferenceNode):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.torque_filter=ResidualCascade()
        self.contact_cfg=dict(self.contact_cfg,enabled=False)

    def _load_checkpoint(self,path,**kwargs):
        loaded=load_observer(path,expected=kwargs.get('expected'))
        # The preserved mean is exactly this original LSTM/head; no sigma output.
        loaded.model=loaded.model.base
        loaded.model.requires_grad_(False);loaded.model.eval()
        return loaded


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=DEFAULT);p.add_argument('--port',type=int,default=8082)
    p.add_argument('--state-file',type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);bundle=a.bundle.resolve();loaded=load_observer(bundle)
    if not 1<=a.port<=65535:p.error('invalid port')
    if a.check_only:
        print(json.dumps(dict(status='checked',bundle=str(bundle),metadata_sha256=sha256(bundle/'metadata.json'),
            architecture='shared_lstm',compensation_enabled=False,signal='LP12 residual then LP5',controller_changes=False),indent=2));return
    if os.environ.get('ROS_DOMAIN_ID')!='74':p.error('Right-arm comparison requires ROS_DOMAIN_ID=74')
    cfg=yaml.safe_load((bundle/'web.yaml').read_text());cfg['web']['port']=a.port
    cfg['model_label']='旧模型 · 共享 LSTM · 无附加补偿'
    cfg['model_notice']='旧模型原始权重与归一化参数保持不变。附加残差补偿关闭；sigma 不作为本轮对比指标。'
    cfg['plot']['torque_display']=dict(primary='ext',storage_prefix='next.shared_baseline.',
        raw_label='旧模型 · 原始外力矩残差（N·m）',filtered_label='旧模型 · 平滑外力矩（N·m）',
        description='旧共享 LSTM 的空载力矩预测。外力矩为反馈减去预测；使用与新模型相同的 12 Hz、5 Hz 两级因果低通。'
                    '可展开原始残差作比较。新模型位于 8083，均不使用附加残差补偿。')
    rclpy.init();executor=SingleThreadedExecutor();node=web=None
    try:
        node=SharedBaselineInferenceNode(parameter_overrides=[Parameter('config_file',value=str(bundle/'inference.yaml'))],
            cli_args=['--ros-args','-r','__node:=next_shared_baseline_right'],use_global_arguments=False)
        web=WebNode(cfg,node_name='right_shared_baseline_dashboard',use_global_arguments=False)
        executor.add_node(node);executor.add_node(web)
        def status():
            if a.state_file:
                write_json(a.state_file,dict(service='running',pid=os.getpid(),bundle=str(bundle),
                    metadata_sha256=sha256(bundle/'metadata.json'),architecture='shared_lstm',
                    state=node.state,reason=node.reason,outputs=node.counts['outputs'],
                    next_topic_root=node.next_topic_root,url=f'http://127.0.0.1:{a.port}/',
                    compensation_enabled=False,sigma_displayed=False,controller_changes=False))
        web.create_timer(1.,status);status();executor.spin()
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        executor.shutdown()
        if web is not None:web.destroy_node()
        if node is not None:node.destroy_node()
        rclpy.try_shutdown()


if __name__=='__main__':main()
