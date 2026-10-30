"""Run the frozen compensation preview and comparison window; never moves arms."""
import argparse
import json
import os
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor
import yaml

from factr2_next.data_collection.quality import write_json
from factr2_next.inference.kinematic_compensation import ROOT, SCHEMA, digest, ResidualModel
from factr2_next.inference.kinematic_observer import KinematicObserver
from factr2_next.visualization.web_node import WebNode


def web_config(port):
    return dict(contract_version='w3_next_v1',side='right',joint_names=[f'right_joint_{j}' for j in range(7)],
        next_topic_root=ROOT,robot_topic_root='/factr2/right',observation_mode='kinematic_residual_research',
        outputs={k:'{next_topic_root}/'+v for k,v in dict(
            external_joint_torque='external_joint_torque',external_joint_torque_raw='external_joint_torque/raw',
            free_joint_torque_pred='free_joint_torque_pred',mse='mse',
            normalized_contact_magnitude='normalized_contact_magnitude',status='status',
            feedback_torque='upstream_external_joint_torque/raw').items()},
        web=dict(host='127.0.0.1',port=port,refresh_hz=10),
        plot=dict(view='joint_torque',max_points=1000,torque_display=dict(
            primary='ext',storage_prefix='next.kinematic_compensated.',
            raw_label='补偿后 · 输出滤波前（N·m）',filtered_label='补偿后 · 最终平滑输出（N·m）',
            reference_label='补偿前 · NEXT 原始残差对照（N·m）',
            description='默认显示最终平滑关节外力矩，可展开补偿前 NEXT 曲线与输出滤波前曲线对照。'
                        '输入力矩与自由预测采用相同 12 Hz 因果滤波，最终输出为 5 Hz 因果低通。'
                        '新增补偿分支使用位置卡尔曼滤波估计速度、加速度；原 LSTM 输入保持训练时口径。'
                        'J1–J7 对应右臂实际关节，单位 N·m。')),
        model_label='NEXT · 速度/加速度补偿与平滑输出（局部研究观察）',
        model_notice='补偿由空载记录离线学习，仅在所学姿态与运动特征范围内启用；越界回退到未补偿残差并标记。'
                     '稳态持续力矩不会被自动归零；滤波会延迟并削弱短脉冲。尚未验证真实接触，原 sigma 不适用于此输出。')


def verify(experiment):
    manifest=json.loads((experiment/'manifest.json').read_text())
    if manifest['schema']!=SCHEMA:
        raise ValueError('wrong experiment schema')
    for file, expected in manifest['files'].items():
        if Path(file).is_absolute() or '..' in Path(file).parts or digest(experiment/file)!=expected:
            raise ValueError('experiment artifact hash mismatch: '+file)
    recipe=json.loads((experiment/'recipe.json').read_text());base=Path(recipe['base'])
    model=ResidualModel.load(experiment/'model.json',digest(base/'metadata.json'))
    if not any(model.enabled):raise ValueError('no validated correction to activate')
    return base,model


def main():
    # Keep the familiar window entry point following an explicitly pinned mean upgrade.
    import sys
    active=Path(__file__).resolve().parents[1]/'config/w3/right/observer_active.json'
    explicit_experiment=any(arg=='--experiment' or arg.startswith('--experiment=') for arg in sys.argv[1:])
    if not explicit_experiment and active.is_file():
        profile=json.loads(active.read_text())
        if profile.get('kind')=='joint_lstm':
            from observe_w3_joint_lstm import main as joint_main
            return joint_main()
        raise ValueError('unsupported active observer profile kind')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,
        help='Explicit model directory; defaults to the pinned right-arm compensation profile')
    p.add_argument('--port',type=int,default=8083);p.add_argument('--state-file',type=Path)
    p.add_argument('--model-label',help='Visible experiment label; does not change the compensation model')
    p.add_argument('--deployment-note',help='Visible deployment limitation; does not alter model values or validity gates')
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.experiment is None:
        profile_path=Path(__file__).resolve().parents[1]/'config/w3/right/compensation_active.json'
        profile=json.loads(profile_path.read_text())
        if profile.get('schema')!='w3_compensation_profile_v1' or profile.get('research_only') is not True:
            raise ValueError('invalid active compensation profile')
        if not a.check_only and os.environ.get('ROS_DOMAIN_ID')!=str(profile['ros_domain_id']):
            p.error('Active right-arm profile requires ROS_DOMAIN_ID='+str(profile['ros_domain_id']))
        a.experiment=Path(profile['experiment'])
        if digest(a.experiment/'model.json')!=profile['model_sha256']:
            raise ValueError('active compensation profile hash mismatch')
        if not a.model_label:a.model_label=profile['model_label']
        if not a.deployment_note:a.deployment_note=profile['deployment_note']
    if not 1<=a.port<=65535:p.error('invalid port')
    exp=a.experiment.resolve();base,model=verify(exp);cfg=web_config(a.port)
    recipe=json.loads((exp/'recipe.json').read_text())
    if recipe.get('source_mean_experiment'):
        cfg['model_label']='NEXT · 高速连续数据重训与速度/加速度补偿'
    if model.mode.startswith('position_'):
        cfg['model_label']='NEXT · 姿态与速度/加速度耦合补偿'
        cfg['plot']['torque_display']['description'] += (' 新版加入姿态与运动学耦合项；'
            '非线性分支使用连续速度权重调节补偿量，完整外力矩不设死区、不自动归零。')
    if model.fallback is not None:
        cfg['model_notice']=('新增模型超域时先回退到原补偿；两者均超域才回退到未补偿残差。'
            '各分支保持原限幅、输入输出滤波和基模型；尚未完成真实接触标定，sigma不适用于补偿后输出。')
    if a.model_label:
        cfg['model_label']=a.model_label
    if a.deployment_note:
        cfg['model_notice']=a.deployment_note+' '+cfg['model_notice']
    if a.check_only:
        print(json.dumps(dict(status='checked',url=f'http://127.0.0.1:{a.port}/',enabled_joints=model.enabled.tolist(),
            source_model_metadata_sha256=model.artifact['base_metadata_sha256'],
            compensation_model_sha256=digest(exp/'model.json'),feature_mode=model.mode,
            deployment_ready=False),indent=2));return
    rclpy.init();node=web=None;executor=SingleThreadedExecutor()
    try:
        node=KinematicObserver(exp/'model.json',base,use_global_arguments=False)
        web=WebNode(cfg,node_name='right_kinematic_compensation_dashboard',use_global_arguments=False)
        executor.add_node(node);executor.add_node(web)
        def status():
            if a.state_file:
                write_json(a.state_file,dict(service='running',pid=os.getpid(),url=f'http://127.0.0.1:{a.port}/',
                    state=node.state,reason=node.reason,outputs=node.counts['outputs'],experiment=str(exp),
                    next_topic_root=ROOT,enabled_joints=model.enabled.tolist(),
                    compensation_applied=bool(node.latest and node.latest['compensation_applied']),
                    compensation_model_sha256=node.model_hash,feature_mode=model.mode,
                    calibration='uncalibrated',deployment_ready=False,contact_validated=False))
        web.create_timer(1.,status);status()
        print(f'Observation only: http://127.0.0.1:{a.port}/ ; base comparison remains at :8082',flush=True)
        executor.spin()
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        executor.shutdown()
        if web is not None:web.destroy_node()
        if node is not None:node.destroy_node()
        rclpy.try_shutdown()


if __name__=='__main__':main()
