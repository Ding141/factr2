#!/usr/bin/python3.10
"""One real W3 capture. Never enables motors, switches modes or publishes targets."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[1]
W3 = Path('/home/dingyj/w3_dual_arm_ws')


def runtime_environment(domain):
    env = dict(os.environ)
    env['ROS_DOMAIN_ID'] = str(domain)
    # Match ordinary W3 terminals; next.sh otherwise defaults to localhost=1.
    env.setdefault('ROS_LOCALHOST_ONLY', '0')
    env.setdefault('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')
    return env


def w3_command(args, env):
    clean = ['/usr/bin/env', '-i', f'HOME={Path.home()}', 'PATH=/usr/bin:/bin',
             'LANG=C.UTF-8', 'PYTHONNOUSERSITE=1', f'ROS_LOG_DIR={W3 / "log/ros"}',
             f'TMPDIR={W3 / "log/tmp"}']
    for key in ('ROS_DOMAIN_ID', 'ROS_LOCALHOST_ONLY', 'RMW_IMPLEMENTATION',
                'FASTRTPS_DEFAULT_PROFILES_FILE'):
        if key in env:
            clean.append(f'{key}={env[key]}')
    return clean + ['/bin/bash', '--noprofile', '--norc', '-c',
                    'set -e; quick_local_only=${ROS_LOCALHOST_ONLY:-0}; '
                    'source /opt/ros/humble/setup.bash; '
                    'source /home/dingyj/w3_dual_arm_ws/install/local_setup.bash; '
                    'export ROS_LOCALHOST_ONLY=$quick_local_only; '
                    'exec "$@"', 'bash', *map(str, args)]


def next_command(args):
    return ['/bin/bash', str(ROOT / 'scripts/next.sh'), *map(str, args)]


def run(command, env, timeout=15):
    result = subprocess.run(command, env=env, cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stdout.strip() or f'Command failed: {command}')
    return result.stdout


def controller_is_active(text, name):
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    return any(line.split() and line.split()[0] == name and
               line.split()[-1] == 'active' for line in text.splitlines())


def parameters(text):
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError('Invalid parameter dump')
    for node in data.values():
        if isinstance(node, dict) and isinstance(node.get('ros__parameters'), dict):
            return node['ros__parameters']
    raise ValueError('Parameter dump has no ros__parameters')


def calibration_from_description(description):
    root = ET.fromstring(description)
    paths = {p.text.strip() for p in root.findall('./ros2_control/hardware/param')
             if p.get('name') == 'offsets_yaml' and p.text}
    if len(paths) != 1:
        raise ValueError('Cannot identify the actual offsets_yaml in robot_description')
    path = Path(paths.pop()).expanduser()
    if not path.is_absolute():
        raise ValueError('Actual calibration path must be absolute')
    if any('gripper_base_link' in link.get('name', '') for link in root.findall('link')):
        raise ValueError('This quick flow expects gripper=false; use the full guide for grippers')
    return path


def prepare(session, side, snapshots, load, tool, motion=None):
    audit = session / 'audit'
    audit.mkdir(parents=True)
    for name, text in snapshots.items():
        (audit / name).write_text(text)
    calibration = calibration_from_description(
        parameters(snapshots['robot_description.yaml'])['robot_description'])
    digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
    profile = ROOT / 'data/quick_capture/profiles' / digest
    profile.mkdir(parents=True, exist_ok=True)
    frozen = profile / 'calibration.yaml'
    if not frozen.exists():
        shutil.copy2(calibration, frozen)
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != digest:
        raise ValueError('Frozen calibration has changed')
    gravity = parameters(snapshots['gravity.yaml'])
    friction_path = gravity.get('friction_model_yaml')
    friction = {}
    if friction_path and Path(friction_path).is_file():
        source = Path(friction_path)
        shutil.copy2(source, audit / 'friction_model.yaml')
        friction = {'friction_model': str(audit / 'friction_model.yaml'),
                    'friction_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    cfg = yaml.safe_load((ROOT / f'config/w3/{side}/record.yaml').read_text())
    cfg.update(output_dir=str(session), session_name=session.name)
    cfg['metadata'] = dict(
        source='real', tool=tool, gripper='absent', load=load,
        calibration={'path': str(frozen)},
        control={'gains': {'position_params': str(audit / 'position.yaml')},
                 'feedforward': {'gravity_params': str(audit / 'gravity.yaml'), **friction}},
        trajectory_id='terminal_joint_targets',
        contact={'present': False, 'label': 'operator_declared_free_motion'},
        temperature={'source': 'unknown_no_sensor', 'condition': 'not_recorded'},
        audit_paths=[str(audit / 'bag')], health_gate_enabled=True)
    if motion is not None:
        plan, raw, digest = motion
        (audit / 'motion.yaml').write_bytes(raw)
        cfg['metadata']['trajectory_id'] = f'{plan["name"]}:sha256:{digest}'
        cfg['metadata']['audit_paths'] += [str(audit / 'motion.yaml'), str(audit / 'motion_run.json')]
    (session / 'record.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))


def summarize(session, env):
    files = sorted(session.glob('*.h5'))
    if len(files) != 1:
        print(f'未得到唯一 H5，检查日志：{session}', flush=True)
        return 1
    result = subprocess.run(next_command(['ros2', 'run', 'factr2_next', 'next_check_h5',
        'check', files[0], '--json', session / 'quality.json']),
        cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=60)
    (session / 'quality.log').write_text(result.stdout)
    report = json.loads((session / 'quality.json').read_text())
    print(f'\nH5：{files[0]}\n质量报告：{session / "quality.json"}')
    for ep, info in report['episodes'].items():
        print(f"{ep}: {info.get('rows', 0)} 行，{info.get('hz', 0):.2f} Hz，"
              f"最大间隔 {info.get('max_gap_seconds', 0)*1000:.2f} ms")
    print('质量检查：' + ('PASS' if report['accepted'] else 'FAIL'))
    if report['errors']:
        print('\n'.join(report['errors']))
    if report['accepted']:
        # Report actual motion coverage without mistaking a static PASS for training coverage.
        coverage = run(next_command(['python', '-c',
            'import h5py,numpy as np,sys; '
            'h=h5py.File(sys.argv[1],"r"); '
            '[(print(e,"Δq(deg)=",np.round(np.ptp(h[e]["joint_pos/data"][:],axis=0)*180/np.pi,3),'
            '"Δq_cmd(deg)=",np.round(np.ptp(h[e]["joint_cmd/data"][:],axis=0)*180/np.pi,3))) '
            'for e in h]', files[0]]), env)
        print(coverage.strip())
    motion_ok = True
    if (session / 'audit/motion.yaml').exists():
        from w3_motion_sequence import load_motion
        plan, _, digest = load_motion(session / 'audit/motion.yaml')
        try:
            motion = json.loads((session / 'audit/motion_run.json').read_text())
            motion_ok = (motion.get('status') == 'completed' and motion.get('motion_sha256') == digest and
                         len(motion.get('events', [])) == len(plan['steps']) and
                         all(e.get('status') == 'completed' for e in motion['events']))
        except (OSError, ValueError):
            motion_ok = False
        print('整套动作执行：' + ('PASS' if motion_ok else 'FAIL（未执行或中断，查看 audit/motion_run.json）'))
    return 0 if report['accepted'] and motion_ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', choices=['left', 'right'], default='left')
    parser.add_argument('--domain', type=int, default=int(os.environ.get('ROS_DOMAIN_ID', '0')))
    parser.add_argument('--load', default='empty_v1', help='Operator-declared actual load identity')
    parser.add_argument('--tool', default='bare_attachment_v1')
    parser.add_argument('--motion', type=Path, help='Freeze the motion file for terminal D; C never publishes it')
    args = parser.parse_args()
    motion = None
    if args.motion is not None:
        from w3_motion_sequence import load_motion
        try:
            motion = load_motion(args.motion)
            if motion[0]['side'] != args.side:
                raise ValueError('--side must match the motion file side')
        except (ValueError, OSError, yaml.YAMLError) as exc:
            parser.error(str(exc))
    import sys
    if not sys.stdin.isatty():
        parser.error('Run capture in an interactive terminal; Enter stops this session')
    env = runtime_environment(args.domain)
    for folder in (W3 / 'log/ros', W3 / 'log/tmp'):
        folder.mkdir(parents=True, exist_ok=True)
    children = []
    handles = []
    started = False
    interrupted = False
    session = ROOT / 'data/quick_capture' / args.side / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session.mkdir(parents=True)
    active_file = session.parent / 'active.json'
    active_info = None
    service = f'/factr2/{args.side}/record'

    def recording(value):
        return run(next_command(['ros2', 'service', 'call', service,
            'std_srvs/srv/SetBool', '{data: ' + str(value).lower() + '}']), env, timeout=4)

    def launch(label, command):
        handle = (session / f'{label}.log').open('w')
        handles.append(handle)
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        children.append((label, process))

    def stop(label):
        for child_label, process in children:
            if child_label == label and process.poll() is None:
                if label == 'adapter':
                    # ros2 launch forwards SIGINT to its child; avoid delivering it twice.
                    process.send_signal(signal.SIGINT)
                else:
                    os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print(f'{label} 尚未退出，保留进程及日志：{session}；不强杀。', flush=True)
                    return False
        return True

    print(f'ROS domain={args.domain}，side={args.side}\n本次目录：{session}', flush=True)
    try:
        print('正在只读检查现有 W3 控制器（不会启动控制端或切换模式）…', flush=True)
        controllers = run(w3_command(['ros2', 'control', 'list_controllers'], env), env)
        if not all(controller_is_active(controllers, name) for name in
                   ('joint_state_broadcaster', 'gravity_compensation_controller', 'joint_position_controller')):
            raise RuntimeError('请先在 W3 网页启动控制端并切到“关节位置”；本脚本不会代你启动或切模式。\n' + controllers)
        if controller_is_active(controllers, 'cartesian_position_controller'):
            raise RuntimeError('本流程要求关节位置模式，笛卡尔控制器应 inactive。')
        snapshots = {'controllers.txt': controllers}
        print('正在保存实际参数和标定…', flush=True)
        for filename, node in [('position.yaml', '/joint_position_controller'),
                               ('gravity.yaml', '/gravity_compensation_controller'),
                               ('robot_description.yaml', '/robot_state_publisher')]:
            snapshots[filename] = run(w3_command(['ros2', 'param', 'dump', node], env), env)
        prepare(session, args.side, snapshots, args.load, args.tool, motion)
        run(next_command(['python', 'scripts/check_w3_configs.py', '--runtime', 'record',
                          '--config', session / 'record.yaml']), env)
        # Refuse to duplicate existing capture components on this side.
        graph = run(w3_command(['ros2', 'node', 'list', '--no-daemon'], env), env)
        if any(name in graph.splitlines() for name in
               ('/w3_health_monitor', f'/w3_next_adapter_{args.side}', f'/quick_recorder_{args.side}')):
            raise RuntimeError('已有健康器/本侧 adapter/quick recorder，请先停止旧采集进程再运行。')
        print('正在启动只读健康器、real adapter、bag 和 recorder，等待新鲜四流…', flush=True)
        launch('health', w3_command(['/usr/bin/python3.10',
               ROOT / 'factr2_w3_adapter/tools/w3_health_monitor.py'], env))
        launch('adapter', next_command(['ros2', 'launch', 'factr2_w3_adapter', 'adapter.launch.py',
               f'side:={args.side}', 'profile:=real']))
        root = f'/factr2/{args.side}'
        topics = ['/w3_robot_bridge_node/state', '/joint_states',
                  '/joint_position_controller/command_state', '/factr2/w3_health']
        topics += [root + '/' + key for key in
                   ('adapter_status', 'joint_pos', 'joint_vel', 'joint_cmd', 'joint_effort')]
        launch('bag', w3_command(['ros2', 'bag', 'record', '-o', session / 'audit/bag', *topics], env))
        launch('recorder', next_command(['ros2', 'run', 'factr2_next', 'next_record', '--ros-args',
               '-p', f'config_file:={session / "record.yaml"}', '-r', f'__node:=quick_recorder_{args.side}']))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            failed = [label for label, proc in children if proc.poll() is not None]
            if failed:
                raise RuntimeError(f'进程提前退出：{failed}，请看本次目录中的对应 .log')
            try:
                reply = recording(True)
                if re.search(r'success\s*[:=]\s*[Tt]rue', reply):
                    started = True
                    break
            except (subprocess.TimeoutExpired, RuntimeError):
                pass
            time.sleep(.5)
        if not started:
            raise RuntimeError('30 秒内无法开录：检查 domain、位置模式、health.log 和 adapter.log。')
        if motion is not None:
            from w3_motion_sequence import write_json
            active_info = {'pid': os.getpid(), 'session': str(session), 'side': args.side,
                           'domain': args.domain, 'motion_sha256': motion[2]}
            write_json(active_file, active_info)
        print('\n已开始录制。现在可在另一个终端发送关节目标。\n运动结束后回到这里按 Enter 停录并检查。', flush=True)
        if motion is not None:
            print(f'D 请执行动作文件：{args.motion.resolve()}\n本次只接受这份文件一次；动作内容已冻结到 audit/motion.yaml。', flush=True)
        while True:
            failed = [label for label, proc in children if proc.poll() is not None]
            if failed:
                raise RuntimeError(f'采集中进程退出：{failed}；本次不视为完整成功，请检查日志。')
            ready, _, _ = select.select([sys.stdin], [], [], .2)
            if ready:
                sys.stdin.readline()
                break
    except KeyboardInterrupt:
        interrupted = True
        print('\n收到 Ctrl-C，正在正常关闭本次采集。', flush=True)
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f'采集失败：{exc}\n日志目录：{session}', flush=True)
        interrupted = True
    finally:
        if active_info is not None:
            try:
                if json.loads(active_file.read_text()) == active_info:
                    active_file.unlink()
            except (OSError, ValueError):
                pass
        if started:
            try:
                recording(False)
            except (RuntimeError, OSError, subprocess.TimeoutExpired):
                print('停录服务无响应，将通过 recorder 的 SIGINT 正常关闭 H5。', flush=True)
        closed = stop('recorder')
        bag_closed = stop('bag')
        stop('adapter')
        stop('health')
        for handle in handles:
            handle.close()
    if started and closed and bag_closed:
        result = summarize(session, env)
        return result if not interrupted else 1
    return 1 if interrupted or not started else 0


if __name__ == '__main__':
    raise SystemExit(main())
