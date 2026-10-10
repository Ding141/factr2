#!/usr/bin/env python3
"""Read-only right-arm endpoint wrench conversion and a separate dashboard."""
import argparse
from collections import deque
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import WrenchStamped
from message_filters import Subscriber, TimeSynchronizer
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import yaml

from w3_coverage_plan import ArmModel
from factr2_next.inference.endpoint_wrench import estimate_wrench, rotate_wrench_at_same_origin
from factr2_next.inference.operating_envelope import OperatingEnvelope


HTML = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>右臂末端力与力矩</title><style>
body{background:#101820;color:#edf3fa;font:15px system-ui;margin:24px auto;padding:0 24px;max-width:1150px}
h1{font-size:24px}p{line-height:1.7;color:#bfcddd}.status{padding:14px;background:#253b4b;border-radius:8px;margin:16px 0}
canvas{background:#16232e;border:1px solid #354656;border-radius:8px;width:100%;height:300px}
.values{display:flex;gap:25px;margin:12px 0 22px;flex-wrap:wrap}.hint{color:#f5cc79}a{color:#8bc8ff}
</style><h1>右臂末端等效力与力矩</h1>
<p>作用点与坐标轴：<b id="frame">right_attachment_point</b>（末端安装点）。假定外部接触作用在末端；符号沿用 NEXT 力矩残差，方向尚需实测确认。</p>
<div class="status" id="status">等待机器人数据…</div>
<h2>末端力矩 Mx / My / Mz · N·m</h2><canvas id="moment" width="1100" height="300"></canvas><div class="values" id="moment-values"></div>
<h2>末端力 Fx / Fy / Fz · N</h2><canvas id="force" width="1100" height="300"></canvas><div class="values" id="force-values"></div>
<p class="hint">不可用区间留空，不代表零外力。训练范围外、接近奇异点或数据过期时暂停末端估计。本页面不控制机器人。</p>
<p><a href="http://127.0.0.1:8081/">查看原始关节外力矩页面</a></p><script>
const colors=['#ff8181','#75e0a1','#8fbfff'];
function plot(id,s,keys,unit){
 const c=document.getElementById(id),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);
 const p={l:70,r:25,t:25,b:48},w=c.width-p.l-p.r,h=c.height-p.t-p.b;
 const t=s.history.t,v=keys.flatMap(k=>s.history[k]).filter(Number.isFinite);
 let lo=v.length?Math.min(0,...v):-1,hi=v.length?Math.max(0,...v):1;if(hi-lo<1e-6){lo-=1;hi+=1}
 const margin=(hi-lo)*.12;lo-=margin;hi+=margin;
 const a=t.length?t[0]:0,b=t.length>1?t[t.length-1]:a+1;
 ctx.font='12px system-ui';ctx.fillStyle='#c2d2df';ctx.strokeStyle='#334553';
 for(let i=0;i<=4;i++){const y=p.t+i*h/4;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(p.l+w,y);ctx.stroke();ctx.fillText((hi-(hi-lo)*i/4).toFixed(2),8,y+4);ctx.fillText((a+(b-a)*i/4).toFixed(1),p.l+i*w/4-12,p.t+h+23)}
 ctx.fillText(unit,p.l,p.t-10);ctx.fillText('时间（s）',p.l+w/2-30,c.height-5);
 keys.forEach((key,i)=>{ctx.strokeStyle=colors[i];ctx.lineWidth=2;ctx.beginPath();let open=false;
  s.history[key].forEach((v,k)=>{if(!Number.isFinite(v)){open=false;return}const x=p.l+(t[k]-a)/Math.max(b-a,1e-6)*w,y=p.t+(hi-v)/(hi-lo)*h;open?ctx.lineTo(x,y):ctx.moveTo(x,y);open=true});ctx.stroke()});
 document.getElementById(id+'-values').innerHTML=keys.map((key,i)=>`<span style="color:${colors[i]}">${key.toUpperCase()}: ${s.wrench===null?'—':s.wrench[key].toFixed(3)+' '+unit}</span>`).join('');
}
let busy=false;async function update(){if(busy)return;busy=true;try{
 const response=await fetch('/snapshot',{cache:'no-store'});if(!response.ok)throw Error('HTTP');const s=await response.json();
 document.getElementById('frame').textContent=s.frame_id;
 document.getElementById('status').textContent=(s.state==='valid'?'估计运行中':'末端估计不可用')+' · '+s.explanation+' · 雅可比条件数 '+(s.geometry.condition_number===null?'—':s.geometry.condition_number.toFixed(1));
 plot('moment',s,['mx','my','mz'],'N·m');plot('force',s,['fx','fy','fz'],'N');
 }catch(e){document.getElementById('status').textContent='连接中断：不能使用之前的估计值'}finally{busy=false}}
setInterval(update,100);update();</script></html>'''


def stamp_ns(msg):
    return msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec


class EndpointObserver(Node):
    def __init__(self, config):
        super().__init__('right_endpoint_wrench_observer')
        self.cfg = yaml.safe_load(Path(config).read_text())
        if self.cfg['side'] != 'right':
            raise ValueError('This deployment requires side=right')
        self.names = [f'right_joint_{i}' for i in range(7)]
        self.frame_id = self.cfg['tool_frame']
        inference = yaml.safe_load(Path(self.cfg['inference_config']).read_text())
        if inference['side'] != 'right' or inference['joint_names'] != self.names:
            raise ValueError('Inference side/order mismatch')
        self.envelope = OperatingEnvelope(inference['operating_envelope'], 7)
        self.model = None
        self.next_status = {'state': 'warming', 'reason': 'waiting_inference_status'}
        self.next_status_at = None
        self.last_pair_at = None
        self.last_stamp = None
        self.last_pair_stamp = None
        self.state, self.reasons, self.wrench = 'unavailable', ['waiting_inputs'], None
        self.geometry = {'condition_number': None}
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.outputs = 0
        web = self.cfg['web']
        if web['host'] != '127.0.0.1' or not 1 <= web['refresh_hz'] <= 20 or not 1 <= web['history_points'] <= 10000:
            raise ValueError('Endpoint web must be bounded and localhost-only')
        self.history = {key: deque(maxlen=web['history_points']) for key in ('t', 'fx', 'fy', 'fz', 'mx', 'my', 'mz')}
        self.publisher = self.create_publisher(WrenchStamped, '/next/right/endpoint_wrench', qos_profile_sensor_data)
        self.status_publisher = self.create_publisher(DiagnosticArray, '/next/right/endpoint_wrench/status', qos_profile_sensor_data)
        self.description_sub = self.create_subscription(String, '/robot_description', self._description,
                                                         QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.status_sub = self.create_subscription(DiagnosticArray, '/next/right/status', self._status, qos_profile_sensor_data)
        self.position_sub = Subscriber(self, JointState, '/factr2/right/joint_pos', qos_profile=qos_profile_sensor_data)
        self.torque_sub = Subscriber(self, JointState, '/next/right/external_joint_torque', qos_profile=qos_profile_sensor_data)
        self.sync = TimeSynchronizer([self.position_sub, self.torque_sub], queue_size=10)
        self.sync.registerCallback(self._pair)
        self.timer = self.create_timer(1. / web['refresh_hz'], self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.httpd = ThreadingHTTPServer((web['host'], web['port']), self._handler())
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.get_logger().info(f'Read-only endpoint dashboard: http://{web["host"]}:{web["port"]}; tool frame={self.frame_id}')

    def _description(self, msg):
        digest = hashlib.sha256(msg.data.encode()).hexdigest()
        if digest != self.cfg['robot_description_sha256']:
            self.model = None
            self._unavailable(['robot_description_mismatch'])
            return
        root = ET.fromstring(msg.data)
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        link = self.frame_id
        while link in by_child:
            link = by_child[link].find('parent').get('link')
        if link != self.cfg['root_frame']:
            raise ValueError('Jacobian root frame mismatch')
        self.model = ArmModel(msg.data, 'right', self.frame_id)

    def _status(self, msg):
        age = (self.get_clock().now().nanoseconds - stamp_ns(msg)) * 1e-9
        found = [s for s in msg.status if s.name == 'factr2/next/right']
        if len(found) != 1 or not 0 <= age <= .25:
            self.next_status_at = None
            return
        self.next_status = {v.key: json.loads(v.value) for v in found[0].values}
        self.next_status_at = time.monotonic()
        if self.next_status.get('state') != 'valid':
            reasons = [self.next_status.get('reason', 'inference_not_valid')]
            if self.geometry.get('reason') == 'near_singular':
                reasons.append('near_singular')
            self._unavailable(reasons)

    def _unavailable(self, reasons):
        with self.lock:
            self.state, self.reasons, self.wrench = 'unavailable', reasons, None

    def _pair(self, position, torque):
        ns = stamp_ns(torque)
        now = self.get_clock().now().nanoseconds
        q = np.asarray(position.position, dtype=float)
        tau = np.asarray(torque.position, dtype=float)
        if (position.name != self.names or torque.name != self.names or q.shape != (7,)
                or tau.shape != (7,) or not np.isfinite(q).all() or not np.isfinite(tau).all()
                or ns <= 0 or stamp_ns(position) != ns or not 0 <= (now - ns) * 1e-9 <= .25):
            self._unavailable(['input_names_shape_or_timestamp']); return
        self.last_pair_at, self.last_pair_stamp = time.monotonic(), ns
        if self.last_stamp is not None and ns <= self.last_stamp:
            self.last_stamp = ns
            self._unavailable(['timestamp_not_advancing']); return
        self.last_stamp = ns
        if self.model is None:
            self._unavailable(['waiting_matching_robot_description']); return
        transform, j = self.model.fk(q, True)
        estimate, geometry = estimate_wrench(j, tau, **self.cfg['solver'])
        with self.lock:
            self.geometry = geometry
        reasons = []
        if self.next_status_at is None or time.monotonic() - self.next_status_at > .25:
            reasons.append('inference_status_stale')
        elif self.next_status.get('state') != 'valid':
            reasons.append(self.next_status.get('reason', 'inference_not_valid'))
        outside = self.envelope.outside_joints(q.reshape(1, -1))
        if outside:
            reason = 'outside_collection_range:' + ','.join(self.names[i] for i in outside)
            if reason not in reasons: reasons.append(reason)
        if estimate is None:
            reasons.append(geometry['reason'])
        if reasons:
            self._unavailable(reasons); return
        tool_value = rotate_wrench_at_same_origin(estimate, transform[:3, :3])
        msg = WrenchStamped(); msg.header.stamp = torque.header.stamp; msg.header.frame_id = self.frame_id
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = map(float, tool_value[:3])
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = map(float, tool_value[3:])
        with self.lock:
            self.state, self.reasons = 'valid', []
            self.wrench = dict(zip(('fx', 'fy', 'fz', 'mx', 'my', 'mz'), map(float, tool_value)))
            self.outputs += 1
        self.publisher.publish(msg)

    def _tick(self):
        now = time.monotonic()
        if self.last_pair_at is None or now - self.last_pair_at > .25:
            self._unavailable(['paired_input_stale'])
        elif self.next_status_at is None or now - self.next_status_at > .25:
            self._unavailable(['inference_status_stale'])
        with self.lock:
            self.history['t'].append(now - self.started)
            for key in self.history:
                if key != 't': self.history[key].append(self.wrench[key] if self.wrench is not None else None)
            details = dict(state=self.state, reasons=self.reasons, geometry=self.geometry,
                           frame_id=self.frame_id, outputs=self.outputs, source_stamp_ns=self.last_pair_stamp)
        msg = DiagnosticArray(); msg.header.stamp = self.get_clock().now().to_msg()
        s = DiagnosticStatus(); s.name = 'factr2/endpoint/right'
        s.level = DiagnosticStatus.OK if details['state'] == 'valid' else DiagnosticStatus.WARN
        s.message = details['state'] + ':' + ','.join(details['reasons'])
        s.values = [KeyValue(key=k, value=json.dumps(v)) for k, v in details.items()]
        msg.status = [s]; self.status_publisher.publish(msg)

    def _snapshot(self):
        with self.lock:
            reasons = list(self.reasons)
            explanation = []
            for reason in reasons:
                if reason.startswith('outside_collection_range:'):
                    joints = ', '.join('J' + str(int(n.rsplit('_', 1)[-1]) + 1) for n in reason.split(':', 1)[1].split(','))
                    explanation.append(joints + ' 超出模型采集范围')
                elif reason == 'near_singular': explanation.append('姿态接近奇异点，误差会被放大')
                elif reason == 'inconsistent_with_tool_contact': explanation.append('关节残差不符合单一末端接触假设')
                elif 'stale' in reason: explanation.append('等待新鲜同步数据')
                else: explanation.append(reason)
            return dict(state=self.state, reasons=reasons, explanation='；'.join(explanation) or '等效末端负载估计，尚未做接触精度与方向标定',
                        geometry=dict(self.geometry), frame_id=self.frame_id,
                        moment_reference_point=self.frame_id, sign_convention='NEXT_residual',
                        outputs=self.outputs, wrench=None if self.wrench is None else dict(self.wrench),
                        history={k: list(v) for k, v in self.history.items()})

    def _handler(self):
        observer = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_GET(self):
                if self.path == '/': body, kind = HTML.encode(), 'text/html; charset=utf-8'
                elif self.path == '/snapshot': body, kind = json.dumps(observer._snapshot(), allow_nan=False).encode(), 'application/json'
                else: self.send_error(404); return
                self.send_response(200); self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
                self.end_headers(); self.wfile.write(body)
        return Handler

    def destroy_node(self):
        self.httpd.shutdown(); self.httpd.server_close()
        return super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config/w3/right/endpoint_wrench.yaml')
    args = parser.parse_args()
    rclpy.init(); node = None
    try:
        node = EndpointObserver(args.config)
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node is not None: node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == '__main__': main()
