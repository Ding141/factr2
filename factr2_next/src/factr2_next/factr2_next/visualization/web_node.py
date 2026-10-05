from factr2_next.w3_config import validate_w3_config
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32
from diagnostic_msgs.msg import DiagnosticArray


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>FACTR2 NEXT</title>
<style>
:root{color-scheme:dark;--bg:#101214;--panel:#181b1f;--line:#2b3036;--text:#e8edf2;--muted:#98a2ad}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}
main{max-width:1280px;margin:22px auto;padding:0 18px}.top{display:flex;align-items:end;justify-content:space-between;gap:18px;margin-bottom:14px}
h1{font-size:20px;margin:0}.pill{color:var(--muted);font-size:13px}.plot{background:#14171a;border:1px solid var(--line);border-radius:8px;overflow:hidden}
canvas{display:block;width:100%;height:580px}.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:12px}
.group{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px}.group h2{font-size:12px;font-weight:650;color:var(--muted);margin:0 0 8px;text-transform:uppercase}
.items{display:flex;flex-wrap:wrap;gap:7px}.item{display:flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:6px;padding:5px 7px;cursor:pointer;user-select:none}
.item.off{opacity:.42}.sw{width:10px;height:10px;border-radius:2px;flex:0 0 auto}input{accent-color:#4ea1ff;margin:0}.value{color:var(--muted);font-variant-numeric:tabular-nums}.status{margin-top:8px;color:var(--muted);font-size:12px}
</style></head>
<body><main>
<div class="top"><div><h1>FACTR2 NEXT</h1><div class="pill" id="topic-root">waiting</div></div><div class="pill" id="status">connecting</div></div>
<div class="plot"><canvas id="plot" width="1240" height="580"></canvas></div>
<div class="controls" id="controls"></div>
</main>
<script>
const canvas=document.getElementById("plot"), ctx=canvas.getContext("2d");
const controls=document.getElementById("controls"), statusEl=document.getElementById("status");
let data={t:[]}, controlsReady=false;
const groups=__GROUPS__;
const series=groups.flatMap(g=>g[1]).map(s=>({key:s[0],label:s[1],color:s[2],visible:loadVisible(s[0],s[3])}));
function loadVisible(key, fallback){const v=localStorage.getItem("next."+key); return v===null?fallback:v==="1"}
function setVisible(key, value){localStorage.setItem("next."+key,value?"1":"0")}
function finite(xs){return (xs||[]).filter(Number.isFinite)}
function setupControls(){
  controls.innerHTML=groups.map(([name,items])=>`<section class=group><h2>${name}</h2><div class=items>${
    items.map(([key,label,color])=>{
      const s=series.find(x=>x.key===key), checked=s.visible?"checked":"";
      return `<label class="item ${s.visible?"":"off"}" data-key="${key}"><input type=checkbox ${checked}><span class=sw style="background:${color}"></span><span>${label}</span><span class=value id="v-${key}">--</span></label>`;
    }).join("")
  }</div></section>`).join("");
  controls.querySelectorAll(".item").forEach(item=>{
    const key=item.dataset.key, box=item.querySelector("input"), s=series.find(x=>x.key===key);
    box.onchange=()=>{s.visible=box.checked; item.classList.toggle("off",!s.visible); setVisible(key,s.visible); draw();};
  });
  controlsReady=true;
}
function draw(){
  if(!controlsReady) setupControls();
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const pad={l:62,r:18,t:22,b:42}, W=canvas.width-pad.l-pad.r, H=canvas.height-pad.t-pad.b;
  const t=data.t||[], xs=finite(t);
  const ys=series.flatMap(s=>s.visible?finite(data[s.key]):[]);
  let xmin=xs.length?xs[0]:0, xmax=xs.length?xs[xs.length-1]:10, ymin=Math.min(...ys), ymax=Math.max(...ys);
  if(xs.length<2){xmax=10}
  if(!Number.isFinite(ymin)||!Number.isFinite(ymax)||ymin===ymax){ymin=-1;ymax=1}
  const m=(ymax-ymin)*0.08; ymin-=m; ymax+=m; grid(pad,W,H,xmin,xmax,ymin,ymax);
  for(const s of series){ if(!s.visible) continue; drawLine(s,pad,W,H,xmin,xmax,ymin,ymax); }
  for(const s of series){ const el=document.getElementById("v-"+s.key), vals=finite(data[s.key]); if(el) el.textContent=vals.length?vals[vals.length-1].toFixed(3):"--"; }
}
function drawLine(s,pad,W,H,xmin,xmax,ymin,ymax){
  const t=data.t||[], vals=data[s.key]||[]; ctx.strokeStyle=s.color; ctx.lineWidth=2; ctx.beginPath(); let open=false;
  for(let i=0;i<t.length;i++){const y=vals[i]; if(!Number.isFinite(y)){open=false;continue}
    const px=pad.l+(t[i]-xmin)/Math.max(1e-6,xmax-xmin)*W, py=pad.t+(1-(y-ymin)/(ymax-ymin))*H;
    open?ctx.lineTo(px,py):ctx.moveTo(px,py); open=true;
  } ctx.stroke();
}
function grid(pad,W,H,x0,x1,y0,y1){
  ctx.strokeStyle="#293039"; ctx.fillStyle="#9aa4ae"; ctx.lineWidth=1; ctx.font="12px system-ui";
  for(let i=0;i<=5;i++){const x=pad.l+i*W/5, y=pad.t+i*H/5;
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+W,y); ctx.stroke();
    ctx.fillText((y1-(y1-y0)*i/5).toFixed(2),8,y+4);
    ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,pad.t+H); ctx.stroke();
    ctx.fillText((x0+(x1-x0)*i/5).toFixed(1),x-8,pad.t+H+24);
  }
}
const events=new EventSource("/events");
events.onerror=()=>{statusEl.textContent="disconnected"};
events.onmessage=(ev)=>{
  data=JSON.parse(ev.data); document.getElementById("topic-root").textContent=`topics: ${data.next_topic_root}  samples: ${(data.t||[]).length}`;
  const age=data.last_valid_age_seconds===null?"--":data.last_valid_age_seconds.toFixed(2)+" s";
  statusEl.textContent=`${data.state} · last valid ${age} · ${data.reason||""}`; draw();
};
setupControls(); draw();
</script></body></html>
"""


def plot_keys(count, feedback=False):
    summary=['t','ext_norm','ext_raw_norm','free_norm','mse','normalized_contact_magnitude','contact']
    prefixes=['ext','raw','free']
    if feedback:
        summary+=['fb_norm','fb_gate'];prefixes+=['fb']
    return tuple(summary+[f'{p}_j{i+1}' for p in prefixes for i in range(count)])


def page(names, feedback=False):
    groups=[['Summary',[
        ['ext_norm','Filtered torque norm (Nm)','#ff5a63',True],
        ['ext_raw_norm','Raw torque norm (Nm)','#f7a1a8',False],
        ['free_norm','Free torque norm (Nm)','#58a6ff',False],
        ['mse','MSE (Nm²)','#ffb454',False],
        ['normalized_contact_magnitude','Contact magnitude','#6ee787',True],
        ['contact','Contact state','#d2a8ff',True]]]]
    colors=['#ff6b6b','#ffa94d','#ffd43b','#69db7c','#4dabf7','#b197fc','#f783ac']
    for label,prefix in [('Filtered torque (Nm)','ext'),('Raw torque (Nm)','raw'),('Free prediction (Nm)','free')]+([('Feedback (Nm)','fb')] if feedback else []):
        groups.append([label,[[f'{prefix}_j{i+1}',name,colors[i%len(colors)],False] for i,name in enumerate(names)]])
    return HTML.replace('__GROUPS__',json.dumps(groups))


TORQUE_KEYS = {
    "ext": ("ext_norm", "ext"),
    "raw": ("ext_raw_norm", "raw"),
    "fb": ("fb_norm", "fb"),
    "free": ("free_norm", "free"),
}


class WebNode(Node):
    def __init__(self, **kwargs):
        super().__init__("factr2_next_visualize", **kwargs)
        default = Path(get_package_share_directory("factr2_next")) / "config" / "visualize.yaml"
        self.cfg = self._load_config(self.declare_parameter("config_file", str(default)).value)
        validate_w3_config(self.cfg, "visualize")
        self.next_topic_root = str(self.cfg.get("next_topic_root", "/next"))
        self.feedback_topic_root = str(self.cfg.get("feedback_topic_root", "/factr2_feedback"))
        self.max_points = int(self.cfg.get("plot", {}).get("max_points", 500))
        self.w3 = 'contract_version' in self.cfg
        self.joint_count = len(self.cfg.get('joint_names',[])) or int(self.cfg.get('joint_count',6))
        self.names = self.cfg.get('joint_names',[f'joint_{i}' for i in range(self.joint_count)])
        if not 1<=self.joint_count<=64 or not 1<=self.max_points<=10000:raise ValueError('web_joint_count_max_points')
        self.feedback = any(k.startswith('feedback_') for k in self.cfg['outputs'])
        self.keys = plot_keys(self.joint_count,self.feedback)
        self.html = page(self.names,self.feedback)
        self.runtime_status={'state':'warming','reason':'waiting_for_status'}
        self.status_received = None
        self.last_valid_mono = None
        self.data = {key: deque(maxlen=self.max_points) for key in self.keys}
        self.latest = {key: np.nan for key in self.keys if key != "t"}
        self.t0, self.seq, self.running = time.monotonic(), 0, True
        self.cond = threading.Condition()

        self._subscribe()
        web = self.cfg.get("web", {})
        host = str(web.get("host", "127.0.0.1"))
        port = int(web.get("port", 8080))
        self.refresh_hz=float(web.get("refresh_hz",10))
        if not 1<=self.refresh_hz<=20:raise ValueError("web_refresh_hz_range")
        try:
            self.httpd = ThreadingHTTPServer((host, port), self._handler())
        except OSError as exc:
            super().destroy_node()
            raise ValueError(f'web_bind_failed {host}:{port}: {exc}') from exc
        self.httpd.daemon_threads=True
        self.server_thread=threading.Thread(target=self.httpd.serve_forever,daemon=True)
        self.server_thread.start()
        self.get_logger().info(f"NEXT plot at http://{host}:{port}")

    def _subscribe(self):
        outputs = self._output_topics(self.cfg["outputs"])
        outputs.setdefault(
            "external_joint_torque_raw",
            outputs["external_joint_torque"] + "/raw",
        )
        self.subs = [
            self._sub(
                JointState,
                outputs["external_joint_torque"],
                lambda m: self._torque_message("ext", m, True),
                qos_profile_sensor_data,
            ),
            self._sub(
                JointState,
                outputs["external_joint_torque_raw"],
                lambda m: self._torque_message("raw", m),
                qos_profile_sensor_data,
            ),
            self._sub(
                JointState,
                outputs["free_joint_torque_pred"],
                lambda m: self._torque_message("free", m),
                qos_profile_sensor_data,
            ),
            self._sub(
                Float32,
                outputs["mse"],
                lambda m: self._set_scalar("mse", m.data),
                10,
            ),
            self._sub(
                Float32,
                outputs["normalized_contact_magnitude"],
                lambda m: self._set_scalar("normalized_contact_magnitude", m.data),
                10,
            ),
        ]
        self.subs.append(self._sub(DiagnosticArray,self.next_topic_root+'/status',self._status,qos_profile_sensor_data))
        if "contact_state" in outputs:
            self.subs.append(
                self._sub(
                    Bool,
                    outputs["contact_state"],
                    lambda m: self._set_scalar("contact", 1.0 if m.data else 0.0),
                    10,
                )
            )
        if "feedback_torque" in outputs:
            self.subs.append(
                self._sub(
                    JointState,
                    outputs["feedback_torque"],
                    lambda m: self._torque_message("fb", m),
                    qos_profile_sensor_data,
                )
            )
        if "feedback_gate" in outputs:
            self.subs.append(
                self._sub(
                    Float32,
                    outputs["feedback_gate"],
                    lambda m: self._set_scalar("fb_gate", m.data),
                    10,
                )
            )

    def _status(self,msg):
        for s in msg.status:
            if s.name=='factr2/next/'+str(self.cfg.get('side','generic')):
                self.runtime_status={kv.key:json.loads(kv.value) for kv in s.values}
                self.status_received=time.monotonic()
                with self.cond:self.seq+=1;self.cond.notify_all()

    def _torque_message(self,prefix,msg,sample=False):
        if self.w3 and (msg.name!=self.names or len(msg.position)!=self.joint_count or not np.isfinite(msg.position).all()):return
        self._set_torque(prefix,msg.position,sample)

    def _set_norm(self, key, values, sample=False):
        self.latest[key] = float(np.linalg.norm(np.asarray(values, dtype=float)))
        if sample:
            self._append()

    def _set_torque(self, prefix, values, sample=False):
        values = np.asarray(values, dtype=float)
        norm_key, joint_prefix = TORQUE_KEYS[prefix]
        self.latest[norm_key] = float(np.linalg.norm(values))
        for i in range(self.joint_count):
            self.latest[f"{joint_prefix}_j{i + 1}"] = (
                float(values[i]) if i < len(values) else np.nan
            )
        if sample:
            self._append()

    def _set_scalar(self, key, value):
        self.latest[key] = float(value)

    def _append(self):
        self.last_valid_mono=time.monotonic()
        with self.cond:
            self.data["t"].append(time.monotonic() - self.t0)
            for key in self.keys:
                if key == "t":
                    continue
                self.data[key].append(self.latest[key])
            self.seq += 1
            self.cond.notify_all()

    def _snapshot(self):
        with self.cond:
            out = {
                k: [x if np.isfinite(x) else None for x in v]
                for k, v in self.data.items()
            }
            out['next_topic_root']=self.next_topic_root
            out['joint_names']=self.names;out['side']=self.cfg.get('side','generic')
            out['units']='torque Nm; MSE Nm²; contact magnitude configured norm/scale'
            age=time.monotonic()-self.last_valid_mono if self.last_valid_mono is not None else None
            status_age=time.monotonic()-self.status_received if self.status_received is not None else None
            out['last_valid_age_seconds']=age;out['status_age_seconds']=status_age
            out['state']=self.runtime_status.get('state','warming')
            out['reason']=self.runtime_status.get('reason','')
            if (age is not None and age>.25) or (self.w3 and (status_age is None or status_age>.25)):
                out['state']='stale';out['reason']='torque_or_status_timeout'
            out['diagnostics']=self.runtime_status
            return json.dumps(out,allow_nan=False).encode()

    def _handler(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path == "/events":
                    self._events()
                    return
                if self.path=='/snapshot':
                    body=node._snapshot();content_type='application/json'
                else:
                    body=node.html.encode();content_type='text/html'
                self.send_response(200)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _events(self):
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.end_headers()
                last = -1
                while node.running and rclpy.ok():
                    with node.cond:
                        node.cond.wait_for(lambda: node.seq != last or not node.running, .25)
                        last = node.seq
                    try:
                        self.wfile.write(b"data: " + node._snapshot() + b"\n\n")
                        self.wfile.flush()
                        # Coalesce 50 Hz torque/status updates into bounded UI refresh.
                        time.sleep(1.0/node.refresh_hz)
                    except (BrokenPipeError, ConnectionResetError):
                        break

        return Handler

    def destroy_node(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.server_thread.join(timeout=1)
        return super().destroy_node()

    def _topic(self, topic):
        return str(topic).format(
            next_topic_root=self.next_topic_root,
            feedback_topic_root=self.feedback_topic_root,
        )

    def _output_topics(self, outputs):
        outputs = dict(outputs)
        if "normalized_contact_magnitude" not in outputs and "score" in outputs:
            outputs["normalized_contact_magnitude"] = outputs["score"]
        return {key: self._topic(topic) for key, topic in outputs.items()}

    def _sub(self, msg_type, topic, callback, qos):
        return self.create_subscription(msg_type, topic, callback, qos)

    def _load_config(self, path):
        with open(Path(path).expanduser(), "r") as f:
            return yaml.safe_load(f) or {}


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = WebNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
