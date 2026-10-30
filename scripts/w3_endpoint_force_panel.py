"""Passive endpoint-force add-on for coherent old/new joint-torque dashboards."""
from collections import deque
import hashlib
import threading
import time
import numpy as np

from w3_coverage_plan import ArmModel
from factr2_next.inference.endpoint_wrench import estimate_wrench,rotate_wrench_at_same_origin


class EndpointForcePanel:
    def __init__(self,cfg,model_hashes):
        self.cfg=cfg;self.model_hashes=dict(model_hashes);self.model=None
        self.description_reason='waiting_matching_robot_description'
        self.packet=None;self.received=None;self.last_tick=None;self.source_age=None
        self.computed_stamp=None
        self.started=time.monotonic();self.lock=threading.Lock()
        self.latest={k:self.unavailable('waiting_inputs') for k in ('old','new')}
        self.history={k:{name:deque(maxlen=cfg['history_points']) for name in ('t','fx','fy','fz','norm')}
                      for k in ('old','new')}
        estimate_wrench(np.c_[np.eye(6),np.zeros(6)],np.zeros(7),**cfg['solver'])
        if (cfg['solver'].get('mode','full_wrench')!='full_wrench'
            or not 1<=cfg['history_points']<=10000):raise ValueError('endpoint_panel_config')

    @staticmethod
    def unavailable(reason,geometry=None):
        return dict(state='unavailable',reason=reason,force=None,geometry=geometry or {})

    def description(self,text):
        if hashlib.sha256(text.encode()).hexdigest()!=self.cfg['robot_description_sha256']:
            self.model=None;self.description_reason='robot_description_mismatch';return
        model=ArmModel(text,'right',self.cfg['tool_frame'])
        if model.root_frame!=self.cfg['root_frame']:raise ValueError('endpoint_root_frame_mismatch')
        self.model=model;self.description_reason=None

    def ingest(self,packet,received_mono=None):
        if (packet['schema']!='w3_joint_lstm_comparison_v1' or packet['model_sha256']!=self.model_hashes
            or packet['units']!='Nm' or packet['joint_names']!=[f'right_joint_{j}' for j in range(7)]
            or packet['same_input_history'] is not True):raise ValueError('endpoint_pair_identity')
        self.packet=packet;self.received=time.monotonic() if received_mono is None else received_mono

    def tick(self,ros_now_ns,now=None):
        now=time.monotonic() if now is None else now;p=self.packet
        invalid=None;self.source_age=None
        if p is None or self.received is None:invalid='waiting_inputs'
        elif not 0<=now-self.received<=.25:invalid='paired_input_stale'
        else:
            self.source_age=(ros_now_ns-int(p['stamp_ns']))*1e-9
            if not 0<=self.source_age<=.25:invalid='source_timestamp_stale'
        if not invalid and self.model is None:invalid=self.description_reason
        results={}
        if invalid:results={k:self.unavailable(invalid) for k in ('old','new')}
        else:
            values=p['recording_frame'];q=np.deg2rad(np.asarray(values['q_deg'],float))
            transform,jacobian=self.model.fk(q,True)
            for kind in ('old','new'):
                tau=np.asarray(values[kind+'_filtered_nm'],float)
                wrench,geometry=estimate_wrench(jacobian,tau,**self.cfg['solver'])
                reason=p.get(kind+'_reason',p.get('reason','inference_not_valid'))
                if p[kind+'_state']!='valid':results[kind]=self.unavailable(reason,geometry)
                elif wrench is None:results[kind]=self.unavailable(geometry['reason'],geometry)
                else:
                    tool=rotate_wrench_at_same_origin(wrench,transform[:3,:3])[:3]
                    results[kind]=dict(state='valid',reason='ok',geometry=geometry,
                        force=dict(fx=float(tool[0]),fy=float(tool[1]),fz=float(tool[2]),norm=float(np.linalg.norm(tool))))
        with self.lock:
            self.latest=results;self.last_tick=now;self.computed_stamp=int(p['stamp_ns']) if p else None
            for kind in ('old','new'):
                self.history[kind]['t'].append(now-self.started)
                for axis in ('fx','fy','fz','norm'):
                    self.history[kind][axis].append(results[kind]['force'][axis] if results[kind]['force'] else None)

    def snapshot(self,kind,now=None):
        now=time.monotonic() if now is None else now
        with self.lock:
            result=dict(self.latest[kind])
            if self.last_tick is None or not 0<=now-self.last_tick<=.25:
                result=self.unavailable('endpoint_update_stale')
            result.update(frame_id=self.cfg['tool_frame'],root_frame=self.cfg['root_frame'],units='N',
                source_stamp_ns=self.computed_stamp,
                source_age_seconds=self.source_age,estimation_mode='full_wrench',
                input_signal='filtered_external_joint_torque',display_decimals=2,
                history={k:list(v) for k,v in self.history[kind].items()})
            return result


PANEL_HTML='''<section style="margin-top:24px" aria-label="末端受力曲线">
<div class="top"><h1>末端等效力 · N</h1><span class="pill" id="endpoint-force-status">等待数据</span></div>
<p class="description">Fx / Fy / Fz 的坐标轴与作用点为 <b id="endpoint-force-frame">right_attachment_point</b>。
由关节外力矩换算，假设外部载荷作用于末端；不可用区间留空，不代表零外力。</p>
<style>#endpoint-force-plot{height:320px}</style>
<div class="plot"><canvas id="endpoint-force-plot" width="1240" height="320"></canvas></div>
<div class="items" id="endpoint-force-controls" style="margin-top:10px"></div>
</section>'''


PANEL_SCRIPT='''
const endpointSeries=[['fx','Fx','#ff6b6b',true],['fy','Fy','#69db7c',true],['fz','Fz','#4dabf7',true],['norm','|F|','#ffd43b',false]];
let endpointData=null,endpointParentValid=false;
const endpointControls=document.getElementById('endpoint-force-controls');
endpointControls.innerHTML=endpointSeries.map(([key,label,color,visible])=>`<label class="item"><input type="checkbox" data-axis="${key}" ${visible?'checked':''}><span class="sw" style="background:${color}"></span>${label}<span class="value" id="endpoint-value-${key}">—</span></label>`).join('');
endpointControls.querySelectorAll('input').forEach(box=>{box.onchange=()=>drawEndpointForce()});
function endpointReason(reason){
 if((reason||'').startsWith('outside_collection_range:'))return '姿态超出模型训练范围';
 return ({ok:'估算运行中',near_singular:'接近奇异位形，末端力不可辨识',inconsistent_with_tool_contact:'关节残差不符合末端载荷假设',
 waiting_inputs:'等待机器人数据',waiting_matching_robot_description:'等待匹配的机器人模型',robot_description_mismatch:'机器人模型不匹配',
 paired_input_stale:'同步输入已过期',source_timestamp_stale:'源数据时间无效',endpoint_update_stale:'末端数据已过期'})[reason]||reason||'等待末端数据';
}
function updateEndpointForce(packet){
 endpointData=packet.endpoint_force||null;endpointParentValid=packet.state==='valid';
 const d=endpointData,valid=d&&d.state==='valid'&&endpointParentValid;
 const condition=d&&d.geometry?d.geometry.condition_number:null;
 document.getElementById('endpoint-force-status').textContent=(!endpointParentValid?endpointReason(packet.state==='stale'?'paired_input_stale':packet.reason):endpointReason(d?d.reason:'waiting_inputs'))+(Number.isFinite(condition)?' · 条件数 '+condition.toFixed(1):'');
 if(d)document.getElementById('endpoint-force-frame').textContent=d.frame_id;
 for(const [key] of endpointSeries){const v=valid&&d.force?d.force[key]:null;document.getElementById('endpoint-value-'+key).textContent=Number.isFinite(v)?v.toFixed(2)+' N':'—'}
 drawEndpointForce();
}
function drawEndpointForce(){
 const c=document.getElementById('endpoint-force-plot'),ctx=c.getContext('2d'),h=endpointData?endpointData.history:{t:[]};
 ctx.clearRect(0,0,c.width,c.height);const pad={l:70,r:18,t:28,b:50},W=c.width-pad.l-pad.r,H=c.height-pad.t-pad.b;
 const keys=Array.from(endpointControls.querySelectorAll('input:checked')).map(b=>b.dataset.axis),t=h.t||[];
 const ys=keys.flatMap(k=>(h[k]||[]).filter(Number.isFinite));let lo=ys.length?Math.min(0,...ys):-1,hi=ys.length?Math.max(0,...ys):1;
 if(hi-lo<1e-6){lo-=1;hi+=1}const margin=(hi-lo)*.08;lo-=margin;hi+=margin;
 const x0=t.length?t[0]:0,x1=t.length>1?t[t.length-1]:x0+10;
 ctx.font='12px system-ui';ctx.fillStyle='#9aa4ae';ctx.strokeStyle='#293039';ctx.lineWidth=1;
 ctx.fillText('末端等效力（N）',pad.l,16);ctx.fillText('时间（s）',pad.l+W/2-25,c.height-6);
 for(let i=0;i<=5;i++){let y=pad.t+H*i/5,x=pad.l+W*i/5;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+W,y);ctx.stroke();ctx.fillText((hi-(hi-lo)*i/5).toFixed(2),8,y+4);ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,pad.t+H);ctx.stroke();ctx.fillText((x0+(x1-x0)*i/5).toFixed(1),x-8,pad.t+H+24)}
 for(const [key,label,color] of endpointSeries){if(!keys.includes(key))continue;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();let open=false;
  (h[key]||[]).forEach((v,i)=>{if(!Number.isFinite(v)){open=false;return}const x=pad.l+(t[i]-x0)/Math.max(1e-6,x1-x0)*W,y=pad.t+(hi-v)/(hi-lo)*H;open?ctx.lineTo(x,y):ctx.moveTo(x,y);open=true});ctx.stroke()}
}
drawEndpointForce();
'''
