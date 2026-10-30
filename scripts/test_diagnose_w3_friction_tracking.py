"""Check command reconstruction against the actual independent C++ encoder."""
import subprocess
from pathlib import Path

import numpy as np
import pytest

from diagnose_w3_friction_tracking import quantize,smooth_target,constant_runs


@pytest.fixture(scope='module')
def driver_reference(tmp_path_factory):
    root=Path('/home/venom/dual_arm_robot/src')
    out=tmp_path_factory.mktemp('driver_reference');source=out/'main.cpp'
    source.write_text('''
#include "w3_robot_bridge/damiao_protocol.hpp"
#include "ieir_controllers/smooth_friction.hpp"
#include <iostream>
#include <iomanip>
int main() {
  int kind; double x,a,b,c,d,e,f;
  std::cout << std::setprecision(17);
  while(std::cin >> kind >> x >> a >> b >> c >> d >> e >> f) {
    if(kind==12 || kind==16) {
      auto code=kind==12 ? w3_robot_bridge::protocol::float_to_u12(x,a,b)
                         : w3_robot_bridge::protocol::float_to_u16(x,a,b);
      auto value=kind==12 ? w3_robot_bridge::protocol::u12_to_float(code,a,b)
                          : w3_robot_bridge::protocol::u16_to_float(code,a,b);
      std::cout << code << " " << value << "\\n";
    } else {
      std::cout << "0 " << ieir_controllers::smooth_friction(x,a,b,c,d,e,f) << "\\n";
    }
  }
}
''')
    executable=out/'reference'
    subprocess.run(['g++','-std=c++17','-O0','-I'+str(root/'W3_ROBOT/w3_robot_bridge/include'),
        '-I'+str(root/'ieir_controllers/include'),str(source),str(root/'W3_ROBOT/w3_robot_bridge/src/damiao_protocol.cpp'),'-o',str(executable)],check=True,capture_output=True,text=True)
    return executable


def test_float32_reconstruction_matches_real_driver(driver_reference):
    cases=[]
    for bits,limit in [(12,45.),(12,20.),(12,50.),(12,54.),(12,5.),(16,12.5)]:
        values=np.r_[[-2*limit,-limit,-1e-8,0,1e-8,limit,2*limit],np.linspace(-.05,.05,401)]
        for x in values:cases.append((bits,float(x),-limit,limit))
    payload='\n'.join(f'{bits} {x:.17g} {lo} {hi} 0 0 0 0' for bits,x,lo,hi in cases)+'\n'
    result=subprocess.run([str(driver_reference)],input=payload,text=True,capture_output=True,check=True)
    reference=np.array([[float(x) for x in line.split()] for line in result.stdout.splitlines()])
    assert len(reference)==len(cases)
    for case,row in zip(cases,reference):
        bits,x,lo,hi=case;code,value=quantize(x,lo,hi,bits)
        assert code==row[0]
        assert value==row[1]


def test_smooth_friction_reconstruction_matches_controller(driver_reference):
    cases=[(v,.01,.47615128002450663,.5022121183888698,.006677568,.7,.8)
        for v in np.r_[0.,np.deg2rad([.1,-.1,.6,-.6,1.2,-1.2,2.4,-2.4]),10.,-10.]]
    payload='\n'.join('0 '+' '.join(format(x,'.17g') for x in row) for row in cases)+'\n'
    result=subprocess.run([str(driver_reference)],input=payload,text=True,capture_output=True,check=True)
    reference=np.array([float(line.split()[1]) for line in result.stdout.splitlines()])
    actual=[smooth_target(v,pos,neg,b,gain,tr,limit) for v,tr,pos,neg,b,gain,limit in cases]
    np.testing.assert_allclose(actual,reference,rtol=1e-14,atol=1e-15)


def test_flat_runs_do_not_join_different_encoder_codes():
    starts,ends=constant_runs(np.array([2,2,2,3,3,2]))
    np.testing.assert_array_equal(starts,[0,3,5]);np.testing.assert_array_equal(ends,[2,4,5])
