"""Standard DiagnosticArray formatting used by both isolated environments."""
import json
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue


def diagnostic(stamp, name, values, ok, reason):
    msg = DiagnosticArray()
    msg.header.stamp = stamp
    status = DiagnosticStatus()
    status.name = name
    status.hardware_id = 'w3_next_v1'
    status.level = DiagnosticStatus.OK if ok else DiagnosticStatus.ERROR
    status.message = reason
    for key, value in values.items():
        status.values.append(KeyValue(key=str(key), value=json.dumps(value, separators=(',', ':'))))
    msg.status = [status]
    return msg
