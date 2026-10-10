import tempfile
from pathlib import Path
from combine_w3_session_trials import control_signature

def test_identical_snapshot_content_at_different_paths_matches_and_changes_reject():
    with tempfile.TemporaryDirectory() as tmp:
        a=Path(tmp)/'first.yaml';b=Path(tmp)/'second.yaml'
        a.write_text('kp: [10, 35]\n');b.write_text(a.read_text())
        one={'gains':{'position_params':str(a)}};two={'gains':{'position_params':str(b)}}
        assert control_signature(one)==control_signature(two)
        b.write_text('kp: [10, 70]\n')
        assert control_signature(one)!=control_signature(two)
        assert one['gains']['position_params']==str(a)
