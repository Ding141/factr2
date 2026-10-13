"""Interruption must leave the latest complete best weight snapshot intact."""
import json
import numpy as np
import pytest
import torch
from factr2_next.training.w3_train import persist_best_snapshot
from factr2_next.data_collection.quality import sha256


def test_atomic_best_snapshot_survives_failed_next_save(tmp_path,monkeypatch):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    cfg={'model':{'type':'lstm'}};state={'weight':torch.tensor([1.,2.])}
    norm={'x_mean':np.zeros(21,dtype=np.float32),'y_std':np.ones(7,dtype=np.float32)}
    p=persist_best_snapshot(tmp_path,state,norm,cfg,{'best_epoch':1,'val_loss':[.02]}, {'source':'real'},manifest)
    pointer=json.loads((tmp_path/'best_snapshot.json').read_text())
    assert pointer['complete'] and pointer['sha256']==sha256(p)
    snapshot=torch.load(p,weights_only=False)
    torch.testing.assert_close(snapshot['model_state_dict']['weight'],state['weight'])
    np.testing.assert_array_equal(snapshot['normalization']['y_std'],norm['y_std'])
    assert snapshot['manifest_sha256']==sha256(manifest)
    def interrupted(*args,**kwargs):raise OSError('interrupted write')
    monkeypatch.setattr(torch,'save',interrupted)
    with pytest.raises(OSError):
        persist_best_snapshot(tmp_path,state,norm,cfg,{'best_epoch':2}, {'source':'real'},manifest)
    assert json.loads((tmp_path/'best_snapshot.json').read_text())==pointer
    assert sha256(p)==pointer['sha256']
