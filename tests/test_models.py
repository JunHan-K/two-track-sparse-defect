import pytest
import torch

from sds.losses import seg_loss
from sds.models.segformer import SegFormerSDS


@pytest.mark.parametrize("head,fusion,n_aux", [("original", "none", 0), ("stagewise", "none", 4),
                                               ("stagewise", "small", 5)])
def test_heads_shapes_and_inference_graph(head, fusion, n_aux):
    m = SegFormerSDS(pretrained=None, head=head, fusion=fusion).train()
    x = torch.randn(2, 3, 64, 64)
    out = m(x, return_aux=True)
    assert out["logits"].shape == (2, 1, 64, 64) and len(out["aux"]) == n_aux
    target = (torch.rand(2, 1, 64, 64) > 0.9).float()
    loss, logs = seg_loss(out, target, torch.ones_like(target))
    loss.backward()
    assert torch.isfinite(loss) and "loss_main" in logs
    m.eval()
    with torch.no_grad():
        inf = m(x, return_aux=False)
    assert inf["aux"] == {}  # without fusion the stage heads are training-only


def test_stage_heads_do_not_change_the_deployed_output():
    torch.manual_seed(0)
    m = SegFormerSDS(pretrained=None, head="stagewise").eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a = m(x, return_aux=False)["logits"]
        b = m(x, return_aux=True)["logits"]
    assert torch.equal(a, b)
