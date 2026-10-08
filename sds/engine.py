import torch
import torch.nn.functional as F

from .metrics import PixelEvaluator


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict_batch(model, batch, heads=("main",), amp=True, resolution="original", dtype=torch.float64):
    """Run the model and return per-image probability maps.

    heads: iterable containing "main" and/or training-only stage heads (s1..s4).
    resolution: "original" -> padding removed + upsampled to the original image size
                "resized"  -> padding removed, network input resolution
    dtype: of the probabilities. Evaluation keeps float64: a float32 sigmoid is exactly 1.0 above logit ~16.6, so
           confident pixels would tie in the AP ranking (tools/measure_modes.py times the float32 deployment path).
    Returns: list (per image) of {head: prob tensor (h, w)}.
    """
    need_aux = any(h != "main" for h in heads)
    with torch.autocast("cuda", enabled=amp and batch["image"].is_cuda):
        out = model(batch["image"], return_aux=need_aux)
    dims = getattr(model, "dims", ())
    maps = {}
    for h in heads:
        if h == "main" or (dims and str(h) == str(dims[-1])):
            maps[h] = out["logits"]
        else:
            maps[h] = out["aux"][h if h in out["aux"] else int(h)]
    results = []
    for i in range(batch["image"].shape[0]):
        nh, nw = batch["resized_hw"][i].tolist()
        oh, ow = batch["orig_hw"][i].tolist()
        per = {}
        for h, logit in maps.items():
            p = torch.sigmoid(logit[i : i + 1, :, :nh, :nw].to(dtype))
            if resolution == "original" and (nh, nw) != (oh, ow):
                p = F.interpolate(p, size=(oh, ow), mode="bilinear", align_corners=False)
            per[h] = p[0, 0]
        results.append(per)
    return results


@torch.no_grad()
def evaluate_loader(model, loader, dataset, device, heads=("main",), amp=True, keep_per_image=True, on_image=None,
                    **evaluator_kw):
    """Evaluate heads on a loader at original resolution. Returns {head: PixelEvaluator}.

    evaluator_kw: size_edges / comp_edges / category_fn forwarded to PixelEvaluator.
    """
    from .data.dataset import load_mask

    model.eval()
    evs = {h: PixelEvaluator(keep_per_image, **evaluator_kw) for h in heads}
    for batch in loader:
        batch = to_device(batch, device)
        preds = predict_batch(model, batch, heads, amp)
        for i, per in enumerate(preds):
            row = dataset.rows[int(batch["index"][i])]
            gt = torch.from_numpy(load_mask(row["mask"], (row["height"], row["width"]))).to(device)
            for h, p in per.items():
                evs[h].update(p, gt, row["id"], row["label"])
            if on_image is not None:
                on_image(row, per, gt)
    return evs
