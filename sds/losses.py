import torch
import torch.nn.functional as F


def bce_dice(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1.0,
             weight: torch.Tensor | None = None) -> torch.Tensor:
    """L_seg = BCE + soft Dice, both restricted to non-padded pixels.

    Dice is computed over the whole batch (per-image Dice is ill-defined for the
    many defect-free images). `weight` (optional, >= 1) re-weights the BCE term per pixel;
    the normaliser stays the number of valid pixels, so extra weight adds loss.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if weight is not None:
        bce = bce * weight
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    t = target * valid
    dice = 1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return bce + dice


def seg_loss(out: dict, target: torch.Tensor, valid: torch.Tensor, lambda_aux: float = 1.0,
             pos_weight: torch.Tensor | None = None, spec: tuple | None = None) -> tuple[torch.Tensor, dict]:
    """L = L_main + lambda_aux * mean_k L_k over the auxiliary (stage) heads.

    pos_weight: size-aware per-pixel BCE weight of the main head (1 on background).
    spec = (head names, small-only target, its valid map): those stage heads learn small defects only
    (larger components and a margin around them are ignored); the other heads learn all defects.
    """
    main = bce_dice(out["logits"].float(), target, valid, weight=pos_weight)
    logs = {"loss_main": main.detach()}
    loss = main
    if out["aux"]:
        aux = [bce_dice(v.float(), spec[1], spec[2]) if (spec is not None and k in spec[0])
               else bce_dice(v.float(), target, valid) for k, v in out["aux"].items()]
        for d, l in zip(out["aux"].keys(), aux):
            logs[f"loss_{d}"] = l.detach()
        aux = torch.stack(aux).mean()
        loss = main + lambda_aux * aux
        logs["loss_aux"] = aux.detach()
    logs["loss"] = loss.detach()
    return loss, logs
