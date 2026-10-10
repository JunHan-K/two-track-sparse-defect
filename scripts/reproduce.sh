#!/usr/bin/env bash
# Reproduce the paper: every model, 3 seeds, validation-selected thresholds, test split.
#   bash scripts/reproduce.sh vision ours        # our method only (train, mine, refine, evaluate)
#   bash scripts/reproduce.sh vision baselines   # Table I competitors
#   bash scripts/reproduce.sh vision ablation    # Table III (test split)
#   bash scripts/reproduce.sh mvtec_ad ours|baselines|ablation
# Each step is one command; run them on as many GPUs as you have (they are independent per seed).
set -euo pipefail
cd "$(dirname "$0")/.."
DS=${1:?vision|mvtec_ad}; WHAT=${2:?ours|baselines|ablation}
SEEDS=${SEEDS:-"42 1 2"}
P=$([ "$DS" = vision ] && echo vision || echo mvtec)   # experiment-id prefix
sfx() { [ "$1" = 42 ] && echo "" || echo "_s$1"; }
run() { echo "+ $*"; "$@"; }

train_eval() {  # config, then evaluation of the whole-image model on val+test (strict and 3-px tolerant)
  local cfg=$1 id; id=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['experiment_id'])" "$cfg")
  for s in $SEEDS; do
    run python tools/train.py --config "$cfg" --opts seed=$s
    run python tools/measure_efficiency.py --exp outputs/$id$(sfx $s)
    for tol in 0 3; do run python tools/evaluate.py --exp outputs/$id$(sfx $s) --test --tol $tol; done
  done
}

ours() {  # step 1 -> mine native false positives -> refinement with Sparse Defect Replay -> both operating points
  local st=${P}_ours_stage1
  for s in $SEEDS; do
    local x=$(sfx $s)
    run python tools/train.py --config configs/$DS/ours_stage1.yaml --opts seed=$s
    run python tools/mine_native_fp.py --exp outputs/$st$x --out data/replay/$DS/$st$x.json
    run python tools/train.py --config configs/$DS/ours.yaml --opts seed=$s \
        refine.init_checkpoint=outputs/$st$x/best.pt refine.replay_file=data/replay/$DS/$st$x.json
    run python tools/measure_efficiency.py --exp outputs/${P}_ours$x
    run python tools/crop_flops.py --exp outputs/${P}_ours$x
    for mode in precision recall; do for tol in 0 3; do
      run python tools/evaluate_zoom.py --exp outputs/${P}_ours$x --mode $mode --test --tol $tol
    done; done
  done
}

case $WHAT in
  ours) ours ;;
  baselines)
    for c in configs/$DS/baselines/*.yaml; do train_eval "$c"; done
    if [ "$DS" = vision ]; then
      for s in $SEEDS; do  # SegFormer-B0 on native-resolution tiles; MagNet with both losses
        run python tools/crop_flops.py --exp outputs/vision_segformer_b0$(sfx $s)
        for tol in 0 3; do run python tools/evaluate_zoom.py --exp outputs/vision_segformer_b0$(sfx $s) --mode recall --test --tol $tol; done
        for loss in unified authors; do
          run python tools/external/magnet_seg.py --seed $s --loss $loss
          for tol in 0 3; do run python tools/external/magnet_seg.py --seed $s --loss $loss --eval-only --test --tol $tol; done
        done
      done
    else
      for s in $SEEDS; do
        run python tools/external/supersimplenet_seg.py --dataset mvtec_ds --seed $s
        for tol in 0 3; do run python tools/external/supersimplenet_seg.py --dataset mvtec_ds --seed $s --eval-only --test --tol $tol; done
      done
    fi ;;
  ablation)  # Table III: precision mode for every row, validation-selected thresholds, test split
    # (1) training components (everything except refine_*); (2) refinements of the unfused step-1 model
    #     (ablation/stage_heads_size_aware.yaml) with replay mined from it; run "ours" and "baselines" first
    abl() {  # config, init checkpoint id, replay id
      local c=$1 id; id=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['experiment_id'])" "$c")
      for s in $SEEDS; do
        local x=$(sfx $s) opts="seed=$s"
        [ -n "${2:-}" ] && opts="$opts refine.init_checkpoint=outputs/$2$x/best.pt refine.replay_file=data/replay/$DS/$3$x.json"
        run python tools/train.py --config "$c" --opts $opts
        for tol in 0 3; do run python tools/evaluate_zoom.py --exp outputs/$id$x --mode precision --test --tol $tol; done
      done
    }
    for c in configs/$DS/ablation/*.yaml; do case $(basename "$c") in refine_*) ;; *) abl "$c" ;; esac; done
    base=${P}_abl_stage_heads_size_aware  # VISION; MVTec AD has no unfused step 1, its controls refine ours_stage1
    [ -f configs/$DS/ablation/stage_heads_size_aware.yaml ] || base=${P}_ours_stage1
    if ls configs/$DS/ablation/refine_*.yaml >/dev/null 2>&1; then
      for s in $SEEDS; do
        f=data/replay/$DS/$base$(sfx $s).json
        [ -f "$f" ] || run python tools/mine_native_fp.py --exp outputs/$base$(sfx $s) --out "$f"
      done
      for c in configs/$DS/ablation/refine_*.yaml; do abl "$c" $base $base; done
    fi
    for s in $SEEDS; do  # rows that reuse runs of the other groups
      for id in ${P}_segformer_b0 ${P}_ours_stage1 ${P}_ours; do
        for tol in 0 3; do run python tools/evaluate_zoom.py --exp outputs/$id$(sfx $s) --mode precision --test --tol $tol; done
      done
    done ;;
  *) echo "unknown: $WHAT"; exit 1 ;;
esac
