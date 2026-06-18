#!/usr/bin/env bash
# End-to-end edge-ControlNet pipeline. Run inside the `cm` conda env.
#
#   conda activate cm
#   bash scripts/run_demo_pipeline.sh
#
# Stages: prep data -> fine-tune ControlNet -> edge-conditional sample -> metric.
# The defaults below are a TINY CPU-friendly smoke run (50 steps). For a real
# GPU run, see the "GPU" block at the bottom.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"   # ensure `import cm` works even without pip install -e .

CKPT=${CKPT:-checkpoints/cd_imagenet64_lpips.pt}   # frozen backbone
DATA=${DATA:-data/demo_shapes64}
OUT=${OUT:-runs/demo_ct}
export OPENAI_LOGDIR=$OUT
mkdir -p "$OUT"

# Shared backbone architecture flags (must match the checkpoint).
ARCH="--image_size 64 --num_channels 192 --num_res_blocks 3 --num_head_channels 64 \
 --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True \
 --resblock_updown True --weight_schedule uniform"

echo "==> [1/4] prepare tiny synthetic dataset"
[ -d "$DATA" ] || python scripts/prep_demo_data.py --out_dir "$DATA" --n 300 --size 64

echo "==> [2/4] fine-tune ControlNet (frozen backbone, conditional CT)"
# IMPORTANT: --lr_anneal_steps bounds the run. With lr_anneal_steps=0 the upstream
# run_loop condition `not self.lr_anneal_steps or ...` is ALWAYS true, so training
# never stops on total_training_steps (it runs until you Ctrl-C) and would save a
# ~5.8GB checkpoint every save_interval until the disk fills. We set it = the step
# budget so the smoke run stops cleanly after one save.
python scripts/cm_train.py \
  --training_mode consistency_training \
  --use_controlnet True --hint_channels 1 \
  --controlnet_init_checkpoint "$CKPT" \
  --data_dir "$DATA" $ARCH \
  --loss_norm l2 \
  --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 \
  --global_batch_size 8 --batch_size 8 --lr 1e-4 --use_fp16 False \
  --total_training_steps 50 --lr_anneal_steps 50 --save_interval 50 \
  --save_keep_last 2 --log_interval 5 --canny_random True

LAST=$(ls -1 "$OUT"/model*.pt | sort | tail -1)
echo "    trained checkpoint: $LAST"

echo "==> [3/4] edge-conditional sampling (triptychs)"
python scripts/demo.py --model_path "$LAST" $ARCH \
  --use_controlnet True --image_dir "$DATA" --output "$OUT/triptychs" \
  --sampler onestep --steps 1 --fixed_class 0

echo "==> [4/4] edge-fidelity metric"
# --training_mode consistency_training so image_sample uses distillation=True
# (boundary-condition c_skip/c_out), matching how the model was trained.
python scripts/image_sample.py --model_path "$LAST" $ARCH \
  --training_mode consistency_training \
  --use_controlnet True --edge_dir "$DATA" --batch_size 8 \
  --sampler onestep --steps 1 --fixed_class 0
python scripts/eval_edge_fidelity.py --npz_glob "$OUT/edge_samples_rank*.npz"

echo "Done. See $OUT/triptychs and $OUT/edge_samples_rank0.npz"

# ---------------------------------------------------------------------------
# GPU / real training (4x A6000), conditional CT, LPIPS loss, fp16:
#
#   export OPENAI_LOGDIR=runs/edge_ct_imagenet64
#   mpiexec -n 4 python scripts/cm_train.py \
#     --training_mode consistency_training --use_controlnet True --hint_channels 1 \
#     --controlnet_init_checkpoint checkpoints/cd_imagenet64_lpips.pt \
#     --data_dir /path/to/imagenet64 $ARCH \
#     --loss_norm lpips --target_ema_mode fixed --start_ema 0.95 \
#     --scale_mode fixed --start_scales 40 \
#     --global_batch_size 256 --lr 1e-4 --use_fp16 True \
#     --total_training_steps 100000 --lr_anneal_steps 100000 --save_interval 5000 --canny_random True
#
#   NOTE: --lr_anneal_steps is REQUIRED to stop (see step 2 comment). Each save is
#   ~5.8GB (full backbone+ControlNet x {model,ema,target} + opt); with save_interval
#   5000 over 100k steps that's ~20 saves = ~116GB. Watch disk, or raise save_interval.
#   (Disk-lean follow-up: save only model.controlnet -- see the tech doc TODO.)
#
# Live interview demo (Gradio web UI):
#   python scripts/demo.py --model_path runs/edge_ct_imagenet64/model<NNN>.pt \
#     --use_controlnet True --gradio --share
# ---------------------------------------------------------------------------
