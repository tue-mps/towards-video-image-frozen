#!/usr/bin/env bash
# Smoke-test every inference script with every released architecture, with and
# without --visualize. Reports PASS / FAIL / SKIP per (task, arch, mode) combo
# and exits non-zero if any combo failed. Missing checkpoints or backbone
# weights are reported as SKIP rather than FAIL, so you can run this even with
# only a subset of weights downloaded.
#
# Configure paths via the variables below (or override them with env vars):
#
#   REPO             repo root (auto-detected from this script's location)
#   PYTHON           Python interpreter to use ("python" by default)
#   CKPTS_ROOT       directory with per-task fine-tuned ckpts (default: $REPO/ckpts)
#   DINOV3_WEIGHTS   path to the official DINOv3 .pth file
#   RVM_WEIGHTS      path to the converted RVM *_pytorch_wrapper.pth file
#
# Example — set everything inline, then run:
#
#   REPO=/path/to/towards-video-image-frozen \
#   DINOV3_WEIGHTS=/path/to/dinov3_vitl16.pth \
#   RVM_WEIGHTS=/path/to/pretrain_rvm_large16_256_pytorch_wrapper.pth \
#       bash scripts_inference/run_all_inference_tests.sh
#
# Or just edit the defaults below to point at the locations on your machine.

set -uo pipefail

# -- Paths (edit these or override via env vars) ----------------------------
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"   # e.g. "/path/to/towards-video-image-frozen"
PYTHON="${PYTHON:-python}"                          # e.g. "/path/to/conda/envs/myenv/bin/python"
CKPTS_ROOT="${CKPTS_ROOT:-$REPO/ckpts}"             # released ckpts live at $CKPTS_ROOT/<Task>/<file>.ckpt
DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-/path/to/dinov3_vitl16.pth}"
RVM_WEIGHTS="${RVM_WEIGHTS:-/path/to/pretrain_rvm_large16_256_pytorch_wrapper.pth}" #converted pure PyTorch RVM ckpt
# ---------------------------------------------------------------------------

cd "$REPO"

# task_name | inference script | ckpt subdir (joined to $CKPTS_ROOT) | input video | extra args
# extra args is parsed with word-splitting (so it may include flags + paths).
TASKS=(
    "SthSthV2|scripts_inference/inference_ssv2.py|SthSthV2|video_examples/ssv2/74225.mp4|--class-names video_examples/ssv2/class_names.txt"
    "PerceptionTest|scripts_inference/inference_pt.py|PerceptionTest|video_examples/perception_test/video_6860_000000.mp4|--query video_examples/perception_test/video_6860_000000.query.json"
    "ScanNet|scripts_inference/inference_scannet.py|ScanNet|video_examples/scannet/scene0011_00.mp4|"
    "NuScenes|scripts_inference/inference_nuscenes.py|NuScenes|video_examples/nuscenes/scene-0268.mp4|"
    "Waymo|scripts_inference/inference_waymo.py|Waymo|video_examples/waymo/1024360143612057520_3580_000_3600_000.mp4|--query video_examples/waymo/1024360143612057520_3580_000_3600_000.query.json"
)

# arch | ckpt filename inside the task dir | backbone weights
ARCHS=(
    "rvm|RVM_readout.ckpt|$RVM_WEIGHTS"
    "dinov3_rvmrnn|DINOv3+Gated.ckpt|$DINOV3_WEIGHTS"
    "dinov3_gatedmambamix|DINOv3+GMM.ckpt|$DINOV3_WEIGHTS"
)

LOGDIR="/tmp/inference_tests_$$"
mkdir -p "$LOGDIR"
echo "Logs will be written to $LOGDIR"

pass=0
fail=0
skip=0
results=()

for task_def in "${TASKS[@]}"; do
    IFS='|' read -r task script ckpt_subdir video extra <<< "$task_def"
    ckpt_dir="$CKPTS_ROOT/$ckpt_subdir"
    if [ ! -f "$video" ]; then
        echo "[SKIP] $task  (input video missing: $video)"
        skip=$((skip + 1))
        results+=("SKIP    $task  *  *  (video missing)")
        continue
    fi
    for arch_def in "${ARCHS[@]}"; do
        IFS='|' read -r arch ckpt_name bw <<< "$arch_def"
        ckpt="$ckpt_dir/$ckpt_name"
        for viz_flag in "" "--visualize"; do
            mode="noviz"
            [ -n "$viz_flag" ] && mode="viz"
            label=$(printf '%-15s | %-22s | %-5s' "$task" "$arch" "$mode")
            log="$LOGDIR/${task}_${arch}_${mode}.log"
            if [ ! -f "$ckpt" ]; then
                echo "[SKIP] $label  (ckpt missing: $ckpt)"
                skip=$((skip + 1))
                results+=("SKIP    $label  (ckpt missing)")
                continue
            fi
            if [ ! -f "$bw" ]; then
                echo "[SKIP] $label  (backbone weights missing: $bw)"
                skip=$((skip + 1))
                results+=("SKIP    $label  (backbone weights missing)")
                continue
            fi
            echo
            echo "================================================================"
            echo "[RUN ] $label"
            cmd=("$PYTHON" "$script"
                 --video "$video"
                 --arch "$arch"
                 --backbone-weights "$bw"
                 --tuned-ckpt "$ckpt")
            # Append extra args (word-split) and the viz flag.
            # shellcheck disable=SC2206
            extra_arr=($extra)
            cmd+=("${extra_arr[@]}")
            [ -n "$viz_flag" ] && cmd+=("$viz_flag")
            echo "    ${cmd[*]}"
            echo "----------------------------------------------------------------"
            if "${cmd[@]}" > "$log" 2>&1; then
                tail -3 "$log" | sed 's/^/    /'
                echo "[PASS] $label"
                pass=$((pass + 1))
                results+=("PASS    $label")
            else
                tail -15 "$log" | sed 's/^/    /'
                echo "[FAIL] $label  (full log: $log)"
                fail=$((fail + 1))
                results+=("FAIL    $label  (log: $log)")
            fi
        done
    done
done

echo
echo "================================================================"
echo "SUMMARY"
echo "================================================================"
for r in "${results[@]}"; do
    echo "  $r"
done
echo
echo "pass=$pass  fail=$fail  skip=$skip"
echo "logs: $LOGDIR"

exit "$fail"
