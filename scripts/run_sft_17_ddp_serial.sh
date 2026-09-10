#!/usr/bin/env bash
# 1.7B 冷启动 v1/v2 对比 SFT — 双卡 DDP 串行(先 v1 后 v2)
# 同硬件(2×4090 DDP)、同配方(LoRA r16 lr1e-5, 3 epochs + 早停)保证可比
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_PY=${HOME}/.conda/envs/agopd-vllm-281/bin/python
RUN="${ENV_PY} -m torch.distributed.run"

# v1: 旧数据(4B-GRPO rejection,1300 条)→ 3 epochs = ~975 steps(DDP global batch 4)
${RUN} --nproc_per_node=2 --master_port=29521 scripts/run_sft_smoke.py \
  --data data/coldstart-17-sft/train.jsonl \
  --eval-data data/coldstart-17-sft/val.jsonl \
  --model models/Qwen3-1.7B --output outputs/sft-coldstart-17-v1-ddp3ep \
  --max-length 2560 --learning-rate 1e-5 --lora-rank 16 \
  --max-epochs 3.0 --max-steps 2000 --eval-steps 100 --save-steps 100 --patience 4 \
  > logs/sft17_v1_ddp3ep.log 2>&1
echo "[$(date +%H:%M)] V1_DDP_DONE"

# v2: 4B RL 轨迹(4795 条)→ 3 epochs ≈ 3600 steps
${RUN} --nproc_per_node=2 --master_port=29522 scripts/run_sft_smoke.py \
  --data data/coldstart-17-v2-sft/train.jsonl \
  --eval-data data/coldstart-17-v2-sft/val.jsonl \
  --model models/Qwen3-1.7B --output outputs/sft-coldstart-17-v2-ddp3ep \
  --max-length 2560 --learning-rate 1e-5 --lora-rank 16 \
  --max-epochs 3.0 --max-steps 4000 --eval-steps 200 --save-steps 200 --patience 4 \
  > logs/sft17_v2_ddp3ep.log 2>&1
echo "[$(date +%H:%M)] V2_DDP_DONE"
echo "SERIAL_SFT_ALL_DONE"
