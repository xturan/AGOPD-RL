# Infrastructure Notes

## Local Runtime Observed

```text
Python: 3.12.7
PyTorch: 2.5.1+cu124
NVIDIA driver: 560.35.03
nvidia-smi CUDA: 12.6
GPU: 2x NVIDIA GeForce RTX 4090, 24 GiB each
CUDA visibility: 2 devices
Dual-GPU tensor computation: passed
```

The current local shell can communicate with the NVIDIA driver and both GPUs
can execute CUDA work. GPU 1 has a small desktop graphics allocation, so local
training should reserve both cards but avoid unrelated GUI workloads when
running long jobs.

## Training Stack

The project spec defaults to `verl` for GRPO/post-training and vLLM or SGLang for
rollout generation. The current `verl` installation guide recommends CUDA >= 12.8
and careful version matching between PyTorch and inference backends.

Primary reference:

```text
https://github.com/verl-project/verl/blob/main/docs/start/install.rst
https://verl.readthedocs.io/en/latest/start/install.html
```

## Cloud Smoke-Test Stack

The current cloud image exposes 4xA100 with NVIDIA's Torch 2.7 build. The latest
`verl` main branch is not compatible with this stack: it requires a newer
server-based rollout backend and no longer wires `HFRollout` into the worker.

For the first GRPO smoke test, use the following deliberately pinned profile:

```text
verl: v0.5.0
rollout: Hugging Face, synchronous
worker: legacy FSDP
torch: existing NVIDIA 2.7 build
numpy: 1.26.4
tensordict: 0.9.1
transformers: 4.55.4
vLLM: not required
```

Prepare the cloud checkout without replacing the working NVIDIA Torch build:

```bash
cd ${AGOPD_ROOT}/verl
git fetch --tags
git switch --detach v0.5.0

env -u PIP_CONSTRAINT python3 -m pip install \
  "numpy==1.26.4" "transformers==4.55.4" \
  --break-system-packages

env -u PIP_CONSTRAINT python3 -m pip install \
  --no-deps "tensordict==0.9.1" \
  --break-system-packages

python3 -m pip install --no-deps --no-build-isolation -e . \
  --break-system-packages

patch -p1 < \
  ${AGOPD_ROOT}/patches/verl-v0.5.0-hf-rollout-repeat.patch

patch -p1 < \
  ${AGOPD_ROOT}/patches/verl-v0.5.0-hf-rollout-position-ids.patch
```

Then run `bash scripts/run_grpo_smoke_hf.sh` from the project root. This profile
is for correctness validation. Formal high-throughput training should use a
separate, version-matched verl + vLLM image.

The smoke launcher uses `src/agopd/reward/verl_adapter.py`, so rewards follow the
project's binary `0/1` contract instead of verl v0.5.0's built-in `-1/1` DAPO
reward. Generated samples and verifier details are written to
`outputs/grpo_smoke_hf/rollouts/<step>.jsonl`.

The Qwen3 thinking-mode smoke defaults follow the model's recommended sampling
parameters (`temperature=0.6`, `top_p=0.95`, `top_k=20`). It uses eight prompts
with two responses each and a 2048-token response limit, preserving 16 total
rollouts while satisfying the legacy FSDP batch-divisibility constraint. For this legacy HF
rollout, `use_remove_padding` stays disabled because it monkey-patches the
FlashAttention path used by generation with KV cache.

The compatibility patch keeps `rollout.n>1` but forces Hugging Face generation
to return one sequence per already-repeated prompt. Without it, verl v0.5.0
repeats both in `ray_trainer` and `HFRollout`, producing incompatible batch sizes
(for example, 16 input rows and 32 generated rows).

The second patch lets Qwen3's `generate()` derive position IDs from its attention
mask. Passing legacy left-padded position IDs directly causes degenerate
generation, even outside FSDP.

## Local RTX 4090 Mirror

The local machine can mirror the validated cloud vLLM stack on two RTX 4090
cards without changing system Python or the cloud launchers. The setup uses a
isolated Conda environment at `~/.conda/envs/agopd-vllm-281`, extracts the checked-in
`verl-v0.6.1.tar.gz` into `.runtime/verl-v0.6.1`, and keeps the critical pins:

```text
Python: 3.12
PyTorch: 2.8.0
vLLM: 0.10.2
verl: v0.6.1
NumPy: 1.26.4
Transformers: 4.55.4
tensordict: 0.9.1
FlashInfer: 0.3.0 (optional; binary wheel only)
```

Run these commands inside the local GPU host/container:

```bash
cd /path/to/RL
export AGOPD_PROXY_URL=http://<PROXY_HOST:PORT>
bash scripts/bootstrap_local_vllm.sh
bash scripts/check_local_env.sh
bash scripts/run_local_grpo_vllm.sh
```

The bootstrap exports `PYTHONNOUSERSITE=1` so packages from `~/.local` cannot
silently satisfy dependencies inside the Conda environment.

The bootstrap script deliberately does not pass an index URL, so the global
Tsinghua PyPI configuration is preserved. It also unsets a possible inherited
`PIP_CONSTRAINT`, which previously caused unrelated NumPy/Transformers
resolution conflicts. FlashInfer is installed only when a matching binary wheel
is available; the local GRPO wrapper uses two GPUs while preserving the
cloud global batch size of 8; set `AGOPD_TRAIN_STEPS=10` or pass Hydra overrides
for a longer/shorter run. The local SFT wrapper sets `torchrun` to two workers:

```bash
bash scripts/run_local_sft.sh --max-steps 50
```

The local driver check has now passed. The driver reports CUDA 12.6 while the
validated cloud vLLM stack uses Torch 2.8/CUDA 12.8, so run the environment
bootstrap and its package/CUDA check before starting a long local job. If the
Torch 2.8 wheel fails to initialize, keep the current working Torch 2.5.1
environment intact and record the error before changing pins; the cloud
environment remains unaffected.
