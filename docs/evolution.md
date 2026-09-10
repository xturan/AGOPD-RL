# AGOPD-RL Evolution Log

This document records the project path, validated decisions, incidents, and
their resolutions. Update it whenever an environment, data, rollout, reward,
or algorithm decision changes.

## Current Status

Date: 2026-08-30

The deterministic reward, DAPO split generation, GRPO advantage utilities, and
advantage gate are implemented and covered by local tests. The 4xA100 cloud
environment has completed a one-step GRPO execution with non-zero reward
variation, advantages, and policy loss. A four-GPU 50-step LoRA SFT smoke run
also completed successfully on the OpenR1 corpus. SFT plus `/no_think` has
improved the GRPO smoke rollout quality, and the Qwen3 legacy HF rollout
compatibility patches are validated.

## Target Path

```text
Qwen3-1.7B student + deterministic math reward
  -> GRPO baseline with non-zero group advantages
  -> compact math-format SFT only if valid rollout pass@G remains too low
  -> Advantage-Gated OPD
  -> disagreement gate and teacher budget ablations
  -> vLLM/SGLang migration for formal high-throughput training
```

The teacher is Qwen3-4B. The initial dataset is DAPO-Math-17k. The research
metric is not only accuracy: it also includes teacher token ratio, teacher GPU
hours, and performance gain per teacher compute.

## Verified Assets

| Asset | Status | Notes |
| --- | --- | --- |
| DAPO-Math-17k | Ready | 1,791,700 rows; 17,917 repeated prompt IDs; split creation deduplicates by `extra_info.index`. |
| Qwen3-1.7B | Verified | Both safetensor SHA256 values match the official Qwen repository. |
| Qwen3-4B | Downloaded | Reserved as the teacher for later OPD. |
| OpenR1 SFT smoke corpus | Ready | Compact, answer-normalized chat examples from one verified OpenR1 default shard. |
| Local project tests | Passing | `23 passed` as of this update. |
| Cloud hardware | Ready | 4x NVIDIA A100-SXM4-80GB, CUDA available. |

## Environment Decisions

| Decision | Rationale |
| --- | --- |
| Use CUDA 12.x images with driver 550 | CUDA 13.x requires a newer host driver. |
| Pin smoke stack to verl v0.5.0 | Current verl main requires a newer server rollout stack and no longer supports the legacy HF rollout path. |
| Keep NumPy 1.26.4 | The NVIDIA Torch build could not initialize NumPy when NumPy 2.x was installed. |
| Use global Tsinghua PyPI mirror | Do not override the configured global index with an Aliyun index. |
| Use HF rollout only for correctness smoke tests | Formal training should move to a version-matched vLLM or SGLang image. |

## Rollout Incident Log

| ID | Symptom | Root Cause | Resolution | Status |
| --- | --- | --- | --- | --- |
| R1 | `Rollout hf with mode async not found` | Current verl worker registry supports server backends only. | Pin smoke stack to verl v0.5.0 legacy FSDP worker with `rollout.mode=sync`. | Resolved |
| R2 | Batch size `16` versus `32` on `DataProto.union` | v0.5.0 repeated prompts in `ray_trainer` and also used `num_return_sequences=rollout.n` in HF rollout. | Apply `verl-v0.5.0-hf-rollout-repeat.patch`; generation returns one sequence for each already-repeated prompt. | Resolved |
| R3 | `real_train_batch_size (4) must be divisible by ... (8)` | Legacy FSDP requires `train_batch_size * rollout.n` divisible by GPU count. | Use `train_batch_size=8`, `rollout.n=2` on 4 GPUs. | Resolved |
| R4 | Repetitive multilingual text, nested `<think>`, all reward zero | Qwen3 thinking mode was sampled with `temperature=1.0`, `top_k=0`; also old HF rollout monkey-patched padding path. | Use Qwen-recommended sampling: `0.6 / 0.95 / 20`, disable `use_remove_padding` for HF smoke. | Partially resolved |
| R5 | Degenerate DAPO generation persisted outside FSDP | v0.5.0 passes left-padded `position_ids` to Qwen3 `generate()`. A single-GPU reproduction became coherent when that argument was omitted. | Apply `verl-v0.5.0-hf-rollout-position-ids.patch`. | Resolved |
| R6 | Every rollout reaches the 2048-token response cap | Qwen3 thinking trajectories are longer than the smoke response budget. | OpenR1 LoRA SFT plus `/no_think` reduced cap rate from 100% to 12.5%. | Resolved for smoke |
| R7 | Qwen3-4B teacher did not solve a sampled hard DAPO problem within 2048 thinking tokens | The available teacher is not reliable enough to synthesize the primary SFT corpus online. | Use OpenR1-Math-220k `default` verified traces as the primary SFT corpus. Keep teacher generation as optional data augmentation. | Resolved |
| R8 | Compact SFT cold start was unverified | The project lacked a distributed SFT path and formatted supervision corpus. | 8k answer-normalized OpenR1 examples + 4xA100 LoRA SFT completed for 50 steps in 38.4 seconds; final mean training loss was 1.16. | Resolved |
| R9 | LoRA merge command exited without creating model files | `merge_sft_adapter.py` defined `main()` but lacked the executable entrypoint. | Add the `if __name__ == "__main__"` entrypoint and verify `--help` output before upload. | Resolved |
| R10 | SFT quality improvement was unverified in RL | The SFT adapter had not been evaluated on on-policy DAPO rollout. | One-step SFT + `/no_think` GRPO smoke reached 18.75% reward accuracy, versus 6.25% before SFT. | Resolved |
| R11 | Long SFT run on the 8k corpus completed | The compact SFT smoke was only 50 steps. | The long 8k-corpus SFT run has been reported complete; final loss, duration, and checkpoint integrity are pending capture. | Pending verification |
| R12 | Long-SFT model one-step GRPO quality was mixed | A single 16-rollout batch is noisy, while SFT reduced response length and increased confidence. | Merged long-SFT model completed one GRPO step with reward mean 0.125, mean response length 819, and cap rate 25%. Run 10 steps before judging quality. | Open |
| R13 | HF rollout takes about 107 seconds per step | HF generation is sequential and dominates the 113-second GRPO step. | Migrate to vLLM rollout in a separate Torch 2.8 venv. | Resolved |
| R14 | vLLM venv contains Torch 2.13, NumPy 2.5, and verl 0.5 simultaneously | An unpinned dependency install upgraded Torch and mixed the old system/verl environment with the new vLLM stack. | Keep the polluted venv as a fallback and create a clean venv with Torch 2.8, vLLM 0.10.2, NumPy 1.26.4, and verl v0.6.1. | Resolved |
| R15 | vLLM run fails before rollout because `flash_attn` is missing | verl v0.6.1 defaults actor/ref FSDP initialization to FlashAttention 2, even when rollout uses vLLM and padding removal is disabled. | Set `actor_rollout_ref.model.override_config.attn_implementation=sdpa`; vLLM keeps its own optimized generation kernels. | Resolved |
| R16 | vLLM sampling initially reported missing FlashInfer | FlashInfer is an optional sampling optimization absent from the initial vLLM venv. | Install the version-matched `flashinfer-python==0.3.0`; vLLM now uses FlashInfer sampling without warnings. | Resolved |
| R17 | GPU compute/memory utilization is about 38% with vLLM | The 1.7B model and small per-step request batch underfill 4xA100; `gpu_memory_utilization=0.35` also limits KV cache capacity. | Benchmark a larger request batch, `max_num_batched_tokens=16384`, `gpu_memory_utilization=0.5`, and `enforce_eager=False` after the current run. Keep TP=1. | Planned |
| R18 | 10-step vLLM GRPO baseline completed | A one-step run cannot reveal reward or entropy trends. | 10 steps completed in 6m05s at roughly 36.5s/step, with reward variation, no OOM, and no monotonic entropy collapse. | Resolved for baseline |
| R19 | 50-step run completed but Ray reported a killed DataLoader worker at exit | Four Ray workers plus `data.dataloader_num_workers=8` created unnecessary CPU memory pressure; the checkpoint was already saved. | Keep `global_step_50`; use `data.dataloader_num_workers=0` or `2` for resumed long runs and preserve periodic checkpoints. | Resolved for continuation |
| R20 | Resume failed loading `data.pt` after lowering DataLoader workers | The saved StatefulDataLoader state was created with multiple workers and cannot be loaded by a single-process DataLoader. | Back up `global_step_50/data.pt`, resume model/optimizer/RNG/scheduler, and start a fresh dataloader state with zero workers. | Resolved |
| R21 | Resume from step50 to step100 completed | Continuation needed confirmation after skipping the incompatible dataloader state. | Model, optimizer, RNG, and scheduler loaded; step100 checkpoint saved with no worker kill or OOM. Reward remained variable, with visible steps reaching 0.3125 and 0.375. | Resolved |
| R22 | Intermediate metrics were only visible in console output | The vLLM launcher used `logger=["console"]`, so there were no time-series event files. | Enable `logger=["console","tensorboard"]` and set a persistent `TENSORBOARD_DIR` in the vLLM launcher. | Resolved |
| R23 | 200-step run returned to shell around step128 without a traceback in the captured log | Ray logs show the driver called `ray.shutdown()`, then GCS received `SIGTERM` and cleaned up workers. No contemporaneous kernel OOM entry was found. | Treat this as external driver/session termination until the platform supplies a termination event. Use a persistent `nohup`/scheduler job and resume from the latest complete checkpoint. | Resolved diagnostically |
| R24 | Evaluation artifact was incomplete at step128 | The run produced 128 rollout files and a TensorBoard event file, but no final 200-step checkpoint; the captured log ends immediately after the step128 progress line. | Pull the event file, all rollout JSONL files, held-out validation parquet, and launcher log; evaluate the completed prefix. Cloud inspection confirms `global_step_120` is complete and is the resume point. | In progress |
| R25 | The smoke validation set had only 128 randomly selected rows | A single small random slice is noisy and the old split did not record a durable dataset fingerprint or quality manifest. | Build `dapo-verl-v1` with unique prompt IDs, exact prompt-fingerprint de-duplication, fixed seeded hash selection, prompt-length strata, a manifest, and a quality report. Use it unchanged for every checkpoint comparison. | Resolved |
| R26 | SFT and GRPO step120 underperformed the base model on the fixed DAPO validation set | OpenR1 SFT improved termination and answer formatting but transferred poorly to DAPO accuracy; the GRPO baseline then optimized a sparse reward with low group diversity and did not recover the lost accuracy. | Do not promote the SFT or GRPO checkpoint. Recheck the SFT corpus and learning rate, add a fixed validation gate before long RL, and increase rollout group size before implementing AGOPD. | Open |
| R27 | The first SFT diagnostic mixed a no-thinking format with a shifted answer distribution | All 8k prompts used `/no_think`; none of the assistant traces used Qwen3 `<think>` tags; DAPO ground truths are numeric while the SFT corpus also contains letter and other answer forms. | Treat current SFT as a format experiment, not a capability-preserving cold start. Compare base no-thinking GRPO against a lower-intensity, DAPO-aligned SFT ablation before selecting a student checkpoint. | Open |
| R28 | GPU0 retained an unusable 43 GiB CUDA context | The cloud driver reported an unnamed stale process that could not be reset; no active training process owned it. | Run P2 on free GPU1-3 with `n_gpus_per_node=3` and a divisible batch size. Recreate the cloud container before the formal four-GPU run. | Open |
| R29 | Base -> GRPO(n=4) 10-step smoke completed | The clean Base initialization produced nonzero reward variation and a mean effective group rate of 35.0%; step10 was slightly above Base on the matched full validation protocol. | Keep the run as a healthy baseline candidate, repeat with matched seeds or a larger evaluation before claiming a gain, then proceed to 50-100 steps if the signal holds. | Resolved for smoke |
| R30 | Base -> GRPO(n=4) 50-step four-GPU run completed | Reward improved modestly, but fixed validation accuracy moved only from 22.17% to 24.32% and truncation did not improve. | Treat GRPO as a stable but near-plateau baseline. Preserve `global_step_50`, run a matched repeat/second seed if needed, then move to Vanilla OPD. | Resolved for baseline |
| R31 | Vanilla OPD smoke could not reach the first step on the CUDA 12.9/vLLM 0.10.2 stack | verl v0.6.1 has no distillation trainer; v0.8.0 supplies OPD but its async vLLM rollout server conflicts with colocated FSDP/Teacher GPU allocation, while its HF fallback has been removed. | Keep the working GRPO environment unchanged. Isolate OPD in `agopd-opd-venv-080`; next use a version-matched OPD image or a separate rollout node/resource pool before measuring Vanilla OPD. | Blocked by runtime topology |
| R32 | The OPD conflict is enforced by the cloud GPU policy | All four A100s report `Compute Mode: Exclusive_Process`; changing to Default from inside the container returns `Insufficient Permissions`. | Ask the platform/host administrator to run `nvidia-smi -c 0 -i 0,1,2,3`, or provide a node with Default compute mode/MPS support. Do not change the working GRPO venv. | Blocked by host permission |
| R40 | Local 4090 resources were not yet reproducibly usable | The project had cloud-specific absolute paths and no local environment bootstrap; the current Codex shell also cannot see the local NVIDIA driver. | Add a project-local venv bootstrap with the validated Torch 2.8/vLLM 0.10.2/verl v0.6.1 pins, a two-GPU environment check, fixed-set evaluation, and local GRPO/SFT wrappers. Run the final CUDA check inside the actual 4090 host/container. | Implemented; hardware verification pending |
| R41 | Local GPU visibility needed confirmation | The earlier setup shell could not access NVIDIA, but the current local runtime may expose the host devices. | Verified `nvidia-smi`, two RTX 4090 devices, driver 560.35.03/CUDA 12.6, and real FP16 matrix multiplication on both GPUs. Keep the cloud-compatible bootstrap pins under validation because the local driver/runtime differs. | Resolved for hardware access |
| R42 | FlashInfer blocked the local dependency install | The Tsinghua mirror served `flashinfer-python==0.3.0` as a source archive; pip began building isolated Torch dependencies. | Make FlashInfer optional in the local bootstrap and install it only with `--only-binary=:all:` when a matching wheel exists. Core vLLM installation proceeds independently. | Resolved diagnostically |
| R43 | Direct Conda-Python vLLM smoke loaded the system libstdc++ | Calling the environment's absolute Python does not activate Conda's library path, so ICU could resolve the system C++ runtime instead of Conda's ABI-compatible copy. | Local check/run wrappers export `LD_LIBRARY_PATH=${AGOPD_ENV_DIR}/lib` and disable user-site packages. | Fixed; smoke verification pending |
| R44 | Local dependency downloads were slow and root temporary storage filled | Large CUDA wheels went through a slow direct mirror path and pip unpacked them under `/tmp`, which had limited free space. | Route bootstrap downloads through the local HTTP proxy at `<PROXY_HOST:PORT>`, keep pip temp/cache on the project disk, and pin the resolver's OpenCV/CuPy/SciPy choices. | Resolved |
| R45 | Local Conda vLLM runtime needed final functional validation | Package imports alone do not prove that the local CUDA runtime, vLLM engine, and model weights work together. | Run a one-sample Qwen3-1.7B vLLM evaluation on GPU0; model loading and generation passed at about 58 output tokens/s. | Resolved |
| R46 | Local two-GPU Conda environment completed | The first installation attempt filled `/tmp`, and the second exposed resolver conflicts and a missing Conda library path. | Use the proxy/cache relocation, NumPy-compatible pins, `PYTHONNOUSERSITE=1`, and `LD_LIBRARY_PATH=<conda>/lib`. Final `pip check` is clean; Torch 2.8/vLLM 0.10.2/verl 0.6.1 and dual-GPU compute are verified. | Resolved |
| R40 | Local 4090 resources were not yet reproducibly usable | The project had cloud-specific absolute paths and no local environment bootstrap; the current Codex shell also cannot see the local NVIDIA driver. | Add a project-local venv bootstrap with the validated Torch 2.8/vLLM 0.10.2/verl v0.6.1 pins, a two-GPU environment check, and local GRPO/SFT wrappers. Run the final CUDA check inside the actual 4090 host/container. | Implemented; hardware verification pending |

## Reward and Signal Status

The project reward adapter is `src/agopd/reward/verl_adapter.py`. It exposes the
project's deterministic binary reward contract to verl:

```text
correct final answer -> 1.0
wrong or missing final answer -> 0.0
```

It accepts `Answer: 99`, `Answer: $99`, and `\boxed{99}`. Rollout JSONL files
record score, predicted answer, and verifier reason.

Observed before R5 was fixed:

```text
all rewards = 0
all group advantages = 0
actor pg_loss = 0
```

This is not a usable RL signal. Do not run multi-step GRPO until a one-step run
contains coherent outputs and at least some reward variation within groups.

Validated after R5:

```text
step = 1
reward mean = 0.0625, min = 0.0, max = 1.0
advantage min/max = -0.7071 / 0.7071
actor pg_loss = -0.1768
```

Validated after SFT plus `/no_think`:

```text
step = 1
reward mean = 0.1875, min = 0.0, max = 1.0
advantage min/max = -0.7071 / 0.7071
response length mean = 1100 / 2048
response cap rate = 12.5%
```

This is a 3x relative increase in smoke accuracy and an 87.5 percentage-point
decrease in response cap rate compared with the pre-SFT rollout. The first-step
PPO policy loss is zero because the on-policy ratio starts near one; advantage
variation confirms that multi-step GRPO now has usable learning signal.

The merged long-SFT model's one-step check produced reward mean `0.125`,
non-zero advantages, response length mean `819`, cap rate `25%`, and entropy
`0.119`. The lower one-batch accuracy than the 50-step adapter is not yet a
regression claim; use a multi-step run to separate sampling variance from
over-confident SFT behavior.

## SFT Decision Rule

Do not use SFT to mask an inference or rollout bug. Qwen3-1.7B has passed its
native SDPA and FlashAttention generation checks.

Run compact math-format SFT when DAPO trajectories are coherent but rarely
reach a final answer before the rollout cap. This condition is now met: the
position-ID-fixed smoke run achieved 6.25% accuracy, 25% answer-format rate,
and 6.25% closed-thinking rate, with every response capped at 2048 tokens.

The SFT objective is to make solutions concise and reliably terminate in
`Answer:` or `\boxed{}`, creating reward diversity for GRPO. The primary data
source is a compact subset of verified `open-r1/OpenR1-Math-220k` default
traces, prepared by `scripts/prepare_openr1_sft.py`. Teacher-generated data is
optional augmentation only; it is retained only when the deterministic verifier
accepts it. Do not use final answers alone.

## Next Actions

1. Summarize the SFT + `/no_think` rollout JSONL to confirm final-answer format
   rate and inspect successes/failures.
2. Run a 10-step multi-step GRPO baseline from the merged long-SFT model and
   compare it with the 50-step adapter checkpoint.
3. Evaluate the step50, step100, and step120 checkpoints on a fixed held-out DAPO slice;
   do not select a checkpoint from a single rollout batch reward.
4. Move formal experiments to a version-matched vLLM/SGLang image before OPD
   and teacher-budget measurements.

Create the formal fixed validation set once from the full DAPO parquet:

```bash
PYTHONPATH=src python3 scripts/prepare_dapo_validation.py \
  --input data/dapo-math-17k/data/dapo-math-17k.parquet \
  --output-dir data/dapo-verl-v1 \
  --val-size 1024 \
  --seed 42 \
  --append-no-think
```

This creates:

```text
data/dapo-verl-v1/train.parquet
data/dapo-verl-v1/val.parquet
data/dapo-verl-v1/manifest.json
data/dapo-verl-v1/quality_report.json
```

The validation set is selected after collapsing the 100 repeated records per
DAPO prompt. The train parquet contains the remaining unique prompts, so it
cannot leak a validation prompt ID or an exact duplicate prompt text. Keep the
manifest with every experiment; do not regenerate this directory during a
checkpoint comparison.

The vLLM migration keeps the host environment untouched by using a dedicated
venv with Torch 2.8 and vLLM 0.10.2, while the existing HF environment remains
available for debugging.

Validated vLLM one-step baseline on 2026-08-30:

```text
generation time = 34.7 seconds
total step time = 47.8 seconds
throughput = 113.2 tokens/second
reward mean = 0.1875
advantage min/max = -0.7071 / 0.7071
response length mean = 1179.8 / 2048
response cap rate = 31.25%
flashinfer = enabled
```

Validated 10-step vLLM baseline:

```text
steps = 10
runtime = 6m05s
mean step time ~= 36.5s
reward variation = present; some steps reached 0.1875 and some 0.0
response cap rate = roughly 0.06 to 0.31 per step
entropy range = roughly 0.086 to 0.154
OOM/rollout crash = none
```

From the next run, start the metrics dashboard with:

```bash
tensorboard --logdir ${AGOPD_ROOT}/tensorboard \
  --bind_all --port 6006
```

The launcher writes one scalar series per training step, including reward,
advantage, entropy, KL, response length, cap rate, timing, throughput, and
memory metrics.

For checkpoint evaluation, collect the following minimum artifact set:

```text
tensorboard/grpo_vllm_200step/events.out.tfevents.*
outputs/grpo_vllm_200step/rollouts/*.jsonl
data/dapo-verl-smoke-no-think/val.parquet
/tmp/grpo_vllm_200step.log
```

The optional model-comparison set adds the actor folders for
`global_step_50`, `global_step_100`, and the latest completed checkpoint. The
minimum set is enough to evaluate reward curves, answer-format rate, truncation,
advantage diversity, entropy, KL, throughput, and termination behavior.

Evaluation of the pulled `grpo_vllm_200step` prefix on 2026-08-30:

```text
completed rollout steps = 128 / 200
rollout rows = 2,048 (16 responses per step)
row-level reward accuracy = 234 / 2,048 = 11.43%
answer-marker rate = 88.28%
boxed-marker rate = 78.76%
closed-thinking rate = 99.90%
last complete step = 128
latest saved confirmed checkpoint = global_step_120
```

The per-step reward remained noisy rather than showing a monotonic rise; it
ranged from `0.0` to `0.3125`, with the final step at `0.0625`. This prefix is
useful for diagnosing rollout and metric behavior, but it is not sufficient to
claim that the model improved. A fixed held-out validation run is still needed
to compare the base, step50, step100, and step120 checkpoints.

Fixed validation comparison completed on 2026-08-30 using the same 1,024
`dapo-verl-v1` rows, `/no_think` prompts, sampling seed `42`, and vLLM sampling
parameters `temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_tokens=2048`:

| Model | Correct | Accuracy | Answer marker | Boxed marker | Truncation | Mean response tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B base | 239/1024 | 23.34% | 25.49% | 69.82% | 19.92% | 1261.7 |
| Qwen3-1.7B SFT | 88/1024 | 8.59% | 85.64% | 55.47% | 14.65% | 825.1 |
| GRPO global_step_120 | 123/1024 | 12.01% | 89.65% | 83.59% | 10.35% | 958.7 |

Conclusion: SFT and GRPO improved answer termination and formatting, but both
lost mathematical accuracy relative to the base model. The step120 model is
`11.33` percentage points below base and `3.52` points above SFT, so the GRPO
run partially recovered SFT's loss but did not produce a net gain. This is a
checkpoint selection result, not a sampling claim: all three models were run
with the same fixed prompts and sampling settings.

## SFT Cold-Start Diagnostic

The current SFT corpus contains 8,000 OpenR1 examples. It has zero exact prompt
overlap with DAPO, which is useful for measuring transfer, but its supervision
format differs from DAPO: every prompt ends in `/no_think`, every assistant
trace is visible text without Qwen3 thinking tags, and only 3,717 answers are
plain numeric strings. DAPO's 17,917 unique ground truths are numeric. This is
not automatically invalid, but it makes the current SFT a poor default for
capability-preserving initialization.

The length/mode diagnostic used the same sampling settings on fixed subsets:

```text
mode                         rows   accuracy   truncation
base no-thinking, 512         256      6.64%       86.72%
SFT  no-thinking, 512         256      4.30%       58.59%
base thinking, 512            256      0.00%      100.00%
SFT  thinking, 512            256      0.00%       99.61%
base thinking, 2048            32      3.13%       96.88%
base thinking, 4096            16     12.50%        0.00%
SFT  thinking, 4096            16      6.25%        0.00%
```

The 512-token results show that the budget is too short in both modes. The
2048-token thinking result is still dominated by truncation; thinking needs a
larger budget on this student. The full 1,024-row no-thinking comparison is
therefore the fair result for the current 2,048-token GRPO setup, where base
reached 23.34% and SFT reached 8.59%.

Decision for the next experiment: keep the base no-thinking model as the
reference, and do not promote the current SFT checkpoint. Run a small SFT
ablation with lower learning rate and fewer passes, filter or normalize
answers to the DAPO numeric contract, and evaluate before RL. Keep a separate
thinking track at `max_response_length >= 4096`; do not compare it directly to
the 2,048-token no-thinking track.

As a coarse trend check, the first and last 16 completed steps were:

```text
                     reward   entropy   response_len   cap_rate
steps 1-16            0.0820   0.1022       924.4       17.58%
steps 113-128         0.1367   0.1006      1014.4       15.23%
```

This suggests a weak late-window uplift, but the batch-level variance is large
and the response length is higher. It should not yet be interpreted as a
generalization improvement.

The local diagnostic archive is `agopd-eval-step128.tar.gz` with SHA256
`1186a3f79ded31b718d01ba4ed14bc0d95e9c9bede5f3d1f2ff940e795e6e28c`.
The archive contains the TensorBoard event file, 128 rollout JSONL files, the
held-out validation parquet, and the launcher log. The large step120 actor and
optimizer shards remain on the cloud and should be evaluated there or copied
only when a local model comparison is required.

P0/P2 Base-to-GRPO(n=4) protocol completed on 2026-08-30:

```text
protocol: no-thinking, temperature=0.7, top_p=0.8, top_k=20, max_tokens=2048
P0 Base, dev256: 63/256 = 24.61%
P0 Base, full1024: 227/1024 = 22.17%
P2 GRPO step10, dev256: 78/256 = 30.47%
P2 GRPO step10, full1024: 251/1024 = 24.51%
P2 mean EGR over 10 steps: 35.00%
P2 EGR range: 20.83% to 58.33%
```

The full-set step10 delta over the matched Base run is `+2.34` percentage
points. This is a preliminary positive signal, not yet a statistically secure
claim; the dev256 delta is larger and demonstrates sampling variance. The P2
run used GPUs 1-3 because GPU0 had a stale non-resettable context.

P3 Base-to-GRPO(n=4) 50-step run completed on four GPUs:

```text
runtime = 39m50s
reward mean, first 10 steps = 0.2406
reward mean, last 10 steps = 0.2813
EGR mean = 36.75%
EGR mean, last 10 steps = 32.50%
Base validation accuracy = 22.17%
step50 validation accuracy = 24.32%
step50 truncation rate = 21.97%
```

The `+2.15` percentage-point validation change is encouraging but small; the
unchanged truncation rate and late EGR plateau argue against extending this
exact GRPO configuration indefinitely. The next main comparison is Vanilla
OPD from the same Base initialization. The detailed local record is
`reports/p3_base_grpo_n4_50step.json`.

The 50-step run reached `global_step_50` and saved actor, optimizer, extra state,
and Hugging Face configuration. The shell process later reported a DataLoader
worker killed by `SIGKILL`; this is treated as a host-side cleanup failure, not
loss of the completed checkpoint.

Use the following command on the cloud run directory for the first action:

```bash
PYTHONPATH=src python3 scripts/summarize_rollouts.py \
  outputs/grpo_hf_position_fixed/rollouts/1.jsonl \
  --show 4 --tail-chars 1200
```

Prepare the first OpenR1 SFT shard on the cloud:

```bash
hf download open-r1/OpenR1-Math-220k \
  data/train-00000-of-00010.parquet \
  --repo-type dataset \
  --local-dir data/openr1-math

PYTHONPATH=src python3 scripts/prepare_openr1_sft.py \
  --input data/openr1-math/data/train-00000-of-00010.parquet \
  --output data/openr1-sft/smoke-8000.jsonl \
  --limit 8000
```

Local artifact created on 2026-08-30:

```text
data/openr1-sft/smoke-8000.jsonl
rows: 8000
sha256: 575a2f46b13c4f5fa4292f8649ee2032bd3d12cde66170db88b327d86be20586
Note: conversion appends the authoritative OpenR1 `answer` as the final
`Answer:` line so multiple-choice traces such as `\\boxed{E}` still teach the
numeric answer expected by the DAPO verifier. User prompts also end in
`/no_think` to align SFT with compact rollout behavior.
```

Run the four-GPU LoRA SFT smoke after uploading the artifact:

```bash
bash scripts/run_sft_smoke.sh
```

The smoke run uses 50 optimization steps, 4096 tokens, and saves its LoRA
adapter under `outputs/sft-smoke-qwen3-1.7b`.

The first long SFT run uses the 8,000-example OpenR1 smoke corpus for three
epochs: 3,000 optimizer steps with global effective batch size 8 (four GPUs,
per-device batch size 1, gradient accumulation 2). Save it separately from the
50-step smoke adapter under `outputs/sft-qwen3-1.7b-openr1-8k`.

After a successful SFT smoke, merge its adapter and create no-thinking DAPO
splits before the next GRPO rollout check:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/merge_sft_adapter.py

PYTHONPATH=src python3 scripts/prepare_dapo_splits.py \
  --output-dir data/dapo-verl-smoke-no-think \
  --append-no-think
```

## R33: Exclusive_Process 下的 OPD 隔离

云端 A100 的驱动将所有 GPU 设置为 `Exclusive_Process`，容器内没有权限
执行 `nvidia-smi -c 0` 修改主机级 compute mode。因此 OPD 不能沿用
v0.8 默认的 colocated vLLM 路径，否则 vLLM 子进程会与 FSDP worker 争用
同一张卡。

本轮修改仅作用于独立的 `verl-v0.8.0` 源码：

- `llm_server.py` 在 `rollout.nnodes > 0` 时创建独立 Ray resource pool；
- `vllm_rollout.py` 让所有学生 actor rank 指向同一个 standalone server；
- 云端原文件保留为 `*.pre-standalone`，现有 GRPO 的 v0.6.1 环境和脚本不变；
- OPD 冒烟拓扑为学生 FSDP 2 卡、学生 rollout 1 卡、教师 1 卡。

验证日志已确认 vLLM standalone server 成功分配到独立 GPU，且没有旧的
`CUDA-capable device(s) busy` 或 `vllm_server_1_0` 查找错误。随后流程卡在
standalone vLLM 的服务/RPC 初始化，停在 `skip wake_up in standalone mode`，
没有产出 `step:1` 或 `distillation` 指标。该冒烟进程已停止并释放 GPU；当前
结论是“GPU 隔离修复有效，但 v0.8 standalone 权重同步/RPC 仍需继续修复”。
## R34: OPD 最小闭环验证通过

在学生 FSDP 单卡、独立 rollout 单卡、教师单卡的短上下文配置下，OPD
已完成完整的 1-step smoke：

- `Training Progress: 1/1`，单步约 46.5 秒；
- `actor/distillation/loss=0.6440724`；
- `student_mass=0.9986036`，`teacher_mass=0.9996425`；
- `overlap_ratio=0.6970825`；
- rollout 写入 `outputs/vanilla_opd_smoke/rollouts/1.jsonl`，TensorBoard
  事件文件也已生成。

本次闭环使用了三个兼容性修复：standalone Ray resource pool、独立服务
rank/注册等待、以及无 flash-attn 的纯 PyTorch NestedTensor fallback。
现有 GRPO 版本仍未修改。多卡学生 FSDP 需要单独处理分片权重向单卡
standalone rollout 的汇聚问题，不能直接把本次单卡 smoke 参数用于正式训练。

## R35: OPD 长训练已启动

正式 100-step 训练已在云端启动，任务名为 `vanilla_opd_100step`。当前采用
学生 FSDP 1 卡、学生 standalone rollout 1 卡、教师 1 卡的稳定拓扑，第四张
GPU 保持空闲。

首个正式 step 已完成：

- `Training Progress: 1/100`；
- `actor/distillation/loss=0.3446288`；
- `student_mass=0.9994112`，`teacher_mass=0.9993019`；
- `overlap_ratio=0.7310077`；
- 单步耗时约 74 秒，训练进程仍在运行。

长训日志：`/tmp/vanilla_opd_100step.log`；TensorBoard：
`tensorboard/vanilla_opd_100step`；输出和 checkpoint：
`outputs/vanilla_opd_100step` 及对应的
`checkpoints/agopd-rl/vanilla_opd_100step`。

## R39: 多预算评估结果

在固定验证集前 256 条、统一采样协议下完成了 Base 与 OPD step100 的
2048/4096/8192 budget 扫描。表中 accuracy 为 semantic accuracy：

| model | budget | semantic accuracy | truncation | response mean |
| --- | ---: | ---: | ---: | ---: |
| Base | 2048 | 23.83% | 20.70% | 1275.6 |
| Base | 4096 | 26.17% | 5.86% | 1507.0 |
| Base | 8192 | 26.56% | 3.91% | 1689.6 |
| OPD100 | 2048 | 19.14% | 34.38% | 1490.1 |
| OPD100 | 4096 | 21.48% | 5.86% | 1809.0 |
| OPD100 | 8192 | 21.48% | 1.95% | 1919.7 |

增加预算显著降低了 OPD 截断率，但 OPD 在三档预算下仍低于 Base：差距约为
`-4.69/-4.69/-5.08` 个百分点。因此长度预算能解释一部分 2048 结果，不能
单独解释全部差距；仍需 Teacher 对照和 EOS/位置分析。该实验是 256 条诊断子集，
最终结论仍以固定 1024 条全量评估为准。

## R36: OPD 100-step 结果

100-step 训练已正常完成，进程退出且 GPU 已释放。代表性指标如下：

| step | distillation loss | overlap ratio | entropy | response mean |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3446 | 0.7310 | 0.1124 | 1293.0 |
| 20 | 0.1473 | 0.7410 | 0.2652 | 1291.8 |
| 40 | 0.0835 | 0.7327 | 0.3109 | 1446.4 |
| 60 | 0.0916 | 0.7459 | 0.3771 | 1854.6 |
| 80 | 0.0893 | 0.7423 | 0.3229 | 1620.9 |
| 100 | 0.0715 | 0.7270 | 0.2762 | 1073.5 |

loss 相比 step1 下降约 79.3%，说明学生分布已经明显向教师 top-k 分布靠拢；
overlap ratio 基本维持在 0.73--0.75，尚不能据此断言数学正确率提升。正式
checkpoint 位于 `checkpoints/agopd-rl/vanilla_opd_100step/global_step_100`，
后续需要将其导出为可评估的 Hugging Face 格式，再用固定 `dapo-verl-v1/val.parquet`
做准确率和输出长度评估。

## R37: OPD step100 固定集评估

OPD step100 已导出并在固定 1024 条 `dapo-verl-v1/val.parquet` 上完成 no-think
评估，采样协议为 `temperature=0.6`、`top_p=0.95`、`top_k=20`、最大 2048 tokens。

| model | accuracy | truncation | response mean | answer marker |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen3-1.7B | 23.34% | 19.92% | 1261.7 | 25.49% |
| OPD step100 | 18.65% | 34.28% | 1494.7 | 27.05% |

OPD 比基座下降 `4.69` 个百分点，且截断率明显升高。虽然训练中的
`distillation/loss` 从 `0.3446` 降至 `0.0715`，但这没有转化为数学任务正确率；
当前 step100 checkpoint 不应直接作为正式模型。下一轮应优先检查 response
截断、teacher top-k 目标与任务 reward 的耦合，并以固定集准确率作为 early-stop
指标，而不是只看蒸馏 loss。

## R38: OPD 输出错误诊断

对 OPD step100 的 1024 条验证输出进一步检查后发现：351 条输出达到 2048
token 上限，这些样本准确率为 0%；其余未截断样本准确率为 28.38%。有
`Answer:` 标记的样本准确率为 33.94%，没有标记的样本为 12.99%；有
`\\boxed{}` 的样本准确率为 32.37%，没有标记的样本为 7.99%。

典型错误包括：推理中出现正确中间值，但最终答案重新拼接成错误表达式；
或者长推理在达到上限时没有输出最终答案。因此下一轮应先限制无效冗长推理、
强制稳定的最终答案格式，并把截断率作为核心评估指标。

## R40: Base/Teacher/OPD 三方预算对照

固定验证集前 256 条的三方预算实验已经完成：

| model | budget | semantic accuracy | truncation | response mean |
| --- | ---: | ---: | ---: | ---: |
| Base | 2048 | 23.83% | 20.70% | 1275.6 |
| Base | 4096 | 26.17% | 5.86% | 1507.0 |
| Base | 8192 | 26.56% | 3.91% | 1689.6 |
| Teacher | 2048 | 34.38% | 21.48% | 1315.0 |
| Teacher | 4096 | 38.28% | 7.81% | 1578.9 |
| Teacher | 8192 | 39.45% | 3.91% | 1793.9 |
| OPD100 | 2048 | 19.14% | 34.38% | 1490.1 |
| OPD100 | 4096 | 21.48% | 5.86% | 1809.0 |
| OPD100 | 8192 | 21.48% | 1.95% | 1919.7 |

Teacher 确实比 Base 强，并且更高预算能继续提升；OPD100 即使在 4096/8192
预算下仍低于 Base。因此 2048 截断只能解释部分下降，当前 Vanilla OPD 还存在
真实的目标迁移损伤、数据/协议错配或训练稳定性问题。下一步优先做 EOS 与位置
loss 机制分析，再设计 Advantage-Gated OPD；暂不继续盲训 Vanilla OPD。

## R41: EOS 与位置 KL 机制分析

在三方 2048 输出上完成了 EOS 终止概率分析（32 条样本）：

| model | endpoint EOS probability |
| --- | ---: |
| Base | 0.7604 |
| Teacher | 0.8204 |
| OPD100 | 0.7022 |

OPD 在自身生成终点的 EOS 概率低于 Base 和 Teacher，支持“更不愿结束”
的 termination drift 假设；由于三者终点状态不同，该结果仍属于机制线索，
不是严格的因果证明。

在同一批 OPD rollout 的 16 条样本上计算 Teacher→Student/Base 的分段 KL：

| position | Teacher→OPD | Teacher→Base |
| --- | ---: | ---: |
| early 0--25% | 0.1705 | 0.2279 |
| mid 25--50% | 0.1402 | 0.1812 |
| late 50--75% | 0.1079 | 0.1339 |
| final 75--100% | 0.1038 | 0.1623 |

OPD 在所有位置段都比 Base 更接近 Teacher，尤其是最后 25%。这证明
Vanilla OPD 的概率对齐确实发生了，但也说明“更像 Teacher”本身没有保证
任务正确率提升；下一轮应加入 task-outcome gate 和长度/EOS 约束，而不是继续
增大 Vanilla OPD 的训练步数。

## R43: Same-state EOS 复核

为消除各模型访问状态不同带来的混淆，固定同一批 OPD rollout prefix，分别
计算 Base、Teacher、OPD 在相同状态上的 EOS 概率（32 条样本）：

| position | Base | Teacher | OPD100 |
| --- | ---: | ---: | ---: |
| 25% | 0.0000 | 0.0000 | 0.0000 |
| 50% | 0.0000 | 0.0000 | 0.0000 |
| 75% | 0.0000 | 0.0000 | 0.0000 |
| 90% | 0.0000 | 0.0000 | 0.0000 |
| 100% | 0.7093 | 0.6863 | 0.7022 |

同一状态下三者终点 EOS 概率接近，之前独立轨迹终点分析中的
`Base=0.7604/Teacher=0.8204/OPD=0.7022` 主要由不同模型访问的状态不同造成，
不能据此认定 OPD 迁移了错误 EOS policy。当前主问题仍是：OPD 在 Teacher
分布上实现了更强的 token-level alignment，但在 Base 正确、Teacher 错误的
题目上产生了较高的 negative transfer。

## R42: 四象限迁移与重复退化分析（初版，已被 R44 修正）

在同一批固定 256 条 2048-budget 输出上，按 Base、Teacher、OPD 的 semantic
correctness 做四象限分析：

| category | Base | Teacher | samples | OPD correct | OPD rate |
| --- | --- | --- | ---: | ---: | ---: |
| A | correct | correct | 41 | 24 | 58.5% |
| B | wrong | correct | 47 | 11 | 23.4% |
| C | correct | wrong | 20 | 7 | 35.0% |
| D | wrong | wrong | 148 | 7 | 4.7% |

因此 Positive Transfer Rate 为 `PTR=11/47=23.4%`，Negative Transfer Rate
为 `NTR=13/20=65.0%`，净迁移为 `-41.6` 个百分点。Vanilla OPD 只转移了
少量 Teacher-only 能力，却破坏了大量 Base 原本正确的题目；这比总 accuracy
下降更直接地说明了 unconditional teacher alignment 的问题。

对同一批输出做 4-gram 和重复行统计：Base 的平均 4-gram 重复率为 38.16%，
Teacher 为 37.35%，OPD100 为 37.13%；重复行率分别为 34.22%、33.81%、
32.35%。目前没有证据表明 OPD 的整体重复率显著升高，但 OPD 的最大重复
4-gram 次数均值略高（20.82 vs Base 18.97），仍需对具体失败样本做人工抽查。

下一步优先实现 same-state EOS：固定同一条 OPD prefix，同时计算 Base、
Teacher、OPD 的 EOS 概率，消除当前 endpoint EOS 分析中的 state confounder。

## R44: 评估解析器修正与 P0 结论冻结

在抽查 B/C 象限真实样本时发现，部分输出使用了
`Answer: $ \\boxed{4} $` 这种合法答案格式，但 semantic evaluator 没有先解包
`\\boxed{...}`，导致“Base 实际答对”被计为答错。这个问题不会改变训练 reward
函数，只修正离线评估路径；已在 `src/agopd/reward/math_reward.py` 增加解包逻辑，
并新增回归测试，`pytest -q tests/test_math_reward.py` 为 9 passed。所有相关预算
输出已用修正后的 evaluator 重新计算，旧版四象限数字只保留作审计记录。

修正后的 256 条、2048 budget 结果：

| category | samples | OPD correct | OPD rate |
| --- | ---: | ---: | ---: |
| A: Base correct / Teacher correct | 46 | 24 | 52.2% |
| B: Base wrong / Teacher correct | 42 | 8 | 19.0% |
| C: Base correct / Teacher wrong | 23 | 7 | 30.4% |
| D: Base wrong / Teacher wrong | 145 | 7 | 4.8% |

因此 `PTR=8/42=19.0%`，`NTR=16/23=69.6%`，`PTR-NTR=-50.5pp`。
这支持“无条件 OPD 的破坏性迁移强于正向迁移”，但不支持把问题简单归因于
整体 repetition：修正后 Base/Teacher/OPD 的平均 4-gram 重复率为
38.16%/37.35%/37.13%，重复行率为 34.22%/33.81%/32.35%。OPD 的最大重复
4-gram 次数均值略高（20.82 vs Base 18.97），因此仍作为次要诊断指标。

同一批 OPD trajectory 上的探索性位置 KL 与 correctness 相关系数为
`-0.231`（仅 16 条，不能当作最终统计结论），方向上提示“更贴近 Teacher”
并没有带来更高正确率，甚至可能相反；后续 gate 实验要把它作为待验证假设。

P0 结论冻结为：

1. Teacher 在多预算下明显强于 Base，Vanilla OPD 的概率对齐目标也确实优化成功。
2. OPD100 在 2048/4096/8192 下都落后 Base，不能用截断单独解释。
3. Same-state EOS 没有显示稳定的 policy-level EOS 偏移，暂不加入硬 EOS penalty。
4. 下一步训练顺序为：已有 GRPO baseline → GRPO+Vanilla OPD → Outcome Gate →
   Advantage Gate → Teacher Competence Gate；长度、EOS、repetition 先监控，不作为
   第一轮额外优化目标。

## R46: 评估协议第二次修正（最终冻结）

继续抽查发现 Teacher 的答案末行还可能写成 `**Answer: 91**`。因此仅支持裸
`Answer:` 仍会把正确轨迹标成 invalid。评估器现已同时支持可选的 Markdown 加粗
和 `Final Answer:` 前缀，新增测试后 `pytest -q tests/test_math_reward.py` 为
10 passed。此前 R42/R44 的四象限和预算数字均作废，下面是当前冻结版本。

在固定 256 条、2048 budget 输出上：

| model | semantic accuracy | parse rate | truncation |
| --- | ---: | ---: | ---: |
| Base | 28.91% | 81.64% | 20.70% |
| Teacher | 43.75% | 78.52% | 21.48% |
| OPD100 | 22.66% | 67.58% | 34.38% |

四象限为 A=61、B=51、C=13、D=131；OPD 在 B 中答对 10 条，
在 C 中保留 2 条。因此 `PTR=10/51=19.6%`，`NTR=11/13=84.6%`，
`PTR-NTR=-65.0pp`。这个结论比早期版本更强，但它依赖当前明确冻结的
semantic answer contract；之后所有模型比较必须使用同一个 evaluator 版本。

实际抽查样本也符合这个方向：

- B 类棕 brownie 题，gold=60：Base 输出 `Answer: 25`，Teacher 输出
  `Answer: 60`，OPD 输出 `Answer: 48`，说明 Teacher-only 能力没有被 Vanilla
  OPD 稳定迁移。
- C 类双正六边形题，gold=3：Base 输出 `Answer: $ \\boxed{3} $`，Teacher
  输出 `Answer: 1.5`，OPD 输出 `Answer: 4.5`，说明在 Teacher 错误时，
  无条件模仿会覆盖 Student 原本正确的答案。

因此下一轮仍按 selective intervention 推进，但所有收益判定均以冻结后的
semantic accuracy、parse rate、truncation、PTR/NTR 为准。

## R45: 启动 GRPO + Vanilla OPD 基线

P0 完成后启动下一项训练，脚本为 `scripts/run_grpo_vanilla_opd.sh`，云端实验名为
`grpo_vanilla_opd_50step`。该实验使用 Base Qwen3-1.7B 初始化，DAPO v1 train/val
划分和同一 math verifier，配置 `rollout.n=2`、`train_batch_size=16`，确保 GRPO
具有组内相对优势；损失为任务 GRPO loss 加原始 `forward_kl_topk` OPD loss，
`use_task_rewards=True`、`use_policy_gradient=False`、`distillation_loss_coef=1.0`。

为保持资源隔离，仍使用学生 1 GPU、rollout 1 GPU、Teacher 1 GPU，第四张卡空闲。
输出写入 `outputs/grpo_vanilla_opd_50step`，checkpoint 写入
`checkpoints/agopd-rl/grpo_vanilla_opd_50step`，不会覆盖 Vanilla OPD100 或已有
GRPO 结果。启动检查已通过：Hydra 配置检查通过、训练总步数 50、Teacher 和 Student
两套 vLLM 均完成加载，训练进度已进入 `0/50` 的 rollout 阶段；后续重点观察
`critic/rewards/mean`、`actor/pg_loss`、`distillation/loss`、长度和 TensorBoard
中间 step 曲线。

## R47: GRPO + Vanilla OPD 首个有效 step

云端日志已进入 `training/global_step=3`，说明任务奖励和 OPD loss 都实际参与了
actor 更新，当前启动阶段没有 batch、Ray、vLLM 注册或显存错误。首个有效日志为：

| metric | value |
| --- | ---: |
| `critic/rewards/mean` | 0.2500 |
| `actor/pg_loss` | 0.0421 |
| `distillation/loss` | 0.1875 |
| `actor/distillation/overlap_ratio` | 0.7399 |
| `response_length/mean` | 1225.5 |
| `response_length/clip_ratio` | 0.125 |
| `timing_s/step` | 68.1s |

当前实验继续后台运行到 50 steps。这个结果暂时只证明训练链路和联合目标生效，
不代表最终准确率提升；待完成后必须用冻结的 semantic evaluator 在同一 1024 条
验证集上比较 Base、GRPO、Vanilla OPD 和 GRPO+Vanilla OPD，并抽查实际样本。

## R48: GRPO + Vanilla OPD 训练完成

`grpo_vanilla_opd_50step` 已完成 `50/50`，最终 checkpoint 为
`checkpoints/agopd-rl/grpo_vanilla_opd_50step/global_step_50`，50 份 rollout 和
TensorBoard 日志均已落盘，训练进程正常退出。最终 step 的在线训练指标为：

| metric | value |
| --- | ---: |
| `critic/rewards/mean` | 0.3750 |
| `actor/pg_loss` | 0.00495 |
| `actor/kl_loss` | 0.0570 |
| `distillation/loss` | 0.08945 |
| `actor/distillation/overlap_ratio` | 0.7346 |
| `response_length/mean` | 1228.0 |
| `response_length/clip_ratio` | 0.21875 |
| `timing_s/step` | 134.2s |

训练期间 OPD loss 从首个有效 step 的约 0.1875 降到 0.0895，但在线 reward
不是验证集准确率，不能据此宣布收益。合并后的 HF 模型已生成于
`models/Qwen3-1.7B-grpo-vanilla-opd-50step`，统一 1024 条验证集评估已启动，
结果写入 `reports/eval-dapo-v1/grpo-vanilla-opd-50step.jsonl`。

## R49: GRPO + Vanilla OPD 验证结果

50-step checkpoint 的 1024 条固定验证集评估已经完成：

| metric | value |
| --- | ---: |
| strict reward accuracy | 20.21% (207/1024) |
| semantic accuracy | 24.02% |
| strict contract accuracy | 15.53% |
| answer parse rate | 62.21% |
| truncation rate | 38.96% |
| response length mean | 1537.0 |

与已有 strict reward 结果比较：Base 为 23.34%，OPD100 为 18.65%。因此联合
目标让 Vanilla OPD 的下降得到部分恢复，但 50 steps 后仍未超过 Base；不能把
训练期 reward mean=0.375 误认为验证集收益。为消除 semantic evaluator 版本差异，
已另行启动 Base 的同版本 1024 条重测。

实际抽查到的轨迹体现了双向变化：坐标变换题上 GRPO+OPD 输出正确 `7`，而
OPD100 输出错误 `1`；但无重复数字最大倍数题上 GRPO+OPD 出现大量重复
`Try 9876542016` 且未完成答案，OPD100 输出正确 `120`。因此下一轮 Gate 仍须
同时测 task accuracy、NTR/PTR、长度和 repetition，不能只看 overlap 或平均 reward。

## R50: 模型规模升级决策

基于 1.7B 的机制结果，模型规模列为下一阶段高优先级变量，但不把“1.7B 太小”
直接定为唯一原因。当前证据同时指向两件事：Vanilla OPD 已实现明显的 Teacher
分布对齐却造成负迁移；而 GRPO/能力提升的训练预算仍只有几十步，远小于真正的
capability-scale RL。因此更大的 Student 用于提高方法可分辨性，不能替代足够的
有效 rollout budget。

下一阶段主配置冻结为同系列：

| role | model |
| --- | --- |
| Student | Qwen3-4B |
| Teacher | Qwen3-8B |

理由是 4B 比 1.7B 更可能完整执行 Teacher 的 reasoning branch，8B/4B 的二倍
容量差又比 14B/1.7B 更容易解释。Qwen3-4B 已在云端，Qwen3-8B 当前尚未下载。
1.7B 的 Base/OPD/GRPO+OPD 结果保留为 debug-scale 和机制对照，不删除、不覆盖。

规模实验只改变 model size，以下变量保持不变：DAPO v1 train/val、冻结 semantic
evaluator、prompt 格式、2048 response budget、sampling 参数、`n=2`、`batch=16`、
学习率和资源隔离。先跑三条对照：Base 评估、Pure GRPO、GRPO+Vanilla OPD；确认
4B Student 进入健康区间后，再加入 Outcome Gate 和 Advantage Gate。每个版本先做
50--100 step 诊断以比较方法差异，再决定是否扩展到数百 effective steps；不把
50 step 结果包装成最终能力结论。

本轮规模假设为：若 4B 的 Vanilla OPD 不再出现明显负迁移，说明 1.7B 存在
capacity ceiling；若 4B 仍然出现 Teacher 对齐上升、任务准确率下降，则更强地
支持“Vanilla OPD objective mismatch”，Selective OPD 的必要性反而更明确。

## R51: 云端与本地并行资源分工

本地 Conda 环境 `${HOME}/.conda/envs/agopd-vllm-281` 已验证可用，拥有两张
RTX 4090、Torch 2.8.0+cu128、vLLM 0.10.2、verl 0.6.1，作为快速诊断资源保留。
资源分工冻结为：

| resource | primary work | reason |
| --- | --- | --- |
| local 2×RTX 4090 | 1.7B Pure GRPO、Outcome/Advantage Gate、256 条 smoke/eval | 快速迭代机制和实现 |
| cloud 4×A100 | 4B Student + 8B Teacher scale ablation、完整 1024 eval、长训练 | 更大模型和更长有效 rollout budget |

本地与云端使用独立 experiment name、rollout、checkpoint 和 TensorBoard 目录，
避免实验相互覆盖。两边仍共用固定 DAPO v1 数据划分、semantic evaluator 和
指标定义；本地的 1.7B 结果用于 debug-scale 对照，不替代云端 4B 主实验。

本地第一轮并行任务已启动：`local_grpo_1p7b_50step`，使用 Base Qwen3-1.7B、
Pure GRPO、`n=2`、`batch=16`、50 steps。配置检查通过，TaskRunner 已进入数据
加载和 worker 初始化；由于本地 Ray 会把 TaskRunner 输出写入
`/tmp/ray/session_*/logs/worker-*.out`，不能只看外层 nohup 日志判断任务状态。
本地结果写入 `outputs/local_grpo_1p7b_50step` 和
`tensorboard/local_grpo_1p7b_50step`，不会覆盖云端实验。

## R53: 本地 GRPO 首次退出原因与可观测性修复

首次本地 GRPO 没有进入 step：4090 共置 rollout 使用 `gpu_memory_utilization=0.35`，
vLLM 报 `No available memory for the cache blocks` 后 worker 退出，所以 GPU 随后
回到接近空闲。已将该参数改为本地专用默认 `0.50`，云端脚本仍保持原默认值；下一次
启动前需要确认 vLLM 成功建立 KV cache。

verl 已继续写 TensorBoard event。SwanLab 不替代 TensorBoard，而是把本地 event
转换成可浏览的离线实验：训练期间保留 TensorBoard，训练后执行
`swanlab convert -t tensorboard --tb_logdir tensorboard/<run>`，再用
`swanlab watch --logdir <converted-logdir>` 查看曲线。这样训练指标仍来自 verl 原生
logger，SwanLab 只承担聚合和本地 dashboard 展示，避免修改训练主链路。

本地 runner 已增加独立 `logs/<experiment>.log`，后续不再依赖 Ray 外层 stdout；
Ray worker 详细日志仍位于 `/tmp/ray/session_*/logs/`。SwanLab 0.9.8 已安装到
`agopd-vllm-281` 环境，后续先修复 KV cache 后重启本地 GRPO。

本地 GRPO 重启后已通过 KV cache 初始化并进入训练：`nvidia-smi` 显示两张 4090
分别约 17--18GB 显存占用，TaskRunner 已完成至少 step 2。后续本地 runner 的
rollout 显存配额固定为 0.50，云端稳定脚本默认值不变。

由于 SwanLab 0.9.8 的 CLI TensorBoard converter 会强制执行登录检查，当前无 API
key 的本地环境改用 `scripts/tensorboard_to_swanlab.py`：它直接读取 TensorBoard
scalar events，以 SwanLab offline run 写入 `swanlog/<run>`，支持 `--follow` 持续
同步；当前无 API key 时使用 `mode=local` 生成 `runs.swanlab` 数据库，再用
`swanlab watch <swanlog-dir>` 打开本地 dashboard。TensorBoard 6006
负责实时原生曲线，SwanLab 5092 负责离线聚合，两者数据来源一致。

当前看板已实际启动：TensorBoard PID 为 `535382`，地址为 `http://127.0.0.1:6006`；
SwanLab bridge 正在把 `tensorboard/local_grpo_1p7b_50step` 同步到
`swanlog/local_grpo_1p7b_50step_live_local`，SwanLab watch 运行在
`http://127.0.0.1:5092`。bridge 已完成第 1--2 step 的 event 同步，数据库文件
`runs.swanlab` 正在增长。

SwanLab dashboard extra 已补装（`swanboard==0.1.10b2`）。安装过程中 pip check
显示了环境内原有的若干非本项目冲突（如 opencv/numpy、datasets/fsspec、scikit-image
可选依赖），但 Torch 2.8.0+cu128、vLLM 0.10.2、TensorBoard 2.21.0 和 SwanLab
0.9.8 均可正常 import；不在本轮修改这些无关依赖。

## R52: 本地 Qwen3-8B 权重下载

本地项目盘空间充足，已创建 `models/Qwen3-8B` 并取得模型元数据和 5 个 safetensors
分片索引。Hugging Face 原始 CDN 连接不稳定，自动下载器在元数据阶段无进展；已
停止该通道，改用模型文件镜像进行断点续传。下载只写入 `models/Qwen3-8B`，不会
影响现有 Qwen3-1.7B、Qwen3-4B 或正在运行的本地 GRPO。

完成后必须检查 5 个分片、`model.safetensors.index.json`、`config.json` 和 tokenizer
文件，并用 `AutoConfig`/单卡 vLLM 做加载 smoke test；在完整性验证之前不启动
4B/8B 训练。

Qwen3-8B 已下载完成：5 个 safetensors 分片和索引文件齐全，总目录约 16G；
`AutoConfig` 识别为 Qwen3（36 layers、hidden size 4096），tokenizer 加载成功，
5 个分片均通过 `safe_open` 读取验证。下一步可在云端使用 Qwen3-4B Student +
Qwen3-8B Teacher 做 scale ablation；本地 8B 权重只用于验证/小规模推理，不在
24GB 4090 上直接承担 8B full-parameter 训练。

## R54: 云端低 GPU 利用率诊断与性能 smoke

云端当前空闲是因为上一轮训练和评估均已结束；上一轮联合实验期间，学生 actor
只占 1 GPU，第四张 A100 空闲，rollout/Teacher 的 vLLM 仅申请 0.25 显存，且
`enforce_eager=True` 禁用了 CUDA Graph。step50 的耗时为
`gen=31.7s/ref=12.4s/update_actor=17.5s`，checkpoint 保存还额外耗时 65.7s。

云端依赖复核通过：Torch 2.8.0+cu128、vLLM 0.10.2、FlashInfer 0.3.0。已新增
`scripts/run_cloud_perf_smoke.sh`，默认用学生 2 GPU、rollout 1 GPU、Teacher 1 GPU，
`n=4`、batch=32、vLLM 显存配额 0.50、max batched tokens=16384，并测试关闭 eager。
该脚本与稳定版参数隔离，只跑 3 steps；只有它通过后才用于 4B/8B 主实验。

首次全量性能组合（学生 2 GPU、`n=4`、eager=False）完成了四个服务的模型加载和
CUDA Graph capture，但约 9 分钟仍未进入首个训练 step，已主动停止。该结果说明
CUDA Graph 在 vLLM 层可用，却不能证明当前 standalone 多 GPU 组合端到端可用，
因此不将它直接用于主训练。随后启动了稳定拓扑 smoke：学生 1 GPU、`n=2`，只将
两个 vLLM 的显存配额提高到 0.50、max batched tokens 提高到 16384，并保持
`enforce_eager=True`；待首个 step 后比较实际 gen/ref/update/step 时间。

稳定拓扑 smoke 已完成 3/3：step1/2/3 的总耗时分别为 97.0s、74.4s、138.8s，
其中 step3 包含 71.9s checkpoint 保存；排除保存后，常规耗时约 66.9--74.4s，
与旧配置的 73--76s 基本一致。`gen` 为 35.7/30.5s，`ref` 为 12.6/12.3s，
`update_actor` 为 19.0/17.3s，说明把 vLLM 显存从 0.25 提到 0.50 没有改善
端到端吞吐。最终主实验保持稳定拓扑和 eager 模式，显存配额只在实际出现 KV
cache 不足时提高，不把 A100 空余显存误当作必须填满的目标。

全卡 eager smoke（学生 2 GPU、rollout/Teacher 各 1 GPU）未能在本轮进入有效
step，已停止；此前关闭 eager 的版本还额外经历了较长 CUDA Graph capture。因此
当前 v0.8 standalone 资源池不默认切换学生 2 GPU。后续若要使用第四张卡，优先
测试 Teacher replica 或独立 rollout replica，而不是继续叠加未经验证的 FSDP、
`n` 和 CUDA Graph 参数。

## R55: 云端与本地速度对照

本地 `local_grpo_1p7b_50step` 已完成 50/50，云端稳定性能 smoke 也已完成 3/3。
在相同的 1.7B Student、batch=16、n=2、2048 response budget 附近比较：

| runtime | workload | steady step | generation | actor update |
| --- | --- | ---: | ---: | ---: |
| local 2×RTX 4090 | Pure GRPO | 约 76--81s | 约 36--42s | 约 25s |
| cloud A100 | GRPO+Vanilla OPD | 约 74--76s | 约 30--36s | 约 17--19s |

云端按 wall-clock 略快，并且 generation/ref/update 子阶段更快；但云端包含
Teacher OPD、而本地是 Pure GRPO，不能把这个数字当作严格 benchmark。结论是：
本地 4090 足够承担快速 gate/debug，云端 A100 仍应承担 4B/8B 主实验和长训练。
低 GPU 利用率问题主要来自阶段串行和 standalone 资源池，而不是单纯显存不足。

## R56: 云端 4B/8B scale ablation 启动

云端 Qwen3-8B 已完成复核：目录约 16G，5 个权重分片、索引和 tokenizer 文件
齐全；单卡 HF 加载并生成 64 tokens 成功。scale ablation 的基线评估已启动：
GPU0 运行 Qwen3-4B Base，GPU1 运行 Qwen3-8B Teacher，二者共用 1024 条验证集、
2048 response budget、`temperature=0.6/top_p=0.95/top_k=20/seed=42`。结果将分别
写入 `reports/eval-dapo-v1/base-qwen3-4b.jsonl` 和
`reports/eval-dapo-v1/teacher-qwen3-8b.jsonl`；完成后再启动 4B GRPO 与
4B+8B Vanilla OPD 训练。

## R57: 云端 4B/8B 基线评估结果

同一 1024 条验证集、2048 response budget、`temperature=0.6/top_p=0.95/top_k=20/seed=42`
下，4B 和 8B 基线结果为：

| model | strict reward accuracy | semantic accuracy | parse rate | truncation | mean length |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B Base | 33.01% | 41.02% | 75.20% | 25.20% | 1325.5 |
| Qwen3-8B | 36.43% | 38.48% | 73.63% | 27.44% | 1376.5 |

8B 在 strict verifier 指标上更高，但 semantic accuracy 低于 4B；这说明模型规模
提升并不自动等价于当前数据协议下的更强可用 Teacher，后续 8B supervision 必须
经过 prompt-level competence gate，不能仅依据参数量开启全量 OPD。

在基线已落盘后，下一项启动 Qwen3-4B Base 的 Pure GRPO 50-step，使用同一 DAPO
v1 数据和 `n=2/batch=16/lr=1e-6`，结果独立写入
`models/Qwen3-4B-grpo-50step` 对应的 checkpoint 目录，作为 4B Student 的 RL 对照。

## R58: 4B Pure GRPO 正确启动

在停止旧版错误任务并上传参数化脚本后，`scale_grpo_4b_50step` 已按预期启动：
`trainer.n_gpus_per_node=4`、`train_batch_size=16`、`rollout.n=2`、总步数 50，
并使用独立的 `outputs/scale_grpo_4b_50step` 和
`checkpoints/agopd-rl/scale_grpo_4b_50step`。当前处于数据过滤和模型初始化阶段，
首个训练 step 尚未产生；下一次监督重点是确认 4B actor/vLLM 完成加载并记录首个
reward、长度、generation 和 update_actor 时间。

## R59: 4B Pure GRPO 首个有效阶段

后续监督确认任务已进入 `training/global_step=3`，四张 A100 均有计算任务，显存
约 39GB/卡，GPU 利用率约 42%--57%。step1/2 的 reward mean 为 0.40625、0.21875，
单步耗时约 72--77s；当前资源利用率明显优于 1.7B 单卡 Student 拓扑。训练继续
后台运行至 50 steps，完成后进行 checkpoint 合并和统一验证。

最新监督显示本地 `local_grpo_1p7b_50step` 已完成 50/50，最终 step 耗时 75.9s，
reward mean 0.46875；本地 GPU 随任务结束而释放，但 TensorBoard/SwanLab 看板仍在
运行。云端 `scale_grpo_4b_50step` 已推进到 30/50，step28--30 耗时约
58.5/60.4/57.3s，四张 A100 显存约 39GB/卡、利用率约 42%--54%，没有异常。
4B 训练完成后进入合并和固定验证，再启动 4B+8B Vanilla OPD。

## R61: 本地 checkpoint 与多进程配置

本地下一轮 GRPO 已将 `SAVE_FREQ` 默认设为 10，确保每 10 steps 保存 checkpoint，
训练结束后可以直接做固定验证，不再只有在线 reward。多进程仍保持开启：Ray 调度
任务，2 个 GPU rank 执行 FSDP actor，vLLM 使用 spawn worker，数据加载默认 8 个
workers；本地可用 `CUDA_VISIBLE_DEVICES` 控制 GPU，不需要修改系统级 compute mode。

## R60: 本地 1.7B 与 1.7B/4B 组合对照

本地 `1.7B Pure GRPO` 已完成 50 steps。在线 rollout reward mean 的 50-step
平均约为 0.235，最后一步为 0.46875，后 10 steps 平均约 0.244；这只能说明
训练批次中的采样结果有波动，不能替代训练后固定验证。由于本轮 `save_freq=-1`，
没有可评估 checkpoint，下一轮本地 GRPO 必须按 `save_freq=10` 保存模型。

当前已有的 `Student=1.7B / Teacher=4B` 组合结果（固定 1024 validation，2048
budget）是：Base strict accuracy 23.34%，Vanilla OPD100 strict accuracy 18.65%
（semantic 23.14%），GRPO+Vanilla OPD strict accuracy 20.21%、semantic accuracy
24.02%。因此 GRPO+OPD 相比纯 OPD 有恢复，但仍没有证明超过 Base；本地 1.7B
Pure GRPO 的在线 reward 不能直接与这些验证集数字比较。

云端当前 4B Pure GRPO 正在推进，完成后会得到真正的 `Student=4B` RL 对照；
之后再运行 `Student=4B / Teacher=8B` Vanilla OPD，才能回答规模升级是否降低
负迁移，以及 8B Teacher 的额外能力是否值得迁移。

## R62: checkpoint 版本地重跑与云端进度

独立的 `local_grpo_1p7b_50step_ckpt` 已启动，实际配置确认包含
`save_freq=10`、`dataloader_num_workers=8`、2 GPU FSDP rank 和 vLLM spawn 多进程。
云端 `scale_grpo_4b_50step` 已推进到 48/50，最近 step 耗时约 61s，显存约
45GB/卡，四张 A100 均在工作。云端完成后先合并并评估，再启动 4B+8B Vanilla OPD。

## R63: 发现云端 4B 训练缺少 checkpoint

复核 `scale_grpo_4b_50step` 时发现该任务虽然日志到达 `50/50`，但原命令沿用了
`trainer.save_freq=-1`，云端不存在对应 checkpoint，因此不能合并或做训练后模型
评估。原始 50-step rollout 和日志保留作在线指标记录，不把它冒充成可评估模型。

当前任务已改为新的 `scale_grpo_4b_50step_ckpt` 重跑，使用 `SAVE_FREQ=10`，输出、
checkpoint 和 TensorBoard 全部使用新目录；本地 `local_grpo_1p7b_50step_ckpt` 也
同步采用每 10 steps 保存。只有新 4B checkpoint 合并并通过固定验证后，才启动
4B Student + 8B Teacher OPD。

## R64: Rollout 数量与有效训练预算

当前云端/本地 Pure GRPO 使用 `rollout.n=2`，即每个 prompt 生成两条回答。这个值
适合快速 smoke，但对少 step 的 GRPO 方法差异比较保守。增加到 `n=4` 会让每个
prompt 有更多组内样本，通常改善 advantage 稳定性和探索覆盖；代价是每个 optimizer
step 的 response 数和 rollout token 约增加一倍，显存压力、generation 时间和总计算
预算都会增加。由于 vLLM 会批量并行，墙钟时间通常不会严格翻倍，但不能假设没有
额外时间。

当前 4B checkpoint 重跑保持 `n=2`，用于和已有结果严格对照。Pure GRPO 脚本的
`ROLLOUT_N` 已参数化，下一轮建议先做 `n=4` 的 3--5 step smoke：
`train_batch_size=16` 时每 step 采样 64 条 response；若显存或 step 时间过高，
改用 `train_batch_size=8` 保持每 step 32 条 response，再比较 group size 带来的
收益。通过 smoke 后才把 n=4 用于主实验，不直接跳到 n=8。

## R65: 云端 checkpoint 强制 override 重跑

监督发现第二次 4B 重跑的环境变量未传入最终 Hydra 配置，日志仍显示
`save_freq=-1`，因此该轮没有 checkpoint。已保留全部日志和 rollout，并在新的
`scale_grpo_4b_50step_ckpt2` 命令末尾显式加入 `trainer.save_freq=10`；启动日志
已确认 Hydra 实际值为 `save_freq: 10`，同时重新挂载 watcher `PID=877459`，
训练完成后自动执行合并和固定验证。

## R66: 本地 1.7B 评估与云端 4B 收尾

本地保存 checkpoint 的 1.7B Pure GRPO 已完成固定 1024 条验证：strict reward
accuracy `25.39%`（260/1024），semantic accuracy `27.44%`，截断率 `18.95%`，
平均响应长度 `1206.4`。相对 Base strict `23.34%`，本地 1.7B GRPO 提升约 2.05pp；
这是训练后模型评估，不再只是在线 reward。

云端 `scale_grpo_4b_50step_ckpt2` 已完成 50/50，4 个 model rank 和 4 个 optimizer
rank 均已保存。watcher 已检测到 checkpoint，当前正在自动执行 4B 合并，合并完成
后继续固定验证；该阶段完成后再进入 4B/8B OPD 启动确认。

## R67: 云端 4B Pure GRPO checkpoint 收尾

`scale_grpo_4b_50step_ckpt2` 已完成 50/50，且每 10 steps 保存了 FSDP actor
checkpoint。自动 watcher 使用旧的 v0.8 merger 配置时无法从 checkpoint 自动识别
模型类型，报出 `model_type=opt` 的识别错误；复核 checkpoint 和原始模型配置后确认
实际架构是 `Qwen3ForCausalLM`。随后使用 v0.8 merger 显式传入
`--hf_model_path ${AGOPD_ROOT}/models/Qwen3-4B`，成功合并到
`models/Qwen3-4B-grpo-50step-ckpt2`。这次 workaround 保留了原 checkpoint，不覆盖
之前的模型目录。

合并模型在固定 `dapo-verl-v1` 1024 条验证集上完成评估，配置为
`max_new_tokens=2048`、temperature `0.6`、top-p `0.95`、top-k `20`、seed `42`，
不启用 thinking：

| 指标 | 4B Base | 4B Pure GRPO step50 | 变化 |
| --- | ---: | ---: | ---: |
| strict reward accuracy | 33.01% | 40.23% (412/1024) | +7.23pp |
| semantic accuracy | 41.02% | 40.23% | -0.79pp |
| strict contract accuracy | 未记录 | 28.71% | -- |
| answer parse rate | 75.20% | 82.91% | +7.71pp |
| truncation rate | 25.20% | 17.09% | -8.11pp |
| mean response length | 1325.5 | 1171.9 | -153.6 |

结论是：在当前固定评估协议下，4B Pure GRPO 的严格 reward 指标取得了明确提升，
同时解析率和截断率也改善；但 semantic accuracy 没有同步提升，说明严格 reward
与语义能力仍需分开观察。云端 4B 训练/合并/评估阶段至此完成，下一阶段只在确认后
启动 `Student=Qwen3-4B + Teacher=Qwen3-8B` 的 Vanilla OPD，不自动越过该边界。

## R68: n=4 matched matrix smoke 与 OPD 拓扑阻塞

按照统一矩阵开始验证：两端都使用 `rollout.n=4`、`train_batch_size=8`、
`ppo_mini_batch_size=8`、2048 response cap、`0.6/0.95/20` sampling、学习率
`1e-6`、seed `42`，并使用独立输出目录。Cloud 4B E1 和本地 1.7B E1 都完成了
3 steps，产生非零 reward/advantage，并保存了 step3 checkpoint。实测 n=4 后，
本地 1.7B step1 约 86.5s，云端 4B step1/2 约 63.8/74.7s；之前的 60s 不能
直接外推到 OPD。

E2 纯 OPD smoke 暴露出两个资源边界：

1. 本地 2×RTX 4090 使用 verl v0.8 时需要独立的 Student/rollout、Teacher、
   rollout-server 三个 GPU 资源池，即使把 Teacher cache 调到 `0.80`，仍报
   `Total available GPUs 0 is less than total desired GPUs 1`。因此两卡不能完成
   当前 v0.8 的 1.7B+4B OPD 拓扑。
2. 云端 4×A100 的 `Student=2 + Teacher=2` 会挤掉额外 rollout pool；改成
   `Student=2 + Teacher=1 + rollout=1` 后资源池可以启动，Student 也不再出现
   单卡 optimizer OOM。但 vLLM 0.10.2 的 V1 多进程 worker 在主机
   `Exclusive_Process` compute mode 下报 `CUDA-capable device(s) is/are busy or
   unavailable`；V0 engine 兼容尝试也没有在 smoke 内进入 step。

因此 E1 smoke 已通过，但 E2/E3 和正式 60-step 矩阵没有启动，避免生成不可比较的
残缺结果。已有的工作环境、E1 checkpoint 和所有失败日志均保留；下一步需要平台
提供 Default/MPS compute mode，或增加本地第三张 GPU，再继续同条件 E2/E3。不能用
不同的资源拓扑、不同的模型初始化或不同的训练步数填补矩阵。

本轮结束时云端 GPU2/3 仍由两个在容器进程树中不可见的 CUDA context 占用约
29GiB/卡。容器内 root 无法通过 `kill` 清除它们，属于宿主机/平台级残留状态；在
平台重建容器或宿主机清理之前，不把这两张卡视为可用，也不启动下一轮 OPD。

## R71: 匹配 LoRA 矩阵 smoke 通过并进入正式 E1

为保持 E1/E2/E3 的训练条件一致，4B/8B 云端矩阵与本地 1.7B 控制统一采用
LoRA rank `8`、alpha `16`、`rollout.n=4`、batch `8`、2048 response cap、
`0.6/0.95/20` sampling 和学习率 `1e-6`。此前的 full-parameter E1 smoke 只
作为历史运行性记录，不作为与 LoRA OPD 的最终效果对照。

云端 4B/8B 的 E1 Pure GRPO、E2 OPD-only、E3 GRPO+OPD 均完成 3-step smoke；
E2/E3 都出现真实 `actor/distillation/loss`，权重同步成功且没有 CUDA、IPC、
HF 网络或 OOM 错误。本地 1.7B LoRA E1 smoke 也完成 3/3；本地 E2/E3 仍因
v0.8 需要三个独立 GPU 资源池而被两张 4090 阻断。

随后正式 E1 已在云端 4B 和本地 1.7B 并行启动，各跑 60 steps，每 10 steps 保存。
云端增加 `run_matrix_cloud_sequence.sh` watcher：E1 正常结束后自动清理 Ray，
再以完全相同预算启动 E2 和 E3，三者不会并发抢占 GPU。当前云端 E1 约
`1/60`、本地 E1 约 `2/60`，两边均已进入真实训练。

## R69: 澄清“独立 GPU”与 Exclusive_Process 的边界

此前的独立 GPU 修复解决的是 **Ray 资源池分配**：Student、Teacher 和 rollout
server 不再申请同一张物理卡；这一步在日志中确实表现为不同的
`CUDA_VISIBLE_DEVICES`，因此资源竞争和单卡 4B optimizer OOM 被解决。但它不等于
宿主机允许每张卡创建多个 CUDA context。

当前 vLLM 0.10.2 的 OPD server 仍会在一张已分配的 GPU 上启动 API/engine/worker
多进程。云端驱动处于 `Exclusive_Process` 时，即使这些服务分别被分配到独立 GPU，
其中的 worker 仍可能收到 `CUDA-capable device(s) is/are busy or unavailable`。
之前的 standalone smoke 只证明了 GPU 分配成功，随后仍停在服务/RPC 初始化，没有
产出 step，因此不能称为 OPD 已解决。

结论：独立 GPU 是必要条件但不是充分条件。下一次 OPD 前需要平台重建容器清除
GPU2/3 的隐藏 context，并由宿主机设置 Default compute mode 或启用 MPS；否则不能
保证 vLLM 多进程 OPD 稳定进入首个训练 step。

容器重启后复测了 E2：四卡干净，`Exclusive_Process` 仍存在，但没有再出现 CUDA
busy；同时通过 `VERL_RAY_JOB_ID=<experiment>_<timestamp>_<pid>` 让 IPC socket
变成每次运行唯一的路径，也没有再出现 `Address already in use`。资源分配、4B
Student、8B Teacher 和 rollout server 都完成了初始化，并进入
`LLMServerManager`。新的阻塞点是首次 `actor_rollout_ref_update_weights` 长时间
等待，超过 8 分钟仍未产生 `step:1`；因此独立 GPU 和 IPC 冲突已分别排除，但
verl v0.8 的 Student-to-vLLM 权重同步/RPC 仍未闭环。该进程已停止并释放四张
A100，未启动 E3 或正式训练。

## R72: 云端恢复 Default 计算模式，正式 verl 训练确认正常

云端容器重启后，平台将 4 张 A100 的 CUDA Compute Mode 调整为 `Default`。复核结果为：

- 4 张 GPU 均可见，均报告 `Compute Mode: Default`；
- 每张卡显存占用约 31.7 GiB，GPU 利用率约 25%-29%；
- 原先由 `Exclusive_Process` 引起的 vLLM 多进程 CUDA busy 阻断不再出现；
- 已运行的正式云端 E1 没有重启或覆盖，日志继续推进到 `37/60`；
- 本地正式 E1 推进到 `47/60`，没有新的 Traceback、OOM 或 CUDA 错误。

因此当前云端环境已经能够正常进行标准 verl 训练。GPU 利用率不持续满载并不代表任务暂停，
当前 step 中生成、同步、反向更新和 checkpoint 等阶段的负载不同；进度应以 step、
`training/global_step` 和 `timing_s/step` 为准。云端 E1 结束后，既有 sequence watcher
将按顺序启动已通过 smoke 的 E2、E3，不并发启动额外任务。

## R76: 本地从 E1 step60 接续 17 小时长程 GRPO

本地新增长程实验 `matrix_local_e1_lora_long_440step`，从已完成的
`matrix_local_e1_lora_60step/global_step_60` 接续，目标总步数为 `440`，实际新增
约 `380` steps。配置保持 Pure GRPO 的已验证条件：2×RTX 4090、LoRA rank `8`、
alpha `16`、rollout.n `4`、batch `8`、2048 response cap、sampling `0.6/0.95/20`、
学习率 `1e-6`，每 10 steps 保存 checkpoint。

日志已经确认加载了旧 checkpoint 的 model、optimizer、RNG 和 lr scheduler，进度条为
`60/440`，因此不是从头训练。按此前约 `160s/step` 估算，预计运行约 17 小时，目标
在 2026-09-02 10:00 左右完成。旧的 60-step checkpoint、rollouts 和评估结果全部保留，
长程实验使用独立目录和 TensorBoard run。

随后首个续训 step 已完成，进度为 `61/440`，耗时约 `149s`，没有出现恢复兼容、显存、
CUDA 或数据加载错误，说明断点续训闭环已经通过。

## R73: 本地 1.7B 正式 E1 完成

本地 2×RTX 4090 的 `matrix_local_e1_lora_60step` 已正常完成 `60/60`，总耗时约
`2小时43分51秒`，平均 step 约 `163.9秒`。最终 step 记录了 reward、advantage、
KL 和 entropy 等指标，并成功保存：

`checkpoints/agopd-rl/matrix_local_e1_lora_60step/global_step_60`

进程已正常退出，日志末尾为 `Final validation metrics: None`，这是当前配置
`trainer.test_freq=-1` 的预期表现，不是训练失败。该 checkpoint 现在可以进入合并和
固定验证集评估；云端 4B E1 仍按原计划继续运行，不受本地任务结束影响。

## R74: 本地 E1 checkpoint 合并并开始固定验证

本地 E1 的最终 actor LoRA adapter 已从
`global_step_60/actor/lora_adapter` 合并为独立模型
`models/Qwen3-1.7B-e1-lora-60step`。合并初次尝试发现脚本继承了用户目录中的
Transformers 5.8，生成的 tokenizer metadata 与 vLLM 环境的 Transformers 4.55.4
不兼容；随后使用 `PYTHONNOUSERSITE=1` 和本地 Conda 动态库路径重新合并，模型加载
和单样本 vLLM smoke 均通过。

固定验证已启动，协议与历史基线一致：`dapo-verl-v1/val.parquet` 的 1024 条样本、
`max_new_tokens=2048`、temperature `0.6`、top-p `0.95`、top-k `20`、seed `42`，
单卡 GPU0。后台任务采用 `setsid + nohup`，因此不会因终端退出而被回收；最终结果
写入 `reports/eval-dapo-v1/matrix-local-e1-lora-60step-v3.summary.json`。

评估已完成，结果如下：

| 指标 | 本地 1.7B E1 LoRA 60 step |
| --- | ---: |
| strict reward accuracy | 24.41% (250/1024) |
| semantic accuracy | 27.54% |
| strict contract accuracy | 11.72% |
| answer parse rate | 82.42% |
| truncation rate | 19.24% |
| mean response length | 1256.2 tokens |

相对固定验证中的 1.7B Base（strict `23.34%`、semantic `25.49%`），本轮分别为
`+1.07pp` 和 `+2.05pp`。这说明 LoRA Pure GRPO 在当前 60 step、n=4、no-think
协议下有小幅正向变化，但不是大幅收益；结果可与云端 4B E1 以及后续 E2/E3 对照。

## R75: 云端 4B E1 完成，E2 自动接续

云端 4B LoRA Pure GRPO E1 已完成 `60/60`，总耗时约 `3小时22分48秒`，最终 checkpoint
位于 `checkpoints/agopd-rl/matrix_cloud_e1_lora_60step/global_step_60`。最终一个训练
step 的在线 reward mean 为 `0.1875`；该值只是 rollout batch 的在线信号，不能替代固定
验证集准确率，后续必须对最终 checkpoint 做统一评估。

sequence watcher 已正常接续启动云端 E2 Vanilla OPD-only。当前 Student=4B、Teacher=8B
均已进入初始化流程，四张 A100 仍为 `Default`，暂未出现 CUDA busy、OOM 或 IPC 错误；
E2 尚未产出 `step:1`，正在等待 Teacher vLLM 完成服务初始化。

E2 进入正式 step 后，step1/step2 分别用时约 `131.5s` 和 `135.0s`；云端 E1 最近
step 约 `198-204s`，当前 OPD 轮次实测快约 `30%-35%`。但这不是单纯由 Compute Mode
切换带来的吞吐提升：`Default` 主要解决多进程 CUDA 启动阻断，E2 同时使用了不同的
verl/vLLM OPD 拓扑和阶段划分。GPU 利用率也不会整体同步升高，E2 当前是 GPU0 Student、
GPU1 Teacher、GPU2 rollout，GPU3 空闲，各阶段会交替等待和计算；应以 step time 和
各阶段 timing 评估性能，而不是看四卡平均利用率。

## R77: 云端 E2 完成并自动进入 E3

云端 E2 Vanilla OPD-only 已完成 `60/60`，平均 step 约 `134.2s`，期间没有 CUDA、OOM、
IPC 或 Ray 错误。sequence watcher 随后按计划启动 E3 GRPO+OPD，当前 E3 已推进到
`21/60`，最近 step 约 `130-140s`，蒸馏 loss 和 rollout 均在正常产出。E2 与 E3 使用
相同的 Student/Teacher/rollout 拓扑和训练预算，后续可以直接进行方法对比。

后续监测中 E3 已推进到 `56/60`，累计平均约 `127s/step`，没有发现新的运行错误；
本地长程 GRPO 同期推进到 `129/440`，两项任务均保持独立输出与 checkpoint 目录。

云端 E3 随后正常完成 `60/60`，总耗时约 `2小时12分09秒`，平均约 `132.2s/step`，
最终 checkpoint 和 TensorBoard event 文件均已保存。至此云端匹配矩阵的训练部分
E1 Pure GRPO、E2 Vanilla OPD-only、E3 GRPO+OPD 全部完成，下一阶段是对三个最终
checkpoint 使用同一固定验证集做最终评估。

## R78: TensorBoard 指标观察规范

实际 event 文件确认了训练曲线会逐 step 写入 TensorBoard。后续观察按以下优先级进行：

1. 收益：`critic/rewards/mean`（或等价的 `critic/score/mean`）。这是 rollout batch 的
   在线 reward 均值，用于观察训练趋势，但不能替代固定验证集的最终 accuracy。
2. 输出质量：`response_length/mean`、`response_length/clip_ratio`、
   `response/aborted_ratio`。其中 clip ratio 是达到 2048 上限的比例，持续升高通常表示
   思考过长或答案截断。
3. 训练稳定性：`actor/entropy`、`actor/kl_loss`、`actor/pg_loss`、
   `actor/grad_norm`。entropy 快速塌陷、KL 或 grad norm 突增都需要重点排查。
4. OPD 信号：`actor/distillation/loss`、`actor/distillation/overlap_ratio`、
   `actor/distillation/student_mass`、`actor/distillation/teacher_mass`、
   `actor/distillation/overlap_token_advantage`。E2 应有 distillation loss 且
   `actor/pg_loss` 接近零；E3 应同时有 distillation loss 和非零 policy-gradient loss。
5. 速度：`timing_s/step`、`timing_s/gen`、`timing_s/update_actor`、
   `timing_s/update_weights`（OPD）、`perf/throughput`、`perf/mfu/actor`。
   显存则看 `perf/max_memory_allocated_gb` 和 `perf/max_memory_reserved_gb`。

`critic/advantages/mean` 经过组内归一化后接近 0 通常是正常现象，不能单独当成收益指标。

## R79: 云端 E1/E2/E3 最终评估修正与结果

第一次 standalone 评估发现 E1 合并目录缺少 112 个权重，E2/E3 的旧 merger 目录虽然
包含完整 base 权重，但 vLLM 不会自动加载其中的 `lora_adapter`。因此第一版 E2/E3
相同结果被判定无效。随后增加 `scripts/evaluate_dapo_vllm.py --lora-adapter`，用统一
的 `Qwen3-4B` base 加载各自 adapter，重新完成 E1/E2/E3 的 direct-LoRA 评估。

正式结果均为固定 `dapo-verl-v1` 验证集 1024 条、no-think、2048 response cap、
temperature `0.6`、top-p `0.95`、top-k `20`、seed `42`：

| 实验 | strict reward | semantic | strict contract | parse | truncation | mean length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 Pure GRPO | 31.25% | 39.84% | 25.49% | 74.90% | 25.78% | 1322.7 |
| E2 Vanilla OPD-only | 30.37% | 39.06% | 25.49% | 74.51% | 26.27% | 1335.1 |
| E3 GRPO + OPD | 30.37% | 39.06% | 25.49% | 74.51% | 26.27% | 1335.1 |

E2/E3 的 1024 条生成文本逐条完全一致；它们的 adapter 文件并非同一个文件，但在
当前 60-step、LoRA rank 8 和固定采样设置下没有改变验证集采样轨迹。因此当前不能
声称 E3 比 E2 有收益，下一轮需要检查 LoRA 更新幅度、adapter 应用路径和更长训练预算。

正式报告保存在本地 `reports/eval-dapo-v1/cloud-final/`，其中 `*-final-v3` 是有效
结果；`*-final` 和 `E1-final-v2` 仅作为诊断历史保留，不用于方法结论。

## R80: E2/E3 输出完全一致的参数级诊断

对 E2/E3 最终 adapter 做参数级比较后确认：504 个 LoRA 张量全部存在差异，平均参数
绝对差约 `4.85e-3`，因此两次训练并没有保存成同一个 adapter。训练日志也确认了方法
分支：E2 的 `actor/loss` 等于 distillation loss；E3 的 `actor/loss` 等于
distillation loss 加 policy-gradient loss，说明 task-reward 分支确实进入了优化目标。

但是 LoRA 的有效低秩增量差异非常小。以 layer0 q_proj 为例，E2/E3 的有效 delta
最大绝对差约 `2.04e-5`，平均绝对差约 `2.84e-6`。在当前 BF16 推理、学习率 `1e-6`、
60 steps 和 rank 8 下，这个差异没有改变任何一个固定采样 token，因此 1024 条生成
逐条一致。结论应表述为“当前预算下 E2/E3 的行为差异不可观测”，而不是“E2/E3
算法相同”或“评估结果可以证明 E3 无效”。

## R81: 云端训练拓扑重构 — 2+2 Hybrid smoke

原拓扑（学生 1 GPU + 独立 rollout 1 GPU + Teacher 1 GPU + 1 空闲）的 E1/E2/E3
step 时间约 132s，四卡利用率低。本轮为拓扑验证新增独立 wrapper
`scripts/run_cloud_opd_hybrid.sh`（不改动任何现有 E1/E2/E3 脚本），使用
`rollout.nnodes=0` 的 v0.8 hybrid 模式：Student FSDP2 与两个 hybrid rollout
replica 共置（物理 GPU1/2，`CUDA_VISIBLE_DEVICES=1,2,0,3` 排序保证学生落同一
NUMA），Teacher 两个 TP1 data-parallel replica（物理 GPU0/3），16 个
AgentLoop workers，`enable_sleep_mode=True` + `free_cache_engine=True`，
LoRA adapter-only 权重同步。

`matrix_cloud_e3_hybrid_2p2_smoke`（5 steps，eager，16K tokens）完成：

| 分项 | 旧 E3 1+1+1 | 2+2 eager | 变化 |
| --- | ---: | ---: | --- |
| gen | 41.0s | 38-41s | 持平 |
| old+ref | ~21s | ~13.5s | -36% |
| update_actor | 52.3s | 34-36s | -33% |
| update_weights | 13.4s | 8.7s（实际同步 0.64-0.8s） | -35% |
| **step** | **132.1s** | **~97s** | **-27%** |

关键机制：`llm_server.py` 在 `nnodes=0` 时 hybrid replica 数自动等于
`worker_group.world_size`（学生 rank 数），TP1×DP1 时 2 个学生 rank → 2 个
hybrid replica；Teacher replica 数同理由 `distillation.n_gpus_per_node // per_replica`
推导。权重同步时间从 13-15s 降到 0.64-0.8s 是因为 hybrid 共置 + sleep mode
只同步 LoRA adapter delta。

## R82: 2+2 CUDA Graph + 3+1 拓扑验证

`matrix_cloud_e3_hybrid_2p2_perf`（eager=False + 32768 tokens）把 gen 从 40s
压到 22-23s（CUDA Graph），step 稳定在约 80.5s（相对旧 E3 -39%）。

`matrix_cloud_e3_hybrid_3p1_smoke`（FSDP3 + 3 hybrid replicas + 1 Teacher TP1，
batch=12 × n=4 = 48 序列/步，CUDA Graph + 32K）完成 5/5，零错误：

| 分项 | 3+1 CUDA Graph（48 seqs） | 2+2 CUDA Graph（32 seqs） |
| --- | ---: | ---: |
| gen | 22.7-27.3s | 22-23s |
| old_log_prob | 4.8-5.3s | 9.2-10.1s |
| update_actor | 15.0-16.2s | 34-35s |
| update_weights | 7.9-8.8s | 8.5-11s |
| **step** | **~59s** | **~80.5s** |

3+1 相对旧 E3 拓扑 step 时间下降 **55%**，且每步处理序列数从 32 增至 48
（吞吐约 3.4×）。单 Teacher replica 没有成为瓶颈：old_log_prob 和 update_actor
随学生 rank 增加而减半。

**3+1 的整除约束链**（n=4、16 AgentLoop workers 下）：
1. `batch × n` % 3 == 0（seqlen Karmarkar-Karp 分区，`equal_size=True`）；
2. `batch` % 3 == 0（组级分区，每组 n 条必须同 rank）；
3. `batch × n` % 16 == 0（AgentLoop `prompts.chunk(len(workers))`）；
4. `mini × n` % 3 == 0（`update_actor` 的 `mini_batch_size % dp_size`）。

batch=8×n=4=32 不满足任何 3 整除条件；batch=12×n=4=48、mini=6 全部通过。
因此 3+1 若用于正式矩阵，协议 batch 需要从 8 调整为 3 的倍数（如 12），
或者通过补丁放宽 Karmarkar-Karp 的等分限制。

拓扑结论：**云端采用 3+1 hybrid（FSDP3 + 3 共置 rollout replica + 1 Teacher
TP1），CUDA Graph + 32K batched tokens + 16 AgentLoop workers**。下一步按
AGOPD 流水线计划重构 Teacher-after-Advantage 双分支，再统一 v0.8 重跑
E1/E2/E3/AGOPD 矩阵。

## R83: Teacher-after-Advantage 重构完成（AGOPD 核心）

重要澄清：云端 smoke 实际运行的 trainer 是 legacy `RayPPOTrainer`
（`verl/trainer/ppo/ray_trainer.py`，经 `main_ppo.py` 进入），不是
`main_ppo_sync.py`（后者依赖未安装的 TransferQueue 包）。重构基于
ray_trainer.py 完成，共 6+2 个文件：

- `verl/workers/config/distillation.py`：新增 `AdvantageGateConfig` 和
  `DistillationConfig.teacher_after_advantage` / `advantage_gate` 字段；
- `verl/trainer/config/distillation/distillation.yaml` 和
  `_generated_ppo_trainer.yaml`：schema 同步；
- `verl/experimental/agent_loop/agent_loop.py`：`teacher_after_advantage`
  开启时跳过生成期的 eager teacher 打分；
- 新增 `verl/trainer/ppo/teacher_scoring.py`：`TeacherScoringWorker` Ray
  actor（纯 vLLM HTTP client，无 GPU），异步批量 top-k 打分；
- `verl/trainer/ppo/ray_trainer.py`：fit() 流水线重构
  gen → reward → **adv_early → gate → 提交 teacher 打分(异步)** →
  old/ref（与 teacher 重叠）→ **收集 teacher 结果** → update_actor；
  新增 `_apply_advantage_gate` / `_submit_teacher_scoring` /
  `_collect_teacher_scoring`；
- `verl/workers/utils/padding.py`：`opd_weights` 全长 nested 转换；
- `verl/trainer/distillation/losses.py`：loss 与指标按 opd_weights 加权，
  全零 gate 时零化 loss 防 nan，空集指标守卫。

调试过程中修复的三个问题：GRPO advantage 形状是 (bsz, resp_len) 而非全长
（opd_weights 需嵌入全长张量的响应段）；`teacher_ids` 不能用 -1 占位
（`torch.gather` 触发 CUDA device-side assert，改用有效词汇索引 0）；
空 micro-batch 的 `min()`/`agg_loss` 0/0 守卫。

`matrix_cloud_e3_agopd_3p1_smoke`（5 steps，3+1 拓扑，batch=12/mini=6，
CUDA Graph + 32K）验证通过，零错误：

| 指标 | 结果 |
| --- | --- |
| gate activated_ratio | 12.5% / 12.5% / 20.8% / 14.6%（约 15%） |
| teacher_tokens | 7130-11565/step（仅约 15% 轨迹打分） |
| actor/distillation/loss | 非零（0.0045-0.015） |
| timing_s/step | ~57s（与无 gate 3+1 持平） |
| teacher_score 等待 | 0.006-0.009s（打分完全隐藏在 gen+old/ref 下） |
| adv_early + adv_gate 开销 | ~10ms |

结论：Teacher-after-Advantage 流水线生效——Teacher 算力消耗降至约 15%，
step 时间不变，蒸馏信号正常。正式 400 步训练
`matrix_cloud_e3_agopd_3p1_400step` 已启动（每 10 步保存 checkpoint，
预计约 6.3 小时）。

## R84: AGOPD 400 步正式训练完成

`matrix_cloud_e3_agopd_3p1_400step`（3+1 hybrid 拓扑 + Teacher-after-
Advantage gate，batch=12/mini=6，CUDA Graph + 32K，每 10 步保存）已正常
完成 400/400，总耗时 6h44m50s，平均约 60.7s/step（最后 10 步平均 62.6s），
零错误。

- Checkpoint：40 个（global_step_10 至 global_step_400），
  `latest_checkpointed_iteration.txt = 400`；
- Rollout：400 份；
- Gate 激活比例：早期 20 步平均 20.4%，后期 20 步平均 21.3%——整个训练
  稳定在约 20%，即约 80% 的 Teacher 算力被省下；
- 蒸馏 loss 全程非零（末期 0.0047-0.0133），门控蒸馏持续生效。

下一步：合并 global_step_400 并做固定 1024 条验证集评估，与 E1/E2/E3
60 步结果对照（E1 31.25%、E2/E3 30.37% strict）；本地长程 GRPO
（matrix_local_e1_lora_long_440step）仍在运行（约 87%）。

## R85: AGOPD 400 步固定验证集评估

合并时发现 R79 教训重现：`lora.merge=True` 保存的 checkpoint 中，
`model_world_size_*.pt` 分片与 base 权重完全一致（最大绝对差 0.0），训练
更新全部在独立的 `lora_adapter/` 目录（504 张量，rank 8，lora_A 最大幅度
0.0107）。**评估必须用 `--lora-adapter` 的 direct-LoRA 路径**，直接评估
合并目录会得到与 Base 逐项一致的无意义结果（`agopd-4b-400step.summary.json`
33.01% 判定无效，仅作审计保留）。另外云端 `evaluate_dapo_vllm.py` 是
旧版（无 `--lora-adapter` 参数），已从本地同步新版脚本。

direct-LoRA 评估结果（`reports/eval-dapo-v1/agopd-4b-400step-lora.summary.json`，
固定 1024 条、2048 budget、0.6/0.95/20、seed 42）：

| 实验 | strict | semantic | contract | parse | truncation | 长度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B Base | 33.01% | 41.02% | — | 75.20% | 25.20% | 1325.5 |
| E1 Pure GRPO 60步 | 31.25% | 39.84% | 25.49% | 74.90% | 25.78% | 1322.7 |
| E2 OPD-only 60步 | 30.37% | 39.06% | 25.49% | 74.51% | 26.27% | 1335.1 |
| E3 GRPO+OPD 60步 | 30.37% | 39.06% | 25.49% | 74.51% | 26.27% | 1335.1 |
| **AGOPD 400步** | **32.42%** | 34.28% | **29.39%** | 73.54% | 27.83% | 1395.5 |

分析：
- AGOPD 400 步 strict 32.42% 为 LoRA 系最高（E3 +2.05pp），strict contract
  29.39% 全场最高（E1-E3 均为 25.49%），说明 gate 蒸馏显著提升格式合规；
- 但 semantic 34.28% 明显退化（Base -6.74pp、E3 -4.78pp），truncation
  27.83% 略升——严格 reward 与语义能力分离的现象在 400 步后仍成立；
- caveat：AGOPD 400 步处理序列数 19200 vs E3 的 1920（10×），batch 协议
  12 vs 8，提升不能完全归因于 gate。

本地长程 Pure GRPO（matrix_local_e1_lora_long_440step，440 步）已完成，
同预算 Pure GRPO vs AGOPD 对照是回答"gate 是否有效"的关键实验。

## R86: 为什么 E1/E2/E3/AGOPD 都没超过 Base — reward 曲线诊断

从 TensorBoard 事件提取四个实验的曲线，结论是 reward 曲线全部是平的，
无学习趋势：

| 实验 | 首 5 步 | 末 5 步 | 全程均值 | 趋势 |
| --- | ---: | ---: | ---: | ---: |
| E1 Pure GRPO 60步 | 0.344 | 0.344 | 0.338 | 完全平 |
| E2 OPD-only 60步 | 0.306 | 0.294 | 0.329 | 微降 |
| E3 GRPO+OPD 60步 | 0.363 | 0.306 | 0.336 | 微降 |
| AGOPD 400步 | 0.283 | 0.308 | 0.326 | 平（50 步窗口 0.322→0.337） |

AGOPD 400 步细节：entropy 0.124→0.216 不降反升；响应长度 1162→1405
持续变长；distillation loss 0.040→0.008 说明学生对 teacher topk 的
对齐确实在发生——但未转化为 reward。

**三层因果诊断**：

1. 信号层：rollout 分组统计（12 组×4）显示约 50% 组全错、10% 全对，
   只有约 40% 混合组产生非零 advantage——每步 48 条序列里有效学习信号
   只有约 17-20 条；
2. 动力学层：400 步后 LoRA 有效权重变化比 = 5.18e-4（相对 base 权重
   幅度，BF16 可观测阈值的约 1e4 倍）——更新真实存在且可观测，但
   logits 扰动量级不足以改变 top-k 采样轨迹（E2/E3 60 步时 1024 条
   生成与 base 逐条一致；AGOPD 400 步也只差约 10 条）；
3. 目标层：8B Teacher strict 仅 36.43%（比 4B base 高 3.4pp），蒸馏
   上限低；实际学到的是格式合规（strict contract 29.39% 全场最高）
   而非解题能力。

**最强对照证据**：历史 4B Pure GRPO full-param、n=2、50 步达到 strict
40.23%——同样的 reward/数据/预算，full-param 大幅超过 Base。结论：
瓶颈不在 reward 设计、gate 或数据，而在 LoRA rank 8 + lr 1e-6 的
更新通道太窄。reward 曲线平是该配置下的正常规律。

**验证路径**（路径 1）：Pure GRPO + lr 1e-5/1e-4 × rank 16，20 步
smoke，判据是 reward 曲线是否脱离 0.31-0.34 噪声带、entropy 是否开始
下降。

## R88: Run C 验证 — lr=1e-4 + rank=16 首次大幅超越 Base

LoRA 强度扫描结果：lr=1e-5+rank16 的 20 步 reward mean 0.329 仍在噪声带；
**lr=1e-4 + rank=16 的 20 步 reward mean 0.415 脱离噪声带**。正式 Run C
（Pure GRPO，lr=1e-4，rank=16，60 步，save_freq=10，3+1 拓扑）：
reward 10 步窗口 0.356→0.448 稳步上升（first5 0.267→last5 0.471），
EGR first5 0.383→last5 0.467，是第一个真正学习的 LoRA 实验。

固定 1024 验证集（direct-LoRA，评估脚本需 --lora-rank 16）：

| 实验 | strict | semantic | vs Base |
| --- | ---: | ---: | ---: |
| 4B Base | 33.01% | 41.02% | — |
| E1 Pure GRPO 60步 (lr1e-6 r8) | 31.25% | 39.84% | -1.76pp |
| E2/E3 60步 (lr1e-6 r8) | 30.37% | 39.06% | -2.64pp |
| AGOPD 400步 (lr1e-6 r8) | 32.42% | 34.28% | -0.59pp |
| 4B full-param 50步 | 40.23% | 40.23% | +7.22pp |
| **Run C (lr1e-4 r16 60步)** | **42.48%** | **42.48%** | **+9.47pp** |

Run C strict 42.48% 超越全部历史实验（含 full-param 40.23%），semantic
同步提升（此前 LoRA 实验 semantic 均退化），parse 77.73%（+2.53pp），
截断率下降。结论：R86 诊断闭环验证——lr=1e-6+rank8 更新通道过窄是此前
所有 E1-E3/AGOPD 对比无效的根因；**lr=1e-4 + rank=16 成为新的主实验
基线**，AGOPD gate/Teacher-after-Advantage/SFT cold-start 均需在该
基线上重新验证。

## R89: Hard-Prompt Targeted Cold Start 数据构建（第一批 1000 题）

新 pipeline `scripts/build_coldstart_data.py`（本地验证后上云端 4 卡
分片并行，支持 --shard-idx/--n-shards）：

- Stage A：4B × n=8 估计每题 p̂（全局 seed 抽样,分片连续切段）；
- Stage B：hard 题 4B 扩采到 n=32（Student Rejection Sampling，on-policy）；
- Stage C：still-zero 题 8B × n=8（Teacher Correct Only）；
- 组装：每题最多 2 条 verifier-confirmed 轨迹（防单题垄断）。

本地 30 题 pilot 验证管线 + 显存修复（4B/8B 引擎不能同时在 24GB 4090
共存，Stage C 前释放学生引擎）；200 题本地构建完成。

云端 4×A100 并行 1000 题（每 shard 250）结果：

| 指标 | 值 |
| --- | ---: |
| 分类 | easy=181 (18%) / frontier=420 (42%) / hard=399 (40%) |
| hard 率 | 40%（与 200 题 40%、pilot 33-40% 一致,分布稳定） |
| SFT 轨迹 | **232 条 / 154 unique hard 题** |
| └ 4B student rejection | 182 条 (79%) |
| └ 8B teacher correct | 50 条 (21%) |
| 两来源题集重叠 | 0（完美互补） |
| 质量 | 100% verifier-confirmed；契约格式 95% Answer: + 5% boxed；长度 665-6456 字符 |

数据位置：`outputs/coldstart-1000/`（sft_data.jsonl + buckets.jsonl）。
覆盖 154 个 unique hard 题 ≈ 全训练集 hard 题（~6500）的 2.4%——
作为第一轮 targeted SFT 语料起步，后续按迭代扩量。下一步：以
Run C 基线（lr=1e-4+r16）做 SFT + GRPO Readiness 评估（all-zero↓、
mixed↑、all-one 不爆），验证 ERSR 40%→65%+ 假设。

## R87: 本地 440 步评估 — LoRA 长训练无增益的决定性证据

本地 440 步 Pure GRPO（LoRA rank 8，lr 1e-6，direct-LoRA 评估，固定
1024 条验证集）结果：strict 23.93%、semantic 26.56%、contract 12.30%、
parse 80.86%、truncation 19.92%、长度 1255.0。

本地 1.7B 全对照：

| 实验 | strict | semantic | vs Base |
| --- | ---: | ---: | ---: |
| 1.7B Base | 23.34% | 25.49% | — |
| LoRA E1 60 步 | 24.41% | 27.54% | +1.07pp |
| LoRA long 440 步 | 23.93% | 26.56% | +0.59pp |
| full-param 50 步（lora_rank=0） | 25.39% | 27.44% | +2.05pp |

三个关键发现：
1. LoRA 从 60 步跑到 440 步（7.3× 预算）成绩反降 0.48pp——长训练
   无收益且有轻微退化，与 reward 曲线平的观测一致；
2. full-param 50 步（25.39%）> LoRA 440 步（23.93%），少 8.8× 步数
   高 1.46pp；
3. 与云端 4B 结论一致（full-param 50 步 40.23% vs LoRA 400 步
   32.42%），两个规模互相印证。

**诊断闭环**：reward 曲线平 + 步数翻倍无增益 + full-param 少步数反超
= "LoRA rank 8 + lr 1e-6 更新通道容量不足"是主因，与 reward 设计、
gate、数据无关。LoRA 强度扫描（lr 1e-5/1e-4 × rank 16，20 步）在
云端验证中。

## R90: 1.7B SFT 冷启动训练与评估(2026-09-04)

**数据**:`outputs/coldstart-17-4000/`(1451 条 / 931 prompts,cap 2/prompt,
teacher 4B-GRPO ckpt2 40.23%,Frontier Shaping:hard/frontier 分类 + rejection + backfill)
**训练**:LoRA r16 lr 1e-4,1024 上下文 + 2560 max_len,1200 步(ckpt-1100 为最后完整
adapter;~1200 步 SIGABRT 中断,当时 eval_loss 0.168 平台、仍极缓下降、未触发 early
stop——ckpt-1100 代表收敛态)。eval_loss 下降极缓后中断,无再训价值。
**评估**(direct-LoRA r16,固定 1024 条,seed42):strict **24.71%** (253/1024),
semantic 25.49%,contract 12.79%,trunc 22.07%,长度 1264.7。

| 1.7B 实验 | strict | vs Base | 备注 |
| --- | ---: | ---: | ---: |
| Base | 23.34% | — | |
| **SFT 冷启动 1100 步** | **24.71%** | **+1.37pp** | 本轮 |
| GRPO 50 步 full-param | 25.39% | +2.05pp | 本地历史 |
| E1 60 步 LoRA | 24.41% | +1.07pp | |
| E1 440 步 LoRA | 23.93% | +0.59pp | |

对照 4B 场景(4B SFT 冷启动 39.65% = Base +6.64pp),1.7B 增益弱(+1.37pp)。
嫌疑:数据量小(~931 prompts)与 teacher 上限(4B-GRPO 40% 而非 8B)+ 截断
22%。结论:SFT 冷启动在 1.7B 上可用但增益有限,作为冷启动底座(frontier
题已进分布)的价值高于绝对 accuracy 提升。

## R91: 云端 1.7B 四方法矩阵 E2 崩溃与续跑(2026-09-04)

矩阵(seed42、LoRA r16 lr1e-4、3+1 拓扑、batch12 n4、200 步每臂、hard_ce):
E1 Pure GRPO 03:53 完成 200 步(156GB ckpt)。E2 Pure OPD(hard CE,
use_task_rewards=False,taa=False)12:16 崩于 step 61:
**entropy_from_logits CUDA OOM**(GPU0 已用 56.45GB + 需 20.98GiB fp32
logsumexp 峰值;step 60 时 actor reserved 已爬至 64.2GB,per-step 峰值波动
+ caching allocator 不回落 → 临界崩溃)。
E2 前 60 步健康:entropy 0.43(初始)→ 0.13-0.2 稳定收敛(hard CE 生效);
gen 首步 192s(graph 捕获)后 17-29s;distill loss 0.2-0.3 非零;score mean
0.1875-0.23。global_step_60 checkpoint 完整(actor+optim+extra+data.pt)。

**脚本 bug 修复**(run_matrix_17_continue.sh):
1. run_one shift-2 吞参 bug:第 4 参数(如 hard_ce=False)被静默丢弃 → 改 shift 4
2. teacher_after_advantage 移出 COMMON 每臂显式(无 gate 臂 False / AGOPD True)
3. E2 resume:trainer.resume_mode=resume_path + resume_from_path=global_step_60
   (verl v0.8 _load_checkpoint 支持,global_steps=60 续跑至 200)
4. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 防 reserved 爬坡 OOM

E2 resume 13:38 启动(pid 3193790),后续 E3(GRPO+OPD,taa=False)→ AGOPD
(taa=True + advantage gate + teacher gate)。

**补丁(13:45)**:PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 与 vLLM
CuMemAllocator 不兼容(vLLM 显式 assert "Expandable segments are not compatible
with memory pool" → EngineCore 启动失败)。修复:`vllm_async_server.py`
vLLMHttpServer.__init__ 内剥离该 env(仅 vLLM 进程),FSDP 训练侧保留
expandable_segments。本地与云端源码同步。13:43 重启 E2 resume(pid 3203245)。

## R92: 4B RL 轨迹冷启动 v2 数据与双 SFT 对比实验(2026-09-04)

**动机**:v1 冷启动(4B-GRPO rejection,1451 条)+1.37pp 增幅小;用户提议用 4B 的
RL 训练轨迹做冷启动语料(4B 更强的解题轨迹 + verifier 标注)。

**v2 数据**(云端构建,coldstart-17-v2):4 源 score=1 轨迹合并 —
AGOPD-400 step>250(2321)+ HardCE-100(1392)+ RunC-100(1003)+ GRPO-ckpt2-50(626)
= **5342 条 / 2065 unique 题**(与 v1 题集零重叠)。清洗后 train 4795 + val 532。
输出格式与 v1 同构(messages/source 可溯源到实验+step)。

**SFT 对比**(本地 2×4090 DDP 串行,同硬件同配方:LoRA r16 lr1e-5,3 epochs+早停):
- v1-3ep:旧数据充分训练(对照旧 1.65 epochs 24.71%)
- v2-3ep:4B RL 轨迹
- 环境坑:本地 2×4090 FSDP 训练不可用(NCCL 217);transformers 5.8
  apply_chat_template/BatchEncoding API 变化需兼容;vllm 0.10.2 与 transformers
  5.8 不兼容(all_special_tokens_extended 移除)→ getattr fallback patch。
- 结果:固定 1024 评估对比(v1-ddp3ep vs v2-ddp3ep strict accuracy)

**环境注意**:agopd-vllm-281 env 于 9/3-9/4 间被升级(transformers 5.8.0),
9/3 前 vllm 组合兼容;升级后需 patch(vllm/transformers_utils/tokenizer.py
getattr fallback + LD_LIBRARY_PATH 指向 conda lib)。

## R93: v1 vs v2 冷启动对比评估结果(2026-09-04 16:2x)

固定 1024 验证集,direct-LoRA r16,同配方(3 epochs + 早停,双卡 DDP 硬件一致):

| 模型 | strict | semantic | trunc | 长度 | vs Base |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.7B Base | 23.34% | 25.49% | 19.9% | 1261.7 | — |
| v1-1.65ep(旧,1200 步) | 24.71% | 25.49% | 22.1% | 1264.7 | +1.37pp |
| **v1-3ep**(4B-GRPO rejection 1300 条) | **25.20%** | 27.15% | 22.5% | 1261.5 | **+1.86pp** |
| v2-3ep(4B RL 轨迹 4795 条) | 24.51% | 24.61% | 34.1% | 1420.6 | +1.17pp |

**结论**:
1. v1 数据充分训练(3ep)上限 25.20%(+1.86pp),高于 1.65ep 的 24.71% — 之前
   SIGABRT 中断低估了 v1 数据价值;
2. **v2(4B RL 轨迹全量直训)不敌 v1**(-0.69pp):数据量 3.7× 未转化为增益;
3. 原因:① 长度/截断 — RL 轨迹教出长输出风格(1421 vs 1261 tokens),评估 34%
   被 2048 cap 截断(v1 仅 22%),大量丢分;② 无 hard/frontier 过滤 — v2 含大量
   easy 题(学生已会,教学价值低),v1 的精选过滤有效;
4. **互补性**:v2 与 v1 题集零重叠(2065 vs 931 题)→ 4B RL 轨迹不是无效,
   是"需要改造":候选方向 A=v2+1.7B hard 过滤; B=长度约束+后期 AGOPD 子集;
   C=v1∪v2 合并(题覆盖 ~3000)。
5. 产物:models 侧 adapter `outputs/sft-coldstart-17-v1-ddp3ep/`(25.20% 新 1.7B
   冷启动底座,取代 ckpt-1100 旧底座 24.71%)与 `v2-ddp3ep/`(24.51%,对照臂)。

## R94: comfort 桶 + full-param 四方法矩阵 — 1.7B 首次大幅突破(2026-09-05)

配置:comfort 桶(1472 题,p̂≥0.25,1.7B base 普查)+ full-param lr1e-6 KL0.05
+ batch8×n6 + 184 步(全桶一轮)。固定 1024 评估:

| 方法 | strict | vs Base | 备注 |
| --- | ---: | ---: | ---: |
| Base | 23.34% | — | |
| **GRPO** | **28.42%** | **+5.08pp** | 历史最强 1.7B |
| AGOPD(gate) | 27.05% | +3.71pp | gate 保护蒸馏 |
| OPD pure hardCE | 25.68% | +2.34pp | 正则增益 |
| RL+OPD 无 gate | 24.12% | +0.78pp | 蒸馏污染 GRPO −4.3pp |

**关键链条验证**:①数据分桶(79.6% hard → comfort 聚焦)②n↑(EGR 60→80%)
③full-param 通道(grad 30× LoRA)④lr1e-6+KL0.05 稳定(避免 lr3e-4/1e-5 崩坏)
⑤学习信号(score−p̂ +5.8pp)→ 最终 +5.08pp。
**AGOPD gate 直接价值**:gate 版 vs 无 gate RL+OPD = +2.9pp(teacher 弱分布污染被挡)。
**OPD 数据侧**:comfort 桶 teacher(4B-GRPO)可教题仅 10%、teacher 显著弱 75.8% —
无 gate 蒸馏必然有害(RL+OPD 证实)。
**工程**:entropy_from_logits_with_chunking 支持任意维 patch(云端+本地,解决
full-param 3+1 entropy OOM);DTensor checkpoint 合并脚本(FSDP2 → HF)。

## R95: 普查修正 79.6%→89.2% — 行序断层说法被证伪(2026-09-05)

- 事件:Stage-2B 分桶需要全量 p̂。补测此前缺失的 4,000 题(普查文件只覆盖 12,164/16,164;缺失 = pilot 510 + full 3,490,行位置均匀:前段 24.5%/中段 52%/后段 23.5%)。
- 结果:补测 4,000 题 88.3% hard(p̂=0)。全量合并:16,164 题中 **hard 89.2%**、p̂=0.125 6.9%、[.25,.5) 2.8%、[.5,.75) 0.8%、≥.75 0.3%,mean p̂=0.0235。
- **证伪**:报告 fig1 的"前 4000 行 hard 49.6% / 后段 89.5%"行序断层说法——全量普查按行段切:0-4000 行 89.5% / 4000-12164 行 88.9% / 12164+ 行 89.5%,三段同质。"79.6%" 是 49.6% 插值(无法溯源,应来自早期小样本),已全面修正为 89.2%(fig1 重画、报告全文、RFT 基线、Claim 1)。
- 连带:RFT 轨迹 hard 占比 90.2% 与全量题分布 89.2% 一致(此前以为"轨迹被偏置推向 hard",实为题分布本来 ~89%);comfort 桶等衍生统计不受影响(基于 p̂ 阈值而非行序)。
- 影响:梯度饥饿叙事更强(可学题从 ~17% 收窄到 ~10.5%),方向不变。

## R96: 固定评估协议陷阱 3 — 2048 cap 对长输出模型的系统性低估(2026-09-05)

- 现象:RFT-SFT(5000 步)固定 1024 评估 25.29% < GRPO-180 28.42%;ckpt 曲线 3500 步 26.7% → 5000 步 25.3%(单调退化,疑似过拟合)。
- 诊断(max_tokens 2048→4096 双模型对照):GRPO-180 28.42→30.37%(截断 21.7→5.6%),RFT@5000 25.29→30.96%(截断 36.7→17.3%)。**RFT 能力被 2048 掩盖,4096 下反超 GRPO +0.6pp**。
- 机制:RFT 回放长轨迹(均值 1412 tok)→ 输出分布拉长(4096 下均值 1952 vs GRPO 1515)→ 2048 预算下截断率高 2× → 答案被砍。
- checkpoint 曲线(4096 协议,4 卡并行探针):ck1000 30.1 / ck2000 31.1 / **ck3000 33.5(峰)** / ck4000 32.7 / ck5000 31.0 → 3000 步后过拟合 RFT 数据(90% hard 回放)。
- 协议教训:**评估预算 ≠ 训练预算 ≠ 报告口径**;固定评估协议(§4.1)需记录 max_tokens,跨模型对比必须同预算或报告截断率。

## R97: 教师条件能力图 — rescue 价值严格集中在 hard 桶(C_rescue 直接验证)(2026-09-05)

- val 1,024 按 base p̂ 分 5 桶,4B-GRPO teacher vs GRPO-180 student 各 n=8 同题采样(教师分层评测,§7.5 教训的 P0 修复):
  hard(p̂=0) T−S **+0.033** / p̂∈[.125,.25) **+0.056** / [.25,.5) +0.015 / [.5,.75) **−0.060** / ≥.75 **−0.158**。
- 结论:teacher 4B 的 rescue 优势只存在于 hard 桶(学生救不了的区域);easy 桶 teacher 反而弱(学生 GRPO 已超 teacher)。**全局 bake-off 失败 ≠ teacher 无能力,是考核区域错**;教师选择/门控协议改为按桶条件评测。
- 注:脚本 gmu 0.8+0.72 双模型在 4090(24GB)无法共存(需 ~37GB),该评测只能在 A100;4090 上引擎 init 失败即此因。

## R98: Stage-2B 数据组装与 smoke — AGOPD/GRPO 240 步串行开训(2026-09-06)

- 数据:跳过 ck3000 重普查(4096 全量普查实测 4 并发引擎 0 产出/吞吐 292 tok/s,引擎参数折腾后放弃),用 **GRPO-180 的 p̂'(rft_survey 全量,16,164 全覆盖)分桶**:train_stage2.parquet(7,360 题)= p̂'≥0.125 全收(6,134)+ 20% p̂'=0 掺杂(1,226,seed42)。mean p̂'=0.319,EGR(n6)=60.4%。
- 模型:rft3000 = GRPO-180 + RFT-SFT ck3000 合并(固定评估 33.5%@4096)。教师 4B-GRPO 不变。
- 超参:同 Stage-1 AGOPD-240(lr 1e-6、KL 0.05、full-param、batch8×n6、seed42、max 3073);算法对比 = AGOPD(gate on)vs GRPO(distillation off),唯一差异蒸馏。
- smoke(20 步):两者均通过。GRPO:score 0.44±0.13、entropy 0.10→0.127、kl 0.0004→0.002、grad_norm 0.69-1.15(梯度流动);AGOPD gate 激活 ~30%、teacher 每步 1-3 万 tokens、competent_ratio 波动(不 competent 被 gate 丢弃 = 设计行为)。
- 排期:AGOPD-240(~70s/步 → ~4.7h)→ 自动接 GRPO-240(~4.2h);云端 watch_stage2b.sh 守护(5 分钟自检 + 异常标记 + 自动串联),本地隧道 tensorboard:6007 实时曲线。
- 云端工程教训(本轮大量踩坑,重要):
  1. **vllm 引擎 init 失败/卡死的统一根因 = 显存幽灵**:引擎被 kill 后 CUDA context 不释放(容器内不可见、kill 无效、gpu-reset Not Supported),表现为"GPU 满载但 0 产出"(progress 0/256,0 tok/s)或 "Free memory < gmu" init 失败;唯一解:等自动回收(几分钟)或宿主侧清理。启动前必须确认 4 卡 <1GB。
  2. 4 引擎并发 init 竞争加剧失败率(2 引擎稳定);每轮失败制造新幽灵 → 失败循环。纪律:干净窗口启动、不自动重试 wrapper(重试循环叠进程)、启动后盯到 step 产出再挂监控。
  3. pkill -f 模式会自杀(命令行含匹配串);引擎进程名 VLLM::EngineCore 需按 /proc cmdline 扫描杀。
  4. torch.compile 每次冷启动 ~5s(TORCH_CUDA_ARCH_LIST 未设全 arch 编译)。

## R99: Stage-2B 完成 — GRPO-240 33.2%@4096 新最强,预算瓶颈确认(2026-09-06)

- 训练:rft3000 初始化,AGOPD-240 与 GRPO-240 串行(各 ~4.2-4.7h,full-param lr1e-6 KL0.05 n6 batch8 seed42,同一 train_stage2 7,360 题)。云端 watch_stage2b.sh 守护全程无异常,但"AGOPD 完成→自动接 GRPO"分支因 tqdm 240/240 匹配时机未触发,手动接续(教训:自动串联的完成判定别依赖 tqdm 文本)。
- 固定 1024 评估(FSDP checkpoint 需先 legacy_model_merger 合并;verl ckpt 的 huggingface 目录无权重,直接 vllm 报 "Cannot find any model weights"):

| 模型 | @2048 | @4096 | 截断@2048 |
| --- | --- | --- | --- |
| GRPO-180(Stage-1 对照) | 28.42% | 30.37% | 21.7% |
| **GRPO-240(Stage-2B)** | **29.10%** | **33.20%** | 21.2% |
| AGOPD-240(Stage-2B) | 27.05% | 29.20% | 30.4% |

- **分层诊断**:GRPO-240 增益在 easy(+6.3pp,Stage-1 easy 退化被修复)与 .25-.5(+8.9pp);hard 桶 @4096 0.096(base 0 的近 10 倍)。AGOPD 仅 .5-.75 桶赢 GRPO;teacher 4B 上限(~30%)已低于学生(33%)→ 蒸馏成噪声源,AGOPD 全面落后(与 Stage-1 的 −1.4pp 一致且放大到 −2~−4pp)。
- **预算瓶颈确认(用户洞察验证)**:hard 桶 @2048→@4096 近翻倍(0.054→0.096),训练 rollout 预算 2048 截断长推理 → 无信号 → 训练曲线噪声震荡(score 每步 48 采样 + 截断双噪声);截断率下降是模型"学写短"而非能力提升。突破方向:训练预算 4096(成本 ~2×)。
- 产物:models/stage2b_{agopd,grpo}240(合并模型);曲线 fig12;评估存档 reports/figures/*eval*.summary.json。

## R100: Stage-2C — 4096 训练预算 + 错误长度惩罚 + 4 卡提速(2026-09-06)

- 背景:审计 GRPO-240 轨迹发现四类失败(冗长逃避/伪枚举/漏条件/算错近失),落地"错误长度惩罚":reward = +1(对)/ −C·min(1,len/MAX)(错),C=1.0、MAX=16000 字符(≈4096 tok),env 可配(WRONG_LEN_PENALTY_C),verl_adapter.py 层实现(不动 math_reward)。
- 提速组合(GRPO 无 teacher 却用 3+1 拓扑,GPU3 闲置):STUDENT_GPUS=4、rollout gmu 0.25→0.4、max_num_batched_tokens 16384→32768、4096 rollout 预算(max_model_len 5121)。坑:agent.num_workers 32 与 48(8×n6)不整除 → verl assert;回 16。实测 ~27s/步(smoke),240 步 2.75h——4096 预算比 Stage-2B 的 2048/3卡 还快 28%。
- 固定评估(stage2c_grpo240 合并模型):

| 模型 | @2048(官方) | @4096(上限) | 截断@2048 | len@4096 |
| --- | --- | --- | --- | --- |
| Stage-2B GRPO-240 | 29.10% | 33.20% | 21.2% | 1510 |
| **Stage-2C** | **30.57%** | 31.35% | **8.2%** | 1171 |

- **双面性结论**:官方 2048 协议史上最强(+1.47pp;截断 21.2→8.2%,长度 1510→1171,高效短收尾);但 4096 上限 −1.85pp——分层显示惩罚把 hard 桶长度从 1814 砍到 1267(hard 0.096→0.085),**误伤 hard 长推理**:杀冗长逃避的同时压制了正确方向的长探索。官方指标(2048)与能力上限(4096)是两把尺;惩罚需与预算解耦(按桶条件惩罚/阈值化)。
- 产物:models/stage2c_grpo240;评估存档 reports/figures/stage2c_eval*.summary.json;报告 §Stage-2 2D。

## R101: 思考模式解锁 — 1.7B 追平 4B 的最大单一发现(2026-09-06)

- 用户提出"reward 提升小,是否开推理模式 + 提预算"。排查发现:**所有训练/评估的 closed_thinking_rate = 1.0**(DAPO prompt 带 /no_think 强制直答),Qwen3-1.7B(原生 thinking-capable)被压成直答模型。
- 探测(stage2c_grpo240 同模型同题):prompt /no_think→/think + max_tokens 8192:
  - 128 题对照:no_think 28.91% vs think 42.19%(+13.3pp),长度几乎相同(3786 vs 3807)→ **增益来自思考内容组织,不是长度**
  - 全量 1024(4 卡分片):**think 41.02%**(shard 0.383-0.438 一致),对比官方 no_think@2048 30.57% / @4096 31.35%
  - 对照 4B-GRPO-50step(no_think)40.23% → **1.7B 开 think 追平并超过 4B**
- 长度惩罚复盘(用户判断无收益,文献核实):错误×长度(C=1.0)在官方 2048 协议 +1.47pp(高效收尾),但 4096 上限 −1.85pp(分层:hared 桶平均长度 1814→1267,惩罚压制 hard 长推理)。主流做法无此先例——DAPO overlong shaping(软阈值,只罚超预算,与对错无关)+ repetition penalty(DAPO-Math-17K 消融 +2-3pp)+ context curriculum(DeepScaleR 1.5B:8K→16K→24K 渐进,纯二值 ORM)。**结论:长度约束应只罚"超预算浪费"与"重复循环",预算内自由;预算推进用 curriculum 而非一步跳**
- 甜区结论:数据甜区(comfort/frontier)之外存在模式甜区(thinking);1.7B 之前提升小 = 在 no_think 封印内优化。下一步:think 训练(数据 /think + 预算 8K curriculum)。

## R102: think-RL 110 步试训 + 真实评估修正(2026-09-06)

- think-RL(stage3a_think_120):起点 stage2c_grpo240,旧分桶数据(train_stage2_think_v2,prompt /think),8192 预算(max_response_length=8192,曾漏设导致 2048 截断),纯二值奖励(惩罚已移除),n4+gmu0.5(~95s/步)。
- 数据坑修复:train_stage2.parquet 的 prompt 是 arrow list<struct>(结构化消息),pandas roundtrip 会变 JSON 字符串导致 verl 丢 prompt(rollout 输出"请假条"、input 只剩 "assistant")→ 需保持 list 结构原地改 content。
- 110 步提前终止(在线分 0.7-0.9 饱和震荡,用户判断);权重对比 stage2c:284/311 层有差异但 max diff 仅 2.4e-4。
- **评估误读事故**:4 卡评估 init 失败(残留引擎占显存)未覆盖旧输出文件(think_full_s*.json 同名),汇总读到 stage2c 旧结果误报 41.02%=零提升;清残留+删旧文件重跑后真实结果:**41.60%(426/1024,+0.58pp vs stage2c 41.02%)**。
- 结论:think 域在 no-think 分桶数据上接近饱和(旧分桶=no_think 视角的可学子集,think 已掌握);think 域真正可学子集需全量重新普查。产物:models/stage3a_think110 存档;附录 D 留存。

## R103: CRPTR 反事实教师救援实验 — 错误路径会降低 teacher 可救援性(2026-09-08)

- 目的:检验“student 进入错误路径后,teacher 在 student state 上也救不回来;RFT 从原题重启因此优于 OPD”的机制假设。
- 配置:cloud 4×A100;student=`Qwen3-1.7B base`;teacher=`Qwen3-4B-grpo-50step-ckpt2`;train comfort 题池随机抽取 240 题;student n=4,max_tokens=4096;teacher n=4,max_tokens=4096,temp=0.6,top_p=0.95,top_k=20,seed=42。
- 第一阶段:240 题中 227 题至少有一条 student 错误 trajectory。
- 反事实条件:teacher 从 root 原题作答;或接收 student 错误 prefix 的 25%/50%/75%;另有 ignore-path control(看到草稿但要求忽略)。
- 全量 prefix sweep:Root `57.27%`;Path-25 `55.40%`;Path-50 `47.58%`;Path-75 `39.43%`;Root−Path-50 `+9.69pp`(bootstrap 95% CI `[+4.52,+15.09]`);Root−Path-75 `+17.84pp`(95% CI `[+12.03,+23.66]`)。前缀越深,teacher rescue 越差。
- 长度匹配 paired control:157 题同时拥有 student wrong/correct sample;wrong/correct/ignore prefix 使用完全相同 token 数。Root `67.20%`;wrong-path `55.57%`;correct-path `73.73%`;ignore-path `54.94%`。
- paired 差异:Root−wrong `+11.62pp`(bootstrap 95% CI `[+4.94,+18.47]`);correct−wrong `+18.15pp`(`[+11.78,+24.68]`);ignore−wrong `−0.64pp`(`[-4.78,+3.50]`)。说明错误路径相对正确路径显著更难被接管,同时额外草稿上下文污染是重要组成部分。
- 结论边界:该实验直接支持“teacher 在错误 student context 上的救援能力下降”,但还不能单独证明下降全部来自语义错误而非上下文污染。下一步需加 matched random-filler control,并进行 RFT-root / Root-KD / Path-OPD 的短程同预算 transfer 三臂实验。
- 产物:cloud=`outputs/cptr/student_240.jsonl`, `outputs/cptr/teacher_rescue_merged.jsonl`, `outputs/cptr/control5_merged.jsonl`, `outputs/cptr/control5_summary.json`, `outputs/cptr/control5_examples.json`;local=`reports/cptr/`。

## R104: CRPTR 同预算 transfer pilot — 机制方向复现,短训效应尚不显著(2026-09-08)

- 目的:把 R103 的 teacher rescue 差异推进到 student 学习层,区分完整 root trajectory 优势与错误 student state 伤害。
- 三臂:同一 `Qwen3-1.7B base`,同一 120 条共同 transfer examples,full-param lr=`1e-6`,40 optimizer steps,max sequence=`4096`;`RFT-root` 使用 root teacher verified completion,`Root-KD` 使用 root state teacher token KL,`Path-KD` 使用 matched wrong-prefix state teacher token KL。
- seed42 固定 val 1,024 @4096:no-think — RFT-root `25.293%` / semantic `28.418%`;Root-KD `26.270%` / `29.688%`;Path-KD `24.512%` / `28.027%`。
- seed43 使用同一 142 条候选集合 — RFT-root `26.953%` / `28.809%`;Root-KD `23.828%` / `26.074%`;Path-KD `25.195%` / `27.930%`。
- RFT-root − Path-KD strict 差异在两个 seed 均为正(`+0.78pp`,`+1.76pp`),pooled paired bootstrap 95% CI `[-0.54,+3.08]`;semantic pooled差异 `+0.64pp`,CI `[-1.17,+2.44]`。Root-KD 与 Path-KD 方向不稳定(`+1.76pp` / `−1.37pp`)。
- 结论:学习层 pilot 与 rescue 层方向一致,但 40 步单 seed/小 transfer 集合不足以给出显著的最终能力差异;当前可发表的强证据是 R103 的 Root−wrong `+11.62pp` 与 correct−wrong `+18.15pp` rescue margin。Root-KD 方差说明“root state”不自动等价于 RFT，监督目标/优化噪声仍需单独控制。
- 下一步:扩展 matched transfer token budget 或增加 seed;加入 random-filler control;最终比较 RFT-root、Root-KD、Path-KD 的同题学习增益与 function-space KL/token flip rate。
- 产物:`outputs/cptr_transfer/{rft_root,root_kd,path_kd}-40step*`、`reports/cptr_transfer/*-4096*`;local=`reports/cptr_transfer/`。

## R105: CRPTR random-filler control 与 120 步 transfer — 机制成立,离线学习 proxy 未闭环(2026-09-08)

- random-filler control:157 题,真实 wrong/correct/random prefix 使用相同 token 数;teacher n=4,max_tokens=4096。
- Teacher rescue:Root `65.92%`;random filler `64.01%`;wrong path `56.85%`;correct path `72.61%`。
- 题级差异:random−wrong `+7.17pp`,bootstrap 95% CI `[+0.48,+14.01]`;correct−wrong `+15.76pp`(`[+8.92,+22.61]`);Root−wrong `+9.08pp`(`[+2.55,+15.61]`)。Root−random `+1.91pp`(CI跨0),说明长上下文本身不是主要解释,wrong 语义路径额外造成损伤。
- 120 步 transfer pilot:三臂仍为同一 1.7B base,同一 120 examples,full-param lr=`1e-6`,max_seq=`4096`,no-think @4096评估。RFT-root `24.61%`,Root-KD `26.07%`,Path-KD `26.86%`。
- 解释边界:120 步离线 Path-KD 反超 RFT-root,与40步结果方向不稳定;该 proxy 把静态错误 prefix 与 teacher continuation 固化为训练输入,并不等价于 verl 的 on-policy OPD,因此不能用于否定 R103 的状态救援机制。
- 当前结论升级为:wrong student state 会使 teacher rescue 显著下降,且超出纯上下文长度效应;但要证明 RFT 在真实训练中稳定优于 OPD,必须使用标准 on-policy OPD 重新生成 student rollout,并做同 prompt pool、同 teacher budget、同 seed 的 matched RFT/OPD 实验。
- 产物:cloud=`outputs/cptr/random_control_*`, `outputs/cptr/cptr_transfer/*-120step`;local=`reports/cptr/`, `reports/cptr_transfer/`。

## R106: 标准 on-policy OPD 拓扑 smoke — Ray 环境与 4 卡 hybrid 资源池修复(2026-09-08)

- 目标:在与 R103 机制假设一致的条件下，使用 verl 真正的在线 rollout 训练闭环，避免离线 Path-KD proxy 代替 OPD。训练入口为 `scripts/run_grpo_vanilla_opd.sh`，学生为 1.7B base，教师为固定 4B-GRPO checkpoint-2。
- 首次 smoke 失败原因不是模型或显存，而是 Ray worker 未继承项目内 `verl-v0.8.0-src` 的 `PYTHONPATH`，导致 worker 报 `ModuleNotFoundError: verl.trainer.distillation`。入口脚本现将完整源码与 vLLM 环境路径注入 `ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH`；远端旧脚本保留为 `.pre_ray_pythonpath.bak`。
- 第二次失败暴露资源池重复申请:学生 3 卡 + 教师 1 卡之外，`rollout.nnodes=1` 又申请了 1 卡。确认当前容器已切为 Default compute mode 后，将 rollout 改为 `nnodes=0`，使 3 个 vLLM rollout replica 与学生 GPU 0–2 hybrid 复用，教师独占 GPU 3。
- smoke5 配置:train pool 120 题，batch=8，rollout `n=6`，temperature=0.6、top-p=0.95、top-k=20，max response=4096，student/teacher max model len=5121，full-param FSDP，lr=`1e-6`，KL=`0.05`，3 steps。结果 3/3 完成，step 用时约 `66.4s/43.8s/59.5s`，三份 rollout 与 global_step_3 checkpoint 均落盘，无 Traceback。
- 运行时证据:GPU 0–2 在 rollout/update 阶段约 82–90% 利用率，GPU 3 教师推理阶段约 50–57%；指标 `overlap_ratio≈0.734`、`actor/kl_coef=0.05`、`response/aborted_ratio=0`，说明 3+1 hybrid 拓扑和在线教师服务均真正工作。
- 下一步:在不改变其余协议的前提下启动正式 120-step 标准 on-policy OPD，单独写入 `outputs/cptr_standard_opd_120step`、对应 TensorBoard run 和 checkpoint 目录；完成后与 RFT-root、Root-KD、Path-KD 做同题统一评估。

## R107: 标准 on-policy OPD 正式训练启动(2026-09-08)

- smoke5 通过后清空旧 Ray/vLLM 进程，在同一云端 4×A100 节点启动正式 120 steps。实际生效配置为 `rollout.nnodes=0`、学生 3 卡 hybrid rollout、教师 1 卡 standalone，学生与教师模型、120 题池和 R106 其余参数保持一致。
- 第 1 步已完成，耗时约 `68.6s`；截至 step 12，step 8–12 用时约 `47.5/34.4/94.2/31.7/44.4s`，其中 step 10 的额外时间来自 checkpoint 保存。`outputs/cptr_standard_opd_120step/rollouts/1.jsonl` 已落盘，首个 `global_step_10` 已保存，无 Traceback，当前 ETA 约 1.5–2 小时。训练过程每 10 步保存 checkpoint，正式日志为 `logs/cptr_standard_opd_120step.log`。
- 首步监控:教师 vLLM GPU 3 正常工作，学生 GPU 0–2 正常参与 rollout/update；后续重点观察 `critic/score/mean`、`actor/distillation/loss`、`actor/distillation/overlap_ratio`、`actor/entropy`、`response_length/clip_ratio`、`timing_s/step` 和 checkpoint mtime。

## R108: 远端会话中断与 checkpoint 续跑(2026-09-08)

- 正式 run 首轮推进到 step 15 后，所有 Ray actor 在同一时刻被清理，日志没有当前 OOM、Xid 或 Python Traceback，且 driver 记录为 `ray.shutdown()`；因此该轮不能视为完成。step 10 的 checkpoint tracker 完整，step 11–15 rollout 与日志已备份为 `outputs/cptr_standard_opd_120step_interrupted_step15` 和 `logs/cptr_standard_opd_120step_interrupted_step15.log`。
- 诊断结论:这次是外部会话/driver 生命周期终止，不能归因于训练数值或显存。此前直接把 `nohup` 任务挂在 SSH 命令 shell 上，长期运行仍可能被平台清理；本轮改为 `nohup setsid ... </dev/null`，从 `latest_checkpointed_iteration=10` 自动续跑。
- 续跑已重新到 step 11/120，GPU 0–2 与 GPU 3 均重新工作；后续监控目标是跨过原中断时间点，并确认 `global_step_20`、TensorBoard event 持续增长。正式产物仍使用原目录，旧中断版本独立保存。

## R109: 训练步数截断修复与第三次续跑(2026-09-08)

- 对 R108 的两次“step 15 结束”重新核查后，原因确定为训练池 120 题、batch 8 形成 15 steps/epoch，而入口固定 `trainer.total_epochs=1`；verl 外层 epoch 循环结束，故不是会话回收、OOM 或 GPU 崩溃。
- `scripts/run_grpo_vanilla_opd.sh` 已改为 `TOTAL_EPOCHS=${TOTAL_EPOCHS:-1000}`，正式命令中实际记录为 `trainer.total_epochs=1000`，由 `trainer.total_training_steps=120` 控制终止。前两次中断产物分别独立备份，step 10 checkpoint 保持不变。
- 第三次续跑从 tracker=10 重新启动，已连续通过 step 11–17 并跨过原先的 15-step epoch 边界；step 11–12 用时约 `48.6s/42.1s`，后续约 45 秒/步，当前进度 `17/120`。学生 GPU 0–2 与教师 GPU 3 正常工作，说明 `TOTAL_EPOCHS=1000` 修复有效。下一验证点为 step 20 checkpoint。

## R110: 报告 v5.4 附录重构与图表修正(2026-09-08)

- 新建 `reports/AGOPD_paper_v5_4_working_20260908.html`，原始 v5.3 文件保持不变。报告主线由“教师优势随能力变号”修正为“题目级教师结果优势与状态级错误路径风险的区分”，因为正确 chat-template 重测显示教师在五个能力区间均保持非负优势。
- 正文新增 CRPTR 状态救援证据，并将主实验表格编号顺延：反事实状态表为表 2，主四臂结果为表 3，多阶段结果为表 4；Pass@k 统一为图 6。旧的 +3.3/+5.6/−15.8pp、图 5.2b 和“教师仅在低能力区间更强”等残留表述已清理。
- 附录 A 重构为普查协议、训练池来源与实验边界、跨规模诊断；附录 B 补充配置筛选表；附录 C 补充训练指标定义；附录 D 区分 RFT 检查点与合并初始化；附录 E 分离预算、思考模式和继续 RL；新增附录 F 收纳 CRPTR 设计、结果和 RFT/Path-KD transfer 边界。
- 新增图表脚本与产物：scripts/make_corrected_teacher_map.py 生成 reports/figures/fig2_teacher_map_corrected.png，scripts/make_cptr_rescue_figure.py 生成 reports/figures/figF1_cptr_rescue.png。两张图使用统一英文图内文字、项目配色和无网格版式，并已实际检查无标注重叠。
- HTMLParser 结构检查通过，6 个附录均可解析，新增图表路径存在，原 v5.3 与 v5.4 为不同文件。待浏览器目检和云端 on-policy OPD 完成后回填最终结果。

## R111: 模型角色冻结与报告证据归位(2026-09-08)

- **角色定义冻结**: 报告统一使用 \(S_0\) 表示未经后训练的 Qwen3-1.7B Base，\(S_{\mathrm{RL}}\) 表示同一底座经纯 GRPO 得到的学生检查点，\(T\) 表示固定 Qwen3-4B-GRPO 教师。第 3 节的能力诊断只比较 \(S_0\) 与 \(T\)，第 5 节的主方法比较才进入训练后的检查点。
- **Base 重测**: 使用与教师相同的 300 题抽样序列、五个能力桶、chat-template、temperature=0.6、top-p=0.95、top-k=20、每题 n=8、seed=42，完成 Base 复测。五桶 Base 准确率为 4.17%、17.29%、37.92%、53.96%、70.00%；教师为 13.75%、41.88%、62.92%、79.58%、86.25%；教师优势为 +9.6、+24.6、+25.0、+25.6、+16.3pp。旧 GRPO 学生地图不再用于第 3 节。
- **复现工件**: 新增 scripts/remap_base_capability.py 与 scripts/build_base_teacher_map.py；保留 reports/figures/remap_base_all.json、remap_teacher_all.json、teacher_base_capability_map_corrected.json 及图 2 的 PNG/SVG/PDF 输出。
- **第 3 节扩展**: 删除旧的 §3.4 结果重复；在 §3.3 中正式定义 CRPTR 的四种状态、匹配前缀控制、状态条件结果估计量及 transfer probe 的解释边界。CRPTR 汇总仍为 root 65.92%、random 64.01%、wrong 56.85%、correct 72.61%。
- **附录归位**: 删除正文 §5.5，将 Stage-1→Stage-2A→Stage-2B→Stage-2C 流程与谱系表移入附录 D，删除 D.3 与重复谱系表；附录 F 改为三个真实 CRPTR trajectory exhibit，展示数据来自 reports/cptr/control5_examples.json。
- **图表重做**: 新增 scripts/redesign_report_figures.py，生成图 4、图 5、图 6、图 D2 的 PNG/SVG/PDF。图 4 使用等尺寸双面板；图 5 显式标记 (a)/(b)；图 6 采用观测值附近的纵轴；图 D2 使用带参照线的 RFT 检查点曲线。表 3 增加固定列宽与换行规则，附录 A.1 右栏残留内容删除。
- **写作/绘图工作流**: 本地安装并采用 nature-figure、scientific-visualization、science-plotting、research-writing-skill、writing-in-the-sciences 五组规则，统一执行证据合同、变量/样本/不确定性声明、可访问配色、矢量导出与视觉 QA。原始报告 v5.3 不修改。

## R112: 报告批次收尾与云端训练状态更新(2026-09-08)

- 云端标准 on-policy OPD 已完成 120/120，训练时长约 1:22:30，global_step_120 checkpoint 已保存，日志无 Traceback；当前只标记为待统一评估，不将其结果提前写入主结论。
- 报告 v5.4 通过 HTMLParser 结构检查：标签闭合、附录层级、图表外链、主结果表 6 列布局与真实轨迹表 3 行均通过；旧 v5.3 保持不动。

## R113: 补充科学写作工作流(2026-09-08)

- 通过清理本机 GitHub 代理配置后，补充安装 writing-in-the-sciences，形成“研究论文结构审阅 + 句段改写 + 图表证据审计”的写作工作流。当前本地可用的科研工作流包括 nature-figure、scientific-visualization、science-plotting、research-writing-skill 和 writing-in-the-sciences。

## R114: 最终一致性审计与 Pass@k 数据回溯(2026-09-08)

- Pass@k 重绘过程中发现初版脚本把 Base 与 GRPO 的 (k=1) 值写反；已按归档日志恢复 Base/GRPO/OPD/RL+OPD/AGOPD 的四个 k 值，并新增 reports/figures/passk_five_methods.json 作为独立源数据。
- 旧的 make_corrected_teacher_map.py 已改为兼容入口，统一调用 Base—teacher join 实现，避免再次生成 GRPO 学生旧地图。
- 最终报告审计通过：HTMLParser 无标签错误；主文小节编号连续至 5.5；D.3、§5.5 旧正文块和 A.1 右栏内容均不存在；5 个外链图和 9 个表格可解析，3 个真实轨迹案例可读取。全页 headless 浏览器截图因原报告大量内嵌图像加载超时未作为通过条件，单图视觉检查已完成。

## R115: 图 1 版式基准与逐案例轨迹重排(2026-09-08)

- 用户反馈指出新图标题相对图 1 过大、加粗且缺少统一的图例语义。已将图 2、图 4、图 5、图 6、图 D2 的图内标题统一为居中、小字号、常规字重，并去除背景网格；D2 图例明确青绿色实线为 RFT-SFT 检查点，红色虚线为 GRPO 参照。
- 附录 F 原四列横向轨迹表可读性不足，已改为三个独立案例块。每个块固定呈现“Base 学生错误前缀 → 教师错误状态续写 / 正确状态续写 → 验证器结果 → 案例解释”，并保留更长的真实文本摘录；完整原文仍以 CRPTR JSON 工件为准。
- 本轮重新执行图表生成和脚本编译，单图视觉检查通过；报告 HTML 结构与轨迹案例 DOM 结构通过检查。

## R116: 图内字号复核与轨迹案例可读性修正(2026-09-08)

- 用户复核后确认，前一版图表虽然数据正确，但标题层级与图 1 不一致：标题偏大、加粗、部分左对齐，且新图保留了不必要的背景网格。已将图 2、图 4、图 5、图 6、图 D2 的标题统一为居中、小字号、常规字重，并关闭背景网格。
- D2 的青绿色实线现明确标记为“RFT-SFT checkpoint”，红色虚线标记为“180-step GRPO reference”；正文图注同步说明二者含义。
- 附录 F 的横向四列表会迫使读者跨列重建因果链，已改为三个独立案例块。每块固定呈现题目元数据、Base 错误前缀、教师错误状态续写、教师正确状态续写、验证器结果和案例解读；每个案例保留三段较长轨迹摘录。
- QA: 5 个新图重新生成并逐图检查；HTMLParser 无错误，3 个案例块、9 个 pre 轨迹段、5 个外链图均可解析；脚本编译通过。

## R117: v5.4 图片内嵌(2026-09-08)

- 为便于网页端 GPT 直接读取和修改，已将 v5.4 中全部 12 张图片转换为 PNG data URI 并嵌入 HTML；报告不再依赖同目录下的 figures 外链文件。
- MathJax CDN 脚本仍保留，用于公式渲染；HTMLParser 检查通过，原始 v5.3 未修改。

## R118: RFT-root 与标准 on-policy OPD 统一评估闭环(2026-09-08)

- 按统一计划，在云端补跑严格 RFT-root：Qwen3-1.7B Base、同一 120 题 transfer pool、batch=8、120 optimizer steps、lr=1e-6、max sequence=4096、seed=42。此前的 batch=1 RFT-root 结果只保留为辅助实验。
- 云端标准 on-policy OPD 使用已完成的 global_step_120，并与严格 RFT-root 在同一固定 1,024 题、chat-template、temperature=0.6、top-p=0.95、top-k=20、max_new_tokens=4096 下并行评估。
- 严格结果验证：OPD 263/1024=25.68%，RFT-root 263/1024=25.68%；逐题配对 RFT wins=107、OPD wins=107，差异 0.00pp，bootstrap 95% CI [-2.83,+2.83]，McNemar p=1.0。
- 语义答案验证：OPD 265/1024=25.88%，RFT-root 292/1024=28.52%；RFT-root−OPD=+2.64pp，paired bootstrap 95% CI [-0.10,+5.37]，McNemar p=0.0664。该结果提供方向性支持，但不能宣称 RFT 显著优于 on-policy OPD。
- 严格格式契约：RFT-root 低 4.79pp，原因是其输出更多使用 \boxed{} 而非 Answer: 行；该差异属于格式遵循指标，不能作为数学正确性结论。逐题文件保存在 reports/eval-dapo-v1/cptr_final/，比较脚本为 scripts/compare_cptr_rft_opd.py。

## R119: 附录 F 真实轨迹与 same-state probe 采集启动(2026-09-09)

- **实验目标**: 将附录 F 从“人为错误前缀深度的答对率”重构为真实 Student trajectory 的状态级审计：先定位首次可验证语义错误，再在完全相同的 `user prompt + assistant prefix` 状态上比较 Student/Teacher 的 top-16 分布，并从该状态分别采样 continuation 估计 Teacher rescue rate。
- **正式 Student 采样**: 本地使用 Qwen3-1.7B Base、固定 `data/dapo-verl-v1/val.parquet`、200 prompts × 8 trajectories、temperature=`0.6`、top-p=`0.95`、top-k=`20`、seed=`42`、max_tokens=`2048`。结果为 1,600 条真实轨迹，其中 315 条结果正确、1,285 条结果错误；86 道题同时出现正确和错误轨迹，满足同题配对的初筛条件。原始数据为 `reports/appendix_f_state_probe/student_formal200.jsonl`。
- **语义审计闸门**: 对 86 条同题配对的各一条错误轨迹生成结构化审计候选。自动审计得到 86 条 proposal；仅 19 条同时满足“数学/解释类错误”与“原文 quote 精确匹配”，30 条因属于答案抽取或格式问题被排除，36 条因 quote 无法匹配被排除，1 条 JSON 解析失败。所有自动 proposal 均标记为 `candidate_requires_semantic_review`，不得直接作为论文统计。
- **same-state pilot**: 对 3 条经 Codex 人工复核的高置信错误轨迹，按 root、错误点前后等 6 个位置保存 Student/Teacher HF top-16 logits；并在 root、错误起点、错误后 64 token 三个完全相同的 assistant-prefix 状态上，对两模型各采样 8 次 continuation。工件位于 `reports/appendix_f_state_probe/`，包括 `same_state_probe3.jsonl`、`student_same_state3.jsonl`、`teacher_same_state3.jsonl` 与 `pilot_summary.json`。
- **pilot 结果边界**: 3 个样本的 Teacher rescue rate 为 root=`16.7%`、错误起点=`33.3%`、错误后 64 token=`33.3%`，Student 分别为 `20.8%`、`16.7%`、`8.3%`；方向不稳定，仅证明采集、同状态 token 化、top-16 logits 和结果验证链路可运行，不能支持错误路径机制结论。top-16 entropy 只作解释变量，不能单独当作正确性证据。
- **实现**: 新增 `scripts/collect_state_probe.py`（Student 采样、审计队列、HF top-k probe）、`scripts/audit_error_onset_llm.py`（候选审计器）、`scripts/materialize_audit_proposals.py`（语义候选过滤与 token offset 映射）、`scripts/prepare_manual_probe_pilot.py`（人工复核 pilot）、`scripts/generate_same_state_continuations.py`（同状态续写）和 `scripts/summarize_state_probe.py`（按 prompt 聚类汇总）。v6.0 报告未修改。
- **下一步**: 对 19 条候选逐条进行独立语义复核，剔除“模型认为错误但实际正确”以及“仅最终答案/格式错误”的样本；完成后对 reviewed subset 扩展 same-state probe。附录 F 的核心统计必须来自 reviewed subset，现有 prefix-depth sweep 与四组合 entropy 表降级为历史诊断，不承担新的机制结论。

## R120: 附录 F 同题正确/错误状态配对 pilot 完成(2026-09-09)

- **配对构造**: 从 R119 的 86 道同题正/错候选中，先人工复核 4 条高置信错误 trajectory，并将每条与同题的一条正确 Student trajectory 配对。错误路径的首次错误 token offset 作为正确路径的 matched anchor，避免把不同题目或不同输出长度混为状态差异。
- **same-state probe**: 对 8 条轨迹（4 wrong + 4 matched correct）在 root、错误锚点前 64/16、锚点、锚点后 16/64 共 48 个状态读取 Student/Teacher HF top-16 logits；每个状态保存 top-16 token、logprob、top-16 mass 与条件化 entropy。相同状态定义为 `chat_template(user) + assistant_prefix`，不再插入 `<student_draft>` 额外指令。
- **same-state continuation**: 在 root、first-error anchor、+64 tokens 三个位置，对 Student 与 Teacher 分别以 n=8、temperature=`0.6`、top-p=`0.95`、top-k=`20`、seed=`42`、max_tokens=`1024` 生成，共每模型 192 条续写；结果由同一数学 verifier 判定。
- **pilot 结果**: 错误 trajectory 上 Teacher rescue 为 root=`53.1%`、first error=`40.6%`、+64=`3.1%`；同题 matched correct trajectory 为 `53.1%`、`71.9%`、`50.0%`。按题目配对，Teacher wrong-minus-correct 差异为 first error `-31.25pp`，bootstrap 95% CI `[-68.75,+6.25]`；+64 为 `-46.88pp`，CI `[-93.75,0]`。Student 对应 first error `-37.50pp`、+64 `-31.25pp`。方向符合状态错位假设，但 n=4、区间较宽，只作为附录的 pilot 证据。
- **熵解释**: 错误起点处 Teacher top-16 entropy 的平均值低于同题正确控制，提示部分样本是“低熵地延续错误状态”，但该现象不能单独证明错误；论文图表必须与 continuation rescue 和人工错误标签一起展示。
- **图表工件**: `scripts/plot_appendix_f_probe.py` 生成 `reports/figures/appendix_f_same_state_pilot.png/.svg`，保留每题原始点、均值、wrong/correct matched 线型和 (a)–(d) 面板；源数据为 `reports/appendix_f_state_probe/matched8_summary.json`。v6.0 报告仍未修改。
- **边界与下一步**: 当前 4 条人工复核轨迹用于验证完整证据链和附录案例展示，不足以支撑总体机制结论。下一步优先独立复核其余语义候选，扩展至至少 20–30 个 prompt-level pairs，再重画最终 Figure F 与表 F1；格式/重复错误继续作为排除性审计样本单列。

## R121: 附录 F 五对真实状态配对与 v6.1 报告副本(2026-09-09)

- **数据扩展**: 在 200 题 × 8 的真实 Student sampling 基础上，新增并人工复核第 5 条高置信错误 trajectory（抛物线格点直线题：将 36 个无序因子对重复计数为 72）。最终 reviewed set 为 5 个 prompt-level pairs、10 条 trajectory、60 个 top-16 probe 状态和每模型 240 条 continuation。
- **同题配对结果**: Teacher 在错误 trajectory 上的 rescue rate 为 root=`37.5%`、first-error anchor=`45.0%`、+64 tokens=`7.5%`；同题 matched correct control 为 `45.0%`、`70.0%`、`52.5%`。Teacher wrong-minus-correct 在 first-error anchor 为 `-25.0pp`，prompt-bootstrap 95% CI `[-62.5,+7.5]`；+64 tokens 为 `-45.0pp`，CI `[-85.0,-5.0]`。错误点处区间仍覆盖零，因此只作方向性 pilot；+64 结果显示错误状态持续后差异扩大，但不能外推为总体效应。
- **图表规范**: 以项目统一 `scripts/figstyle.py` 为唯一样式源，新增 `scripts/plot_appendix_f_probe.py` 输出无图内大标题的四面板数据图；坐标/刻度字号与图 1 对齐，主线约 1.8pt、原始点保留，panel label 与图题/图释置于 HTML 图下，导出 PNG/SVG/PDF。视觉检查已完成，legend 不遮挡关键数据。
- **报告重写**: 新建 `reports/AGOPD_paper_v6_1_state_probe_20260909.html`，原 v6.0 不修改。附录 F 现在按“真实轨迹普查 → 首次语义错误 → exact same-state top-16 probe → continuation rescue → prompt-level paired uncertainty → 解释边界”组织；删除旧版以固定前缀深度和四组合 entropy 作为核心证据的叙事，并保留 5 个真实案例的原文片段。
- **核心解释**: 结果支持“错误 Student state 会降低 Teacher 的后续可恢复性，且错误持续后损伤扩大”的方向性机制；不能写成“Teacher 在所有错误状态上都无法救回”，也不能从 top-16 entropy 单独推出全局模型坍塌。训练层 collapse 仍须由逐 step reward、entropy、length 和 abort 曲线另行证明。
- **QA**: v6.1 公式中的反斜杠字符已修复，F 区域不含制表符或未转义 `<student_draft>` 标签；F 区域包含 1 个图、4 个表、5 个真实案例块和 4 个新内嵌 PNG。HTML 结构检查通过，v6.0 与 v6.1 SHA-256 不同且 v6.0 源文件保持存在。
- **下一步**: 继续独立复核剩余候选并扩展到至少 20–30 个 prompt-level pairs；扩展数据完成后，重新估计 first-error 与 +64 的配对差异，并将当前 v6.1 的 pilot 标注升级或降级为正式附录证据。

## R122: 修正 root 配对复用后的 Appendix F 数值(2026-09-09)

- **问题与修复**: wrong/correct matched rows 的 root 状态完全相同，但旧续写脚本对同一 root state 发起两次采样，request-level seed 可能造成非零的伪差异。`scripts/generate_same_state_continuations.py` 现按 `(prompt_id, state, prompt_token_ids)` 去重，对完全相同状态只采样一次，再把结果复用给两个 trajectory alias。
- **修正后的结果**: 5 对 prompt-level pairs、每模型 240 条 alias continuation。Teacher 在 wrong trajectory 的 rescue 为 root=`45.0%`、first-error=`42.5%`、+64=`10.0%`；matched correct control 为 `45.0%`、`77.5%`、`50.0%`。Teacher wrong-minus-correct 为 root=`0.0pp`、first-error=`-35.0pp`（prompt-bootstrap 95% CI `[-65.0,-5.0]`）、+64=`-40.0pp`（`[-80.0,0.0]`).
- **解释**: root 差异被严格消除；错误锚点处 Teacher 恢复率已低于同题正确状态，错误继续 64 token 后差异进一步扩大。由于 reviewed set 仍只有 5 个经选择的 prompt，以上仍属于方向性 pilot，不能替代更大规模的随机/分层审计。
- **工件同步**: `matched10_summary.json`、`figF_state_*` 四面板图和 `AGOPD_paper_v6_1_state_probe_20260909.html` 已按去重后的数据重生成；v6.0 源文件保持不变。旧 R121 数字作为历史运行记录保留，但论文引用以 R122 修正值为准。

## R123: 附录 F 扩展至八对 prompt-level pairs(2026-09-09)

- **数据合并**: 将 5 对 formal pairs 与早期 3 条 high-confidence pilot 按同一协议合并，得到 8 个 prompt-level pairs、16 条 trajectory、96 个 top-16 probe 状态，以及每模型 384 条 alias continuation。完全相同的 root state 只采样一次并复用给 wrong/correct alias。
- **更新结果**: Teacher wrong trajectory 的 rescue 为 root=`32.8%`、first-error=`40.6%`、+64=`18.8%`；matched correct control 为 `32.8%`、`62.5%`、`54.7%`。配对 wrong-minus-correct 为 root=`0.0pp`、first-error=`-21.88pp`（prompt-bootstrap 95% CI `[-46.88,0.00]`）、+64=`-35.94pp`（`[-73.44,-6.25]`）。该结果比 5 对版本更稳定，但仍是选择性 single-seed pilot，不能视为 200 题总体估计。
- **报告内容**: v6.1 附录 F 已同步扩展为 8 个真实案例，表 F2/F3/F4、四面板 Figure F1、案例展示和解释链条全部来自 `matched16_summary.json` 及对应原始 JSONL；v6.0 保持不变。
- **图表口径**: 图 F1 使用图 1 同源 `figstyle.py`：图内无大标题，轴/刻度/线宽统一，面板标签和图题/图释置于图下，保留 prompt-level 原始点；PNG 嵌入 HTML，SVG/PDF 保存为排版工件。
- **解释边界**: 八对数据支持“真实错误状态后的 Teacher 可恢复性低于同题正确状态，且在错误持续后差异扩大”的方向性论断；不能据此宣称所有 Teacher 都无法救回，不能把低 entropy 等同于错误，也不能把 trajectory-level irrecoverability 等同于训练过程的全局 collapse。

## R124: 附录 F v6.2 图表与真实轨迹可读性修订(2026-09-09)

- **图表对齐**: 用户指出组合图的尺寸、位置和图题层级未与图 1 统一。已将 Figure F1 固定为与正文图表相同的 `860px` 内容宽度，四面板使用等尺寸组合画布，共享图例，图题/图释统一置于图下并改为居中、小号常规字重；原始 PNG、SVG、PDF 均保留。
- **编号和表意**: 附录 F1 的案例编号改为 `Case 1`–`Case 8`，删除 UUID 等不可读索引；状态表统一使用 Root、First-error anchor 和 +64 tokens，并在 F5 增加指标字典，解释每个指标的操作性定义、统计单位和不可支持的过度解释。
- **公式排版**: F2 将 top-16 条件熵和 top-16 内部归一化分别放在两个独立公式块中，编号为式 (F.1) 与式 (F.2)，每个公式下方配单位和解释边界，避免在表格或行内混排导致的显示歧义。
- **真实 trajectory 展示**: F3 不再以原始 `<pre>` 直接堆叠 Markdown/LaTeX。网页层保留真实文本并进行最小呈现格式化：移除控制性的 `<think>` 标签，合并 display-math 块交由 MathJax 渲染，识别小标题、分隔线、列表和答案行；每个 Case 固定为五个证据格，读者可以沿着错误前缀、错误单元、匹配正确状态和 Teacher 续写顺序阅读。完整原文仍在 `error_audit_reviewed8.jsonl` 与 continuation JSONL 中。
- **当前证据**: v6.2 使用 8 个 prompt-level pairs。Teacher wrong rescue 为 `32.8% / 40.6% / 18.8%`，matched-correct 为 `32.8% / 62.5% / 54.7%`，root/first-error/+64 的配对差为 `0.00 / -21.88 / -35.94 pp`；该结果仍只承担方向性 pilot 证据，不外推为 200 题总体效果。
- **验证**: `scripts/rewrite_appendix_f_v6_1.py` 编译并成功重生成 v6.2；静态检查确认 8 个 Case、5 张表、1 个内嵌 PNG、2 个公式编号、40 个 trajectory 格和无 UUID/`<pre>`/未转义控制标签。原 v6.0 SHA-256 未改变。

## R125: 附录 F 二次版式与公式兼容性修订(2026-09-09)

- **错位修复**: 实际视觉检查发现 F3 的五格结构会留下孤立的最后一格，且轨迹框高度随摘录长度变化。现改为每个 Case 六格固定高度布局，并新增“本 Case 的证据判读”格；所有轨迹框在网格中等高对齐。
- **乱码修复**: F3 不再将原始 TeX 定界符和 Markdown 标记直接输出到网页。展示层移除 `<think>` 控制标签、清理 `$...$`/`$$...$$`/环境标记、转换常用运算符和小标题；这只影响可读展示，原始输出仍由 JSONL 保留。F2 的两条核心公式进一步改为带上下标的语义 HTML，不依赖 MathJax 才能显示。
- **图 F1 重绘**: 组合图重新生成，左右边距对称、面板绘图区一致、全局图例居中；HTML 仍只嵌入这一张 PNG，同时保留 SVG/PDF 排版文件。
- **QA**: v6.2 重新生成成功；F 区域静态检查为 8 个 Case、5 张表、1 张内嵌图、2 个独立公式块、48 个证据格，无 `<pre>`、UUID、未转义控制标签或 F2 display-math 原始定界符。图 F1 已完成视觉检查。

## R126: 附录 F 可见区域源码定界符清理(2026-09-09)

- **问题复核**: 再次逐项检查后发现，部分题目文本仍含 `\(...\)`，在 MathJax 未加载时会直接显示为源码；这正是网页端残留“乱码”的来源之一。
- **修复**: `plain_math_text` 现在同时清理 display/inline TeX 定界符，并处理常用数学命令；F2 的两条定义继续使用独立的语义 HTML 公式块，F3 的展示层不依赖外部公式脚本。
- **最终检查**: v6.2 重新生成成功，F 区域的 `$`、`\(...\)`、`\[...\]`、`<pre>` 均为 0；8 个 Case、5 张表、1 张内嵌组合图和 2 个公式块均保留，原始轨迹数据未改写。
