"""组装 Frontier 训练集:全量 p̂ 普查(旧 4000 + 新 12164)合并 → frontier 桶
→ 按原 train.parquet 格式输出(供 verl GRPO 直接使用)。
"""
import json
from pathlib import Path

import pandas as pd

OLD_BUCKETS = [
    "outputs/coldstart-17-4000-shard0/buckets.jsonl",
    "outputs/coldstart-17-4000-shard1/buckets.jsonl",
]
NEW_PARQUETS = sorted(Path("outputs/p_hat_survey").glob("p_hat_shard*.parquet"))
TRAIN = "data/dapo-verl-v1/train.parquet"
OUT = "data/dapo-verl-v1/train_comfort.parquet"

# 1. index → (p_hat, bucket)
mapping = {}
for f in OLD_BUCKETS:
    for line in open(f):
        d = json.loads(line)
        mapping[d["index"]] = (d["p_hat"], d["bucket"])
for f in NEW_PARQUETS:
    df = pd.read_parquet(f)
    for _, r in df.iterrows():
        mapping[str(r["index"])] = (float(r["p_hat"]), r["bucket"])

print(f"mapped prompts: {len(mapping)}")

# 2. 全量 train + 过滤 frontier
train = pd.read_parquet(TRAIN)
print(f"train rows: {len(train)}")


def get_index(r):
    ei = r["extra_info"]
    if isinstance(ei, dict) and "index" in ei:
        return str(ei["index"])
    return None


train["_idx"] = train.apply(get_index, axis=1)
matched = train["_idx"].isin(mapping)
print(f"matched by index: {matched.sum()} / {len(train)}")

train["_p_hat"] = train["_idx"].map(lambda i: mapping.get(i, (None, None))[0])
train["_bucket"] = train["_idx"].map(lambda i: mapping.get(i, (None, None))[1])

# 3. Frontier 桶(0.125 ≤ p_hat ≤ 0.75)
# 舒适区:frontier 桶去掉 0.125 边缘档(初训不搞太难)
frontier = train[(train["_bucket"] == "frontier") & (train["_p_hat"] >= 0.25)]
print(f"frontier rows: {len(frontier)}")
hard = train[train["_bucket"] == "hard"]
easy = train[train["_bucket"] == "easy"]
print(f"easy: {len(easy)} hard: {len(hard)} frontier: {len(frontier)} "
      f"(unmatched: {(~matched).sum()})")

# p_hat 分布(供图)
print("comfort p_hat distribution:")
print(frontier["_p_hat"].value_counts().sort_index().to_string())

# 4. 输出(去掉辅助列,保留原格式)
cols = [c for c in train.columns if not c.startswith("_")]
frontier[cols].to_parquet(OUT, index=False)
print(f"WROTE {OUT}: {len(frontier)} rows")

# 可选:easy 少量混合版本(留作对照,暂不输出)
