#!/usr/bin/env bash
# 健壮版:curl 断点续传 + 无限重试(代理不稳定,时通时断)
MODEL_DIR="${AGOPD_ROOT}/models/OpenMath-Nemotron-7B"
PROXY="http://<PROXY_HOST:PORT>"
REPO="nvidia/OpenMath-Nemotron-7B"

dl() {
  local shard="$1"
  local out="$MODEL_DIR/$shard"
  local url="https://huggingface.co/$REPO/resolve/main/$shard"
  # 期望大小(从已有部分推算:目标 0 就下载到 curl 完成;简单起见循环到文件存在且 >1GB)
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    curl -sL -C - --retry 3 --retry-delay 5 -x "$PROXY" -o "$out" "$url" 2>>/tmp/curl_err_$shard.log
    local size=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$size" -gt 7000000000 ]; then
      echo "DONE_OK: $shard ($size bytes) after attempt $attempt"
      break
    fi
    echo "RETRY $shard attempt=$attempt size=$size"
    sleep 8
  done
}

dl model-00001-of-00002.safetensors &
dl model-00002-of-00002.safetensors &
wait
echo "ALL_SHARDS_DONE"
