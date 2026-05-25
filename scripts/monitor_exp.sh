#!/bin/bash
cd /root/DeepResearcher
count=0
while [ $count -lt 20 ]; do
  gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null)
  gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null)
  signal=$(cat /root/DeepResearcher/outputs/verl_examples/gsm8k/signal/signal.json 2>/dev/null || echo "no-signal")
  swanlab_runs=$(ls -t /root/DeepResearcher/swanlog/ 2>/dev/null | head -1)
  processes=$(ps aux | grep -E "ray::WorkerDict|main_ppo" | grep -v grep | wc -l)
  echo "[$(date +%H:%M:%S)] GPU_util=${gpu_util}% GPU_mem=${gpu_mem}MiB signal=$signal workers=$processes swanlab_new=$swanlab_runs"
  count=$((count+1))
  sleep 30
done
