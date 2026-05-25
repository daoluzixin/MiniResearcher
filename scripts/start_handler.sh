#!/bin/bash
set -euo pipefail

export PATH="/home/vipuser/miniconda3/bin:$PATH"
export PYTHONPATH="/root/DeepResearcher:${PYTHONPATH:-}"
export HF_HOME="/root/.cache/modelscope/hub"

cd /root/DeepResearcher

LOG_DIR="/root/DeepResearcher/logs/handler"
mkdir -p "$LOG_DIR"
mkdir -p /root/DeepResearcher/outputs/verl_examples/gsm8k/signal

echo "========================================"
echo "Starting Search Handler at $(date)"
echo "========================================"

PYTHON_BIN="/home/vipuser/miniconda3/bin/python"
LOG_FILE="${LOG_DIR}/handler.log"

# 使用 tee 同时输出到文件和控制台
"$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '/root/DeepResearcher')
import json
import yaml
import threading
import time
from scrl.handler.handler import Handler
from openai import OpenAI
from types import SimpleNamespace

# 加载 agent config
with open('/root/DeepResearcher/scrl/handler/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 确保缓存目录存在
import os
os.makedirs('/root/DeepResearcher/scrl/handler/cache', exist_ok=True)

agent_config = {
    'query_save_path': '/root/DeepResearcher/scrl/handler/cache/search_result.json',
    'search_engine': config.get('search_engine', 'duckduckgo'),
    'search_top_k': config.get('search_top_k', 10),
    'search_region': config.get('search_region', 'us'),
    'search_lang': config.get('search_lang', 'en'),
    'server_url_list': config.get('server_url_list', []),
}

# handler config - 与 generation.py 中的配置保持一致
handler_config = SimpleNamespace(
    data_writing_file='/root/DeepResearcher/outputs/verl_examples/gsm8k/signal/data.json',
    signal_writing_file='/root/DeepResearcher/outputs/verl_examples/gsm8k/signal/signal.json',
    QUERY_SIGNAL=1,    # 与 Hydra config 中 query_signal: 1 对应
    RESPONSE_SIGNAL=0, # 与 Hydra config 中 response_signal: 0 对应
)

client = OpenAI(api_key='dummy', base_url='http://localhost:8000/v1')

handler = Handler(agent_config=agent_config, client=client, handler_config=handler_config)

print('Handler initialized, starting handle_execution loop...')
sys.stdout.flush()
sys.stderr.flush()

# 启动轮询循环
handler.handle_execution()
" 2>&1 | tee -a "$LOG_FILE"
