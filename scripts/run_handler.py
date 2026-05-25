#!/usr/bin/env python3
"""
Search Handler - 独立的信号轮询进程
监控 signal.json，等待 QUERY_SIGNAL，执行搜索，写回结果到 data.json
"""
import os
import sys

# 设置工作目录和 Python 路径
os.chdir('/root/DeepResearcher')
sys.path.insert(0, '/root/DeepResearcher')
# 添加 handler 目录到路径，让 web_search_agent 等包可被找到
sys.path.insert(0, '/root/DeepResearcher/scrl/handler')

import json
import yaml
import time
from types import SimpleNamespace

from handler import Handler
from openai import OpenAI

# 加载配置
config_path = '/root/DeepResearcher/scrl/handler/config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 确保缓存目录存在
os.makedirs('/root/DeepResearcher/scrl/handler/cache', exist_ok=True)

# 转换 yaml 配置为 dict，确保所有键都可用
# agent_config 必须包含 serper_api_key 等 WebSearchAgent 需要的键
agent_config = dict(config)
# 字段名映射：config.yaml 用 reading_agent_model，WebSearchAgent 用 quick_summary_model
if 'reading_agent_model' in agent_config and 'quick_summary_model' not in agent_config:
    agent_config['quick_summary_model'] = agent_config['reading_agent_model']
# 如果 query_save_path 是相对路径，转换为绝对路径
if 'query_save_path' in agent_config:
    query_path = agent_config['query_save_path']
    if not os.path.isabs(query_path):
        query_path = os.path.join('/root/DeepResearcher', query_path)
    agent_config['query_save_path'] = query_path

# Signal 配置 - 与 Hydra config 保持一致
# Hydra config: response_signal=0, query_signal=1
handler_config = SimpleNamespace(
    data_writing_file='/root/DeepResearcher/signal/data.json',
    signal_writing_file='/root/DeepResearcher/signal/signal.json',
    QUERY_SIGNAL=1,
    RESPONSE_SIGNAL=0,
)

client = OpenAI(api_key='dummy', base_url='http://localhost:8000/v1')
handler = Handler(agent_config=agent_config, client=client, handler_config=handler_config)

print('Handler initialized successfully!')
print(f'Polling signal file: {handler_config.signal_writing_file}')
print(f'Data file: {handler_config.data_writing_file}')
print(f'Search engine: {agent_config["search_engine"]}')
sys.stdout.flush()
sys.stderr.flush()

# 启动轮询循环
handler.handle_execution()
