#!/usr/bin/env python3
import subprocess, time, os

# 杀掉旧的 handler
subprocess.run(['ssh', '-p', '25516', 'root@js2.blockelite.cn', 'pkill -f "run_handler.py"'],
               capture_output=True)
time.sleep(1)

# 启动新的 handler
cmd = [
    'ssh', '-p', '25516', 'root@js2.blockelite.cn',
    """nohup env PYTHONPATH="/root/DeepResearcher:$PYTHONPATH" HF_HOME="/root/.cache/modelscope/hub" /home/vipuser/miniconda3/bin/python /root/DeepResearcher/scripts/run_handler.py >>/root/DeepResearcher/logs/handler/handler.log 2>&1 &"""
]
result = subprocess.run(cmd, capture_output=True, text=True)
print("SSH result:", result.returncode, result.stdout, result.stderr)
time.sleep(2)

# 确认进程存在
result2 = subprocess.run(['ssh', '-p', '25516', 'root@js2.blockelite.cn',
    'ps aux | grep run_handler | grep -v grep'], capture_output=True, text=True)
print("Handler process:", result2.stdout)

# 检查日志
result3 = subprocess.run(['ssh', '-p', '25516', 'root@js2.blockelite.cn',
    'tail -5 /root/DeepResearcher/logs/handler/handler.log'], capture_output=True, text=True)
print("Handler log:", result3.stdout, result3.stderr)
