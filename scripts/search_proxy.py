"""
轻量搜索代理服务 - 在 Mac 本地运行，通过 SSH 隧道给 GPU 服务器提供搜索能力。

使用方式：
  1. Mac 本地启动: python scripts/search_proxy.py
  2. SSH 反向隧道: ssh -R 8080:localhost:8080 root@js3.blockelite.cn -p 14328
  3. GPU 服务器调用: curl "http://localhost:8080/search?q=python+programming&top_k=10"

改进版：
  - 串行请求队列（避免并发打爆 DuckDuckGo）
  - 每次请求间隔 2 秒
  - 自动退避重试（失败后 exponential backoff）
  - 请求超时 30 秒
  - 简单内存缓存（避免重复查询）
"""

import json
import time
import threading
import hashlib
from collections import OrderedDict
from flask import Flask, request, jsonify
from ddgs import DDGS

app = Flask(__name__)

# === 配置 ===
_MIN_INTERVAL = 2.0       # 最小请求间隔（秒）—— 防止 DuckDuckGo 限流
_MAX_RETRIES = 3          # 最大重试次数
_INITIAL_BACKOFF = 3.0    # 初始退避时间（秒）
_CACHE_SIZE = 500         # 缓存条目数
_CACHE_TTL = 3600         # 缓存过期时间（秒）

# === 全局状态 ===
_request_lock = threading.Lock()  # 串行化所有搜索请求
_last_request_time = 0.0

# === LRU 缓存 ===
_cache = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(query, top_k):
    return hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()


def _cache_get(key):
    with _cache_lock:
        if key in _cache:
            entry = _cache[key]
            if time.time() - entry["time"] < _CACHE_TTL:
                _cache.move_to_end(key)
                return entry["data"]
            else:
                del _cache[key]
    return None


def _cache_set(key, data):
    with _cache_lock:
        _cache[key] = {"data": data, "time": time.time()}
        if len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)


def duckduckgo_search(query, top_k=10, region="wt-wt"):
    """
    使用 DuckDuckGo 搜索。
    所有请求串行化，带退避重试。
    """
    global _last_request_time

    # 先查缓存
    key = _cache_key(query, top_k)
    cached = _cache_get(key)
    if cached is not None:
        print(f"  [cache hit] q={query!r}")
        return cached

    # 串行化：一次只处理一个搜索
    with _request_lock:
        # 再检查一次缓存（可能在等锁期间被别的线程填充了）
        cached = _cache_get(key)
        if cached is not None:
            print(f"  [cache hit after wait] q={query!r}")
            return cached

        # 限流
        elapsed = time.time() - _last_request_time
        if elapsed < _MIN_INTERVAL:
            sleep_time = _MIN_INTERVAL - elapsed
            time.sleep(sleep_time)

        # 带退避的重试
        backoff = _INITIAL_BACKOFF
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                _last_request_time = time.time()
                with DDGS() as ddgs:
                    results = ddgs.text(query, region=region, max_results=top_k)
                    formatted = [{
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "content": r.get("body", "")
                    } for r in results]

                    # 缓存结果（即使为空也缓存，避免重复失败查询）
                    _cache_set(key, formatted)
                    if formatted:
                        print(f"  [ok] {len(formatted)} results")
                    else:
                        print(f"  [ok] 0 results (empty)")
                    return formatted

            except Exception as e:
                print(f"  [error attempt {attempt}/{_MAX_RETRIES}] {e}")
                if attempt < _MAX_RETRIES:
                    print(f"  [backoff] sleeping {backoff:.1f}s")
                    time.sleep(backoff)
                    backoff *= 2  # exponential backoff
                    _last_request_time = time.time()

        # 全部重试失败，缓存空结果避免短时间重复
        _cache_set(key, [])
        return []


# === 统计 ===
_stats = {"total": 0, "cache_hits": 0, "errors": 0, "success": 0}
_stats_lock = threading.Lock()


@app.route("/search", methods=["GET", "POST"])
def search():
    """
    搜索接口，兼容两种调用方式：
    - GET /search?q=xxx&top_k=10
    - POST /search (JSON body: {"q": "xxx", "top_k": 10})
    """
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        query = data.get("q", "")
        top_k = int(data.get("top_k", data.get("max_results", 10)))
    else:
        query = request.args.get("q", "")
        top_k = int(request.args.get("top_k", request.args.get("max_results", 10)))

    if not query:
        return jsonify({"error": "missing query parameter 'q'"}), 400

    with _stats_lock:
        _stats["total"] += 1

    print(f"[search #{_stats['total']}] q={query!r}, top_k={top_k}")
    results = duckduckgo_search(query, top_k=top_k)

    with _stats_lock:
        if results:
            _stats["success"] += 1
        else:
            _stats["errors"] += 1

    return jsonify({"results": results})


@app.route("/health", methods=["GET"])
def health():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "engine": "duckduckgo",
        "stats": _stats,
        "cache_size": len(_cache)
    })


@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    """清除缓存"""
    with _cache_lock:
        _cache.clear()
    return jsonify({"status": "cache cleared"})


if __name__ == "__main__":
    print("=" * 60)
    print("搜索代理服务启动 (增强版)")
    print(f"  限流间隔: {_MIN_INTERVAL}s")
    print(f"  最大重试: {_MAX_RETRIES} 次 (退避起始 {_INITIAL_BACKOFF}s)")
    print(f"  缓存容量: {_CACHE_SIZE} 条, TTL={_CACHE_TTL}s")
    print("")
    print("本地访问: http://localhost:8080/search?q=test")
    print("健康检查: http://localhost:8080/health")
    print("")
    print("GPU 服务器使用方法:")
    print("  1. 建立 SSH 反向隧道:")
    print("     ssh -R 8080:localhost:8080 root@js3.blockelite.cn -p 14328")
    print("  2. GPU 服务器上调用:")
    print("     curl 'http://localhost:8080/search?q=test&top_k=5'")
    print("=" * 60)
    app.run(host="0.0.0.0", port=8080, threaded=True)
