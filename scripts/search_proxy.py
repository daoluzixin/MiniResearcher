"""
Lightweight local search proxy for development and controlled search-agent experiments.

Example usage:
  1. Start locally:
       python scripts/search_proxy.py
  2. Expose it to a remote trainer with your own tunnel settings:
       ssh -R <remote_port>:localhost:8080 <your-remote-host>
  3. Call from the remote environment:
       curl "http://localhost:<remote_port>/search?q=python+programming&top_k=10"
"""

import hashlib
import threading
import time
from collections import OrderedDict

from ddgs import DDGS
from flask import Flask, jsonify, request

app = Flask(__name__)

_MIN_INTERVAL = 2.0
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 3.0
_CACHE_SIZE = 500
_CACHE_TTL = 3600

_request_lock = threading.Lock()
_last_request_time = 0.0
_cache = OrderedDict()
_cache_lock = threading.Lock()
_stats = {"total": 0, "errors": 0, "success": 0}
_stats_lock = threading.Lock()


def _cache_key(query, top_k):
    return hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()


def _cache_get(key):
    with _cache_lock:
        if key in _cache:
            entry = _cache[key]
            if time.time() - entry["time"] < _CACHE_TTL:
                _cache.move_to_end(key)
                return entry["data"]
            del _cache[key]
    return None


def _cache_set(key, data):
    with _cache_lock:
        _cache[key] = {"data": data, "time": time.time()}
        if len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)


def duckduckgo_search(query, top_k=10, region="wt-wt"):
    global _last_request_time

    key = _cache_key(query, top_k)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    with _request_lock:
        cached = _cache_get(key)
        if cached is not None:
            return cached

        elapsed = time.time() - _last_request_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)

        backoff = _INITIAL_BACKOFF
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                _last_request_time = time.time()
                with DDGS() as ddgs:
                    results = ddgs.text(query, region=region, max_results=top_k)
                    formatted = [
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "content": r.get("body", ""),
                        }
                        for r in results
                    ]
                    _cache_set(key, formatted)
                    return formatted
            except Exception:
                if attempt < _MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
                    _last_request_time = time.time()

        _cache_set(key, [])
        return []


@app.route("/search", methods=["GET", "POST"])
def search():
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

    results = duckduckgo_search(query, top_k=top_k)

    with _stats_lock:
        if results:
            _stats["success"] += 1
        else:
            _stats["errors"] += 1

    return jsonify({"results": results})


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "engine": "duckduckgo",
            "stats": _stats,
            "cache_size": len(_cache),
        }
    )


if __name__ == "__main__":
    print("Search proxy running on http://localhost:8080")
    print("Expose it to a remote trainer using your own SSH tunnel settings.")
    app.run(host="0.0.0.0", port=8080, threaded=True)
