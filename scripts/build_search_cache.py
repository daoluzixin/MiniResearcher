#!/usr/bin/env python3
"""
预缓存搜索结果：对训练数据集前25条问题生成搜索query，
用本地SearXNG搜索并保存为handler可用的缓存格式。

缓存格式：
{
    "query_string": {
        "timestamp": float,
        "organic": [{"title": "", "link": "", "snippet": ""}, ...]
    }
}
"""
import json
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re

# 配置
SEARXNG_URL = "http://localhost:8888"
TOP_K = 10
NUM_SAMPLES = 25
DATA_PATH = "data/train.parquet"
OUTPUT_PATH = "scrl/handler/cache/search_result.json"
MAX_CONCURRENT = 5  # 本地搜索并发数
SEARCH_TIMEOUT = 30

# 信号量控制并发
semaphore = threading.Semaphore(MAX_CONCURRENT)
cache_lock = threading.Lock()
cache = {}


def searxng_search(query, max_retries=3):
    """搜索并返回结果列表"""
    for attempt in range(max_retries):
        try:
            with semaphore:
                response = requests.post(
                    f"{SEARXNG_URL}/search",
                    data={'q': query, 'format': 'json', 'categories': 'general'},
                    timeout=SEARCH_TIMEOUT
                )
                data = response.json()
                results = []
                for r in data.get('results', [])[:TOP_K]:
                    results.append({
                        "title": r.get('title', ''),
                        "link": r.get('url', ''),
                        "snippet": r.get('content', '')
                    })
                return results
        except Exception as e:
            wait = min(2 ** attempt + 1, 15)
            print(f"  Search error (attempt {attempt+1}/{max_retries}): {e}, retrying in {wait}s")
            time.sleep(wait)
    return []


def generate_queries_for_question(question):
    """为一个问题生成多个可能的搜索query"""
    queries = set()
    
    # 1. 问题本身
    queries.add(question)
    
    # 2. 去掉问号
    q_clean = question.rstrip('?').strip()
    queries.add(q_clean)
    
    # 3. 提取关键实体/名词短语作为搜索词
    # 常见的问题模式
    patterns = [
        # "Who directed X" -> "X director"
        (r"[Ww]ho directed (?:a |the )?(?:movie|film)?\s*(.+?)(?:\?|$)", lambda m: [m.group(1) + " director", m.group(1)]),
        # "Where was the director of film X born" -> "X film director", "director of X birthplace"
        (r"[Ww]here was the director of (?:film |movie )?(.+?) born", lambda m: [f"director of {m.group(1)}", f"{m.group(1)} director birthplace", m.group(1)]),
        # "What is the date of death of X" -> "X death date", "X"
        (r"[Ww]hat (?:is|was) the date of death of (.+?)(?:\?|$)", lambda m: [f"{m.group(1)} death date", m.group(1)]),
        # "Who is the father of X" -> "X father", "X"
        (r"[Ww]ho is the father of (.+?)(?:\?|$)", lambda m: [f"{m.group(1)} father", m.group(1)]),
        # "When was the director of film X born" -> "director of X birthday"
        (r"[Ww]hen (?:is|was) the director of (?:film |movie )?(.+?)(?:'s)? (?:born|birthday)", lambda m: [f"director of {m.group(1)} birthday", f"{m.group(1)} director", m.group(1)]),
        # "What is the place of birth of X" -> "X birthplace"
        (r"[Ww]hat is the place of birth of (.+?)(?:\?|$)", lambda m: [f"{m.group(1)} birthplace", m.group(1)]),
        # "who plays X on/in Y" -> "X Y actor", "X Y cast"
        (r"who plays (.+?) (?:on|in) (.+?)(?:\?|$)", lambda m: [f"{m.group(1)} {m.group(2)} actor", f"{m.group(2)} cast", m.group(1)]),
        # "Who wrote X" -> "X writer", "X composer"
        (r"who wrote (.+?)(?:\?|$)", lambda m: [f"{m.group(1)} writer", f"{m.group(1)} composer", m.group(1)]),
        # "Which country the director of film X is from" -> "director of X nationality"
        (r"[Ww]hich country (?:the |)director of (?:film |movie )?(.+?) is from", lambda m: [f"director of {m.group(1)} nationality", f"{m.group(1)} director", m.group(1)]),
        # "Where did the director of film X die" -> "director of X death place"
        (r"[Ww]here did the director of (?:film |movie )?(.+?) die", lambda m: [f"director of {m.group(1)} death", f"{m.group(1)} director", m.group(1)]),
        # General "What is/was X" -> "X"
        (r"[Ww]hat (?:is|was|are|were) (.+?)(?:\?|$)", lambda m: [m.group(1)]),
        # "How many X" -> "X"
        (r"[Hh]ow many (.+?)(?:\?|$)", lambda m: [m.group(1)]),
    ]
    
    for pattern, extractor in patterns:
        match = re.search(pattern, question)
        if match:
            extracted = extractor(match)
            for q in extracted:
                q = q.strip().rstrip('?').strip()
                if q and len(q) > 3:
                    queries.add(q)
    
    # 4. 提取引号中的内容
    quoted = re.findall(r'"([^"]+)"', question) + re.findall(r"'([^']+)'", question)
    for q in quoted:
        if len(q) > 3:
            queries.add(q)
    
    # 5. 提取括号中的内容（通常是年份或补充信息）
    parens = re.findall(r'\(([^)]+)\)', question)
    for p in parens:
        # 如果括号内容包含年份，和前面的词组合
        if re.search(r'\d{4}', p):
            # 找括号前的词
            match = re.search(r'(\w[\w\s]+?)\s*\(' + re.escape(p) + r'\)', question)
            if match:
                queries.add(f"{match.group(1).strip()} {p}")
    
    return list(queries)


def search_and_cache(query):
    """搜索一个query并加入缓存"""
    results = searxng_search(query)
    with cache_lock:
        cache[query] = {
            "timestamp": time.time(),
            "organic": results
        }
    return query, len(results)


def main():
    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # 取前25条
    questions = [row['prompt'][0]['content'] for _, row in df.head(NUM_SAMPLES).iterrows()]
    print(f"Loaded {len(questions)} questions")
    
    # 为每个问题生成搜索query
    all_queries = set()
    for i, q in enumerate(questions):
        queries = generate_queries_for_question(q)
        print(f"[{i:2d}] {q[:60]}... -> {len(queries)} queries")
        all_queries.update(queries)
    
    print(f"\nTotal unique queries to search: {len(all_queries)}")
    print("Starting searches...")
    
    # 并发搜索
    success_count = 0
    empty_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(search_and_cache, q): q for q in all_queries}
        for i, future in enumerate(as_completed(futures)):
            query, num_results = future.result()
            if num_results > 0:
                success_count += 1
            else:
                empty_count += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(all_queries):
                print(f"  Progress: {i+1}/{len(all_queries)} (success: {success_count}, empty: {empty_count})")
    
    # 保存缓存
    import os
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone! Cached {len(cache)} queries ({success_count} with results, {empty_count} empty)")
    print(f"Output: {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
