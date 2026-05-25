#!/usr/bin/env python3
"""
扩充搜索缓存：基于已有搜索结果中的实体和标题，生成更多可能的follow-up query。
模拟模型在多轮搜索中可能生成的query。
"""
import json
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re

SEARXNG_URL = "http://localhost:8888"
TOP_K = 10
NUM_SAMPLES = 25
DATA_PATH = "data/train.parquet"
CACHE_PATH = "scrl/handler/cache/search_result.json"
MAX_CONCURRENT = 5
SEARCH_TIMEOUT = 30

semaphore = threading.Semaphore(MAX_CONCURRENT)
cache_lock = threading.Lock()

# 加载已有缓存
with open(CACHE_PATH, 'r') as f:
    cache = json.load(f)

print(f"Existing cache: {len(cache)} queries")


def searxng_search(query, max_retries=3):
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
            time.sleep(min(2 ** attempt + 1, 15))
    return []


def generate_followup_queries(question, existing_results):
    """基于问题和已有搜索结果，生成模型可能的follow-up query"""
    queries = set()
    
    # 从搜索结果的title和snippet中提取实体
    entities = set()
    for result in existing_results:
        title = result.get('title', '')
        snippet = result.get('snippet', '')
        # 提取可能的人名、电影名等
        # 标题通常包含关键实体
        if title and len(title) > 3:
            entities.add(title.split(' - ')[0].strip())
            entities.add(title.split(' | ')[0].strip())
    
    # 基于问题类型生成follow-up
    q_lower = question.lower()
    
    if 'director' in q_lower:
        # 可能搜索具体导演名
        for entity in list(entities)[:5]:
            if len(entity) > 3 and len(entity) < 60:
                queries.add(f"{entity} director")
                queries.add(f"{entity} film")
                queries.add(entity)
    
    if 'born' in q_lower or 'birth' in q_lower:
        for entity in list(entities)[:5]:
            if len(entity) > 3 and len(entity) < 60:
                queries.add(f"{entity} birthplace")
                queries.add(f"{entity} born")
    
    if 'death' in q_lower or 'died' in q_lower:
        for entity in list(entities)[:5]:
            if len(entity) > 3 and len(entity) < 60:
                queries.add(f"{entity} death date")
                queries.add(f"{entity} died")
    
    if 'who' in q_lower and ('play' in q_lower or 'act' in q_lower):
        for entity in list(entities)[:5]:
            if len(entity) > 3 and len(entity) < 60:
                queries.add(f"{entity} cast")
                queries.add(f"{entity} actor")
    
    if 'wrote' in q_lower or 'composer' in q_lower or 'theme' in q_lower:
        for entity in list(entities)[:5]:
            if len(entity) > 3 and len(entity) < 60:
                queries.add(f"{entity} composer")
                queries.add(f"{entity} written by")
    
    # 通用：从snippet中提取引号内容
    for result in existing_results:
        snippet = result.get('snippet', '')
        quoted = re.findall(r'"([^"]{4,50})"', snippet)
        for q in quoted:
            queries.add(q)
    
    # 过滤掉已在缓存中的
    queries = {q for q in queries if q not in cache and len(q) > 3}
    
    return list(queries)[:15]  # 每个问题最多15个follow-up


def search_and_cache(query):
    results = searxng_search(query)
    with cache_lock:
        cache[query] = {
            "timestamp": time.time(),
            "organic": results
        }
    return query, len(results)


def main():
    df = pd.read_parquet(DATA_PATH)
    questions = [row['prompt'][0]['content'] for _, row in df.head(NUM_SAMPLES).iterrows()]
    
    # 为每个问题生成follow-up queries
    all_new_queries = set()
    for i, question in enumerate(questions):
        # 找到这个问题相关的已有搜索结果
        related_results = []
        for key, val in cache.items():
            # 简单匹配：如果query和问题有重叠词
            q_words = set(question.lower().split())
            k_words = set(key.lower().split())
            if len(q_words & k_words) >= 3:
                related_results.extend(val.get('organic', []))
        
        followups = generate_followup_queries(question, related_results[:20])
        if followups:
            print(f"[{i:2d}] {question[:50]}... -> {len(followups)} follow-up queries")
            all_new_queries.update(followups)
    
    # 去掉已缓存的
    all_new_queries = {q for q in all_new_queries if q not in cache}
    print(f"\nNew queries to search: {len(all_new_queries)}")
    
    if not all_new_queries:
        print("No new queries needed!")
        return
    
    # 搜索
    success_count = 0
    empty_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(search_and_cache, q): q for q in all_new_queries}
        for i, future in enumerate(as_completed(futures)):
            query, num_results = future.result()
            if num_results > 0:
                success_count += 1
            else:
                empty_count += 1
            if (i + 1) % 20 == 0 or (i + 1) == len(all_new_queries):
                print(f"  Progress: {i+1}/{len(all_new_queries)} (success: {success_count}, empty: {empty_count})")
    
    # 保存
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    
    import os
    print(f"\nDone! Total cached: {len(cache)} queries")
    print(f"Output: {CACHE_PATH} ({os.path.getsize(CACHE_PATH) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
