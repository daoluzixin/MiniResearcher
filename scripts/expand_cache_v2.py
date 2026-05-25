#!/usr/bin/env python3
"""
进一步扩充缓存：基于ground truth答案和问题结构，
生成模型在多轮对话中可能搜索的各种query变体。
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


def search_and_cache(query):
    results = searxng_search(query)
    with cache_lock:
        cache[query] = {"timestamp": time.time(), "organic": results}
    return query, len(results)


def main():
    df = pd.read_parquet(DATA_PATH)
    samples = df.head(NUM_SAMPLES)
    
    all_new_queries = set()
    
    for i, (_, row) in enumerate(samples.iterrows()):
        question = row['prompt'][0]['content']
        ground_truth = row['reward_model'].get('ground_truth', '')
        
        # 基于ground truth生成验证性搜索
        if ground_truth:
            all_new_queries.add(ground_truth)
            all_new_queries.add(f"{ground_truth} wikipedia")
            all_new_queries.add(f"{ground_truth} biography")
            
            # 问题+答案组合
            q_keywords = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', question)
            for kw in q_keywords[:3]:
                if len(kw) > 3:
                    all_new_queries.add(f"{ground_truth} {kw}")
                    all_new_queries.add(kw)
        
        # 基于问题结构生成更多变体
        # 提取所有大写开头的词组（可能是专有名词）
        proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', question)
        for noun in proper_nouns:
            if len(noun) > 3 and noun not in cache:
                all_new_queries.add(noun)
                all_new_queries.add(f"{noun} wikipedia")
        
        # 电影相关的常见搜索模式
        film_match = re.search(r'(?:film|movie)\s+(.+?)(?:\s+(?:born|die|death|from)|\?|$)', question, re.I)
        if film_match:
            film_name = film_match.group(1).strip().rstrip('?')
            all_new_queries.add(film_name)
            all_new_queries.add(f"{film_name} film")
            all_new_queries.add(f"{film_name} movie")
            all_new_queries.add(f"{film_name} director")
            all_new_queries.add(f"{film_name} cast")
            all_new_queries.add(f"{film_name} Wikipedia")
            all_new_queries.add(f"{film_name} IMDb")
        
        # 人名相关
        person_patterns = [
            r"([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",  # 2-3 word names
        ]
        for pat in person_patterns:
            persons = re.findall(pat, question)
            for person in persons:
                if len(person) > 5 and person not in ['Red Nose', 'Six Nations']:
                    all_new_queries.add(person)
                    all_new_queries.add(f"{person} wikipedia")
                    all_new_queries.add(f"{person} biography")
                    all_new_queries.add(f"{person} filmography")
    
    # 去掉已缓存的和太短的
    all_new_queries = {q.strip() for q in all_new_queries if q.strip() not in cache and len(q.strip()) > 3}
    print(f"New queries to search: {len(all_new_queries)}")
    
    if not all_new_queries:
        print("No new queries!")
        return
    
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
    
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    
    import os
    print(f"\nDone! Total cached: {len(cache)} queries")
    print(f"Output: {CACHE_PATH} ({os.path.getsize(CACHE_PATH) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
