import requests
import json
import http.client
import time
import threading

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None  # DuckDuckGo not available, use proxy mode instead

_DDGS_SEMAPHORE = None
_DDGS_MIN_INTERVAL = 1.0
_DDGS_LAST_REQUEST_TIME = 0.0
_DDGS_LOCK = threading.Lock()
_DDGS_INITIALIZED = False


def _ddgs_ensure_init(config):
    global _DDGS_SEMAPHORE, _DDGS_MIN_INTERVAL, _DDGS_INITIALIZED
    if not _DDGS_INITIALIZED:
        _DDGS_SEMAPHORE = threading.Semaphore(config.get('duckduckgo_max_concurrent', 5))
        _DDGS_MIN_INTERVAL = config.get('duckduckgo_min_interval', 1.0)
        _DDGS_INITIALIZED = True


def _ddgs_throttled_search(query, top_k, region, lang):
    global _DDGS_LAST_REQUEST_TIME
    with _DDGS_SEMAPHORE:
        with _DDGS_LOCK:
            elapsed = time.time() - _DDGS_LAST_REQUEST_TIME
            if elapsed < _DDGS_MIN_INTERVAL:
                time.sleep(_DDGS_MIN_INTERVAL - elapsed)
            _DDGS_LAST_REQUEST_TIME = time.time()
        with DDGS() as ddgs:
            search_results = ddgs.text(query, region=region, max_results=top_k)
            results = []
            for r in search_results:
                results.append({
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
            return results


def web_search(query, config):
    if not query:
        raise ValueError("Search query cannot be empty")
    if config['search_engine'] == 'google':
        return serper_google_search(
            query=query,
            serper_api_key=config['serper_api_key'],
            top_k=config['search_top_k'],
            region=config['search_region'],
            lang=config['search_lang']
        )
    elif config['search_engine'] == 'bing':
        return azure_bing_search(
            query=query,
            subscription_key=config['azure_bing_search_subscription_key'],
            mkt=config['azure_bing_search_mkt'],
            top_k=config['search_top_k']
        )
    elif config['search_engine'] == 'duckduckgo':
        _ddgs_ensure_init(config)
        return duckduckgo_search(
            query=query,
            top_k=config['search_top_k'],
            region=config.get('search_region', 'us'),
            lang=config.get('search_lang', 'en'),
            max_retries=config.get('duckduckgo_max_retries', 5)
        )
    elif config['search_engine'] == 'baidu':
        return baidu_qianfan_search(
            query=query,
            api_key=config['baidu_qianfan_api_key'],
            top_k=config['search_top_k']
        )
    elif config['search_engine'] == 'proxy':
        return proxy_search(
            query=query,
            proxy_url=config.get('search_proxy_url', 'http://localhost:8080'),
            top_k=config['search_top_k'],
            max_retries=config.get('search_proxy_max_retries', 3)
        )
    elif config['search_engine'] == 'searxng':
        return searxng_search(
            query=query,
            searxng_url=config.get('searxng_url', 'http://localhost:8888'),
            top_k=config['search_top_k'],
            max_retries=config.get('searxng_max_retries', 3)
        )


def searxng_search(query, searxng_url='http://localhost:8888', top_k=10, max_retries=3):
    """Search via local SearXNG instance (meta-search engine aggregating Google, Bing, DuckDuckGo etc.).
    
    SearXNG JSON API requires POST method to return results properly.
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{searxng_url}/search",
                data={'q': query, 'format': 'json', 'categories': 'general'},
                timeout=30
            )
            data = response.json()
            results = []
            for r in data.get('results', [])[:top_k]:
                results.append({
                    "title": r.get('title', ''),
                    "link": r.get('url', ''),
                    "snippet": r.get('content', '')
                })
            if results:
                print(f"searxng search success: {len(results)} results")
            else:
                print("searxng search returned empty results")
            return results
        except Exception as e:
            wait = min(2 ** attempt + 1, 15)
            print(f"SearXNG search error (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait}s")
            time.sleep(wait)
    print(f"SearXNG search failed after {max_retries} retries: {query}")
    return []


def azure_bing_search(query, subscription_key, mkt, top_k, depth=0):
    params = {'q': query, 'mkt': mkt, 'count': top_k}
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

    results = []

    try:
        response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params)
        json_response = response.json()
        for e in json_response['webPages']['value']:
            results.append({
                "title": e['name'],
                "link": e['url'],
                "snippet": e['snippet']
            })
    except Exception as e:
        print(f"Bing search API error: {e}")
        if depth < 1024:
            time.sleep(1)
            return azure_bing_search(query, subscription_key, mkt, top_k, depth+1)
    return results


def serper_google_search(
        query, 
        serper_api_key,
        top_k,
        region,
        lang,
        depth=0
    ):
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
                "q": query,
                "num": top_k,
                "gl": region,
                "hl": lang,
            })
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))

        if not data:
            raise Exception("The google search API is temporarily unavailable, please try again later.")

        if "organic" not in data:
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        else:
            results = data["organic"]
            print("search success")
            return results
    except Exception as e:
        # print(f"Serper search API error: {e}")
        if depth < 512:
            time.sleep(1)
            return serper_google_search(query, serper_api_key, top_k, region, lang, depth=depth+1)
    print("search failed")
    return []


def duckduckgo_search(query, top_k=10, region='us', lang='en', max_retries=5):
    for attempt in range(max_retries):
        try:
            results = _ddgs_throttled_search(query, top_k, region, lang)
            print("duckduckgo search success")
            return results
        except Exception as e:
            wait = min(2 ** attempt + 1, 30)
            print(f"DuckDuckGo search error (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait}s")
            time.sleep(wait)
    print(f"DuckDuckGo search failed after {max_retries} retries: {query}")
    return None


def proxy_search(query, proxy_url='http://localhost:8080', top_k=10, max_retries=3):
    """Search via local proxy server (accessed through SSH reverse tunnel).
    
    The proxy runs on a machine with internet access and exposes a simple
    HTTP API. The GPU server reaches it via SSH reverse tunnel on localhost.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"{proxy_url}/search",
                params={'q': query, 'top_k': top_k},
                timeout=30
            )
            data = response.json()
            results = []
            for r in data.get('results', []):
                results.append({
                    "title": r.get('title', ''),
                    "link": r.get('url', ''),
                    "snippet": r.get('content', '')
                })
            if results:
                print("proxy search success")
            else:
                print("proxy search returned empty results")
            return results
        except Exception as e:
            wait = min(2 ** attempt + 1, 15)
            print(f"Proxy search error (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait}s")
            time.sleep(wait)
    print(f"Proxy search failed after {max_retries} retries: {query}")
    return []


def baidu_qianfan_search(query, api_key, top_k=10, depth=0):
    url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        "messages": [{"role": "user", "content": query}],
        "edition": "lite",
        "search_source": "baidu_search_v2",
        "resource_type_filter": [{"type": "web", "top_k": top_k}]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        data = response.json()

        if 'code' in data and data['code']:
            raise Exception(f"Baidu search API error: {data.get('message', data['code'])}")

        results = []
        for ref in data.get('references', []):
            results.append({
                "title": ref.get('title', ''),
                "link": ref.get('url', ''),
                "snippet": ref.get('snippet', '') or ref.get('content', '')
            })
        print("baidu search success")
        return results
    except Exception as e:
        print(f"Baidu search API error: {e}")
        if depth < 5:
            time.sleep(2)
            return baidu_qianfan_search(query, api_key, top_k, depth+1)
    print("baidu search failed")
    return []


if __name__ == "__main__":
    print(duckduckgo_search("DeepResearcher project", top_k=3))