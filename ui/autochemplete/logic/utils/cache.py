import functools
from hashlib import blake2b
import json
from typing import Dict
import redis
from flask import g, current_app


def get_cache() -> redis.Redis:
    if "cache" not in g:
        g.cache = redis.Redis(
            host=current_app.config["CACHE_REDIS_HOST"],
            port=current_app.config["CACHE_REDIS_PORT"],
            db=current_app.config["CACHE_REDIS_DB"],
        )
    return g.cache


def cache_stats() -> Dict[str, int]:
    r = get_cache()
    hits = r.get("cache_hits") or b"0"
    misses = r.get("cache_misses") or b"0"
    return {"cache_hits": hits.decode("utf-8"), "cache_misses": misses.decode("utf-8")}


def _hash(pydantic_model):
    serialized = pydantic_model.json(exclude={"_hash"}, sort_keys=True)
    return blake2b(serialized.encode("utf-8"), digest_size=20).hexdigest()


def cached(func):
    """
    Decorator for caching single pydantic arg and response functions
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        r = get_cache()
        cache_available = get_cache().ping()

        if not cache_available:
            current_app.logger.warning(
                "Cache not reachable. Falling back to uncached response.")
            return func(*args, **kwargs)

        cache_key = f"{func.__name__}-{_hash(args[0])}"
        if r.exists(cache_key):
            current_app.logger.debug(f"Cache hit: {cache_key}")
            r.incr("cache_hits")
            result = json.loads(r.get(cache_key))
            # check if decorated func has an annotated return type
            if hasattr(func, "__annotations__") and func.__annotations__.get("return"):
                if isinstance(result, dict):
                    if func.__annotations__.get("return") is Dict:
                        return dict(**result)
                    else:
                        return func.__annotations__.get("return")(**result)
                elif isinstance(result, list):
                    return func.__annotations__.get("return")(*result)
                else:
                    return func.__annotations__.get("return")(result)
            else:
                return result
        else:
            current_app.logger.info(f"Cache miss: {cache_key}")
            r.incr("cache_misses")
            result = func(*args, **kwargs)
            if hasattr(result, "json"):
                r.set(cache_key, result.json())
            else:
                r.set(cache_key, json.dumps(result))
            current_app.logger.info(f"Built cache {cache_key}")
            return result

    return wrapper


def clear_cache():
    r = get_cache()
    current_app.logger.info("Cleared cache")
    return r.flushdb()
