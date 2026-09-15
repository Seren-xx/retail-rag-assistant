"""会话存储：Redis 会话（session_id → List），TTL 过期、条数上限与损坏数据容错。"""

import json

from .. import config


class RedisSessionStore:
    """Redis 会话存储：session_id → List，TTL 默认 7 天，裁剪保留最近 max_messages 条。"""

    PREFIX = "retail:session:"

    def __init__(self, client=None, redis_url=None, ttl_seconds=7 * 24 * 3600, max_messages=50):
        self._client = client
        self.redis_url = redis_url or config.redis_url
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages

    def _conn(self):
        if self._client is None:
            self._client = _connect_redis(self.redis_url)
        return self._client

    def _key(self, session_id):
        return self.PREFIX + session_id

    @staticmethod
    def _loads(raw):
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        return obj if isinstance(obj, dict) and "role" in obj else None

    def get(self, session_id):
        raws = self._conn().lrange(self._key(session_id), 0, -1)
        return [m for m in (self._loads(r) for r in raws or []) if m]

    def add(self, session_id, role, content):
        conn = self._conn()
        key = self._key(session_id)
        conn.rpush(key, json.dumps({"role": role, "content": content}, ensure_ascii=False))
        conn.ltrim(key, -self.max_messages, -1)
        conn.expire(key, self.ttl_seconds)

    def clear(self, session_id):
        self._conn().delete(self._key(session_id))


def _connect_redis(redis_url):
    """连接并 ping Redis（1 秒超时）；失败抛异常由调用方处理。"""
    import redis

    client = redis.Redis.from_url(redis_url, socket_connect_timeout=1)
    client.ping()
    return client


def make_session_store(redis_url=None):
    """创建会话存储：连接 Redis（url 缺省取 config.redis_url）。"""
    url = redis_url or config.redis_url
    client = _connect_redis(url)
    return RedisSessionStore(client=client, redis_url=url)
