import os


class BaseConfig(object):
    CACHE_REDIS_HOST = os.environ["CACHE_REDIS_HOST"]
    CACHE_REDIS_PORT = os.environ["CACHE_REDIS_PORT"]
    CACHE_REDIS_DB = os.environ["CACHE_REDIS_DB"]
    SQLALCHEMY_DATABASE_URI = os.environ["DB_CONNECTION_URL"]
