import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

_client = None
_db = None

def get_db():
    """Returns the MongoDB database instance (singleton)."""
    global _client, _db
    if _db is not None:
        return _db

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI not set in environment")

    _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    # Ping to verify connection
    _client.admin.command("ping")
    _db = _client[os.environ.get("MONGODB_DB", "stylevector")]
    return _db


def get_collection(name: str):
    return get_db()[name]
