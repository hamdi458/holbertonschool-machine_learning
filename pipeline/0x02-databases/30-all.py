#!/usr/bin/env python3
"""List all documents in Python
"""

def list_all(mongo_collection):
    """function that lists all documents in a collection"""
    list_mongo = []
    for item in mongo_collection.find():
        list_mongo.append(item)
    return list_mongo
