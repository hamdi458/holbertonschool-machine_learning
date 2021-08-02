#!/usr/bin/env python3
"""Change school topics"""


def update_topics(mongo_collection, name, topics):
    """Function that changes all topics of a school document based on the name """
    values = {'$set': {'topics': topics}}
    mongo_collection.update_many({'name': name}, values)
