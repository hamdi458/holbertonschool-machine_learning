#!/usr/bin/env python3
"""
Test file
"""
import requests
import time
import sys


if __name__ == '__main__':
    req = requests.get(sys.argv[1])
    data = req.json()
    if req .status_code == 404:
        print("Not found")
    elif req .status_code == 200:
        print(data["location"])
    elif req .status_code == 403:
        limit = req.headers["X-Ratelimit-Reset"]
        x = (int(limit) - int(time.time())) / 60
        print("Reset in {} min".format(int(x)))
