#!/usr/bin/env python3
"""
Test file
"""
import requests
import time
import sys


if __name__ == '__main__':
    r = requests.get(sys.argv[1])
    if r.status_code == 200:
        data = r.json()
        print(data['location'])

    elif r.status_code == 403:
        limit = req.headers["X-Ratelimit-Reset"]
        x = (int(limit) - int(time.time())) / 60
        print("Reset in {} min".format(int(x)))
    else:
        print('Not found')
