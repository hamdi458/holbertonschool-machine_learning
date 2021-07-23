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
        exit()

    elif r.status_code == 403:
        tim = data.headers['X-Ratelimit-Reset']
        tim = int(tim) - int(time.time())
        print("Reset in {} min".format(int(tim / 60)))
        exit()

    print('Not found')
