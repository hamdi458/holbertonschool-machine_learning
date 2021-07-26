#!/usr/bin/env python3
""". What will be next?
"""
import requests
import sys
import time

if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    r = requests.get(url)
    lanes = sorted(r.json(), key=lambda i: i['date_unix'])
    date_unix = lanes[0]['date_unix']
    for item in r.json():
        if item['date_unix'] == date_unix:
            date = item['date_local']
            lrocket = item['rocket']
            lanpad_id = item['launchpad']
            break
    lan_name = item['name']
    url = "https://api.spacexdata.com/v4/rockets/{}".format(lrocket)
    r = requests.get(url)
    rocket_name = r.json()['name']
    url = "https://api.spacexdata.com/v4/launchpads/{}".format(lanpad_id)
    r = requests.get(url)
    lpad_name = r.json()['name']
    lpad_locality = r.json()['locality']
    print("{} ({}) {} - {} ({})".format(lan_name, date, rocket_name,
                                        lpad_name, lpad_locality))
