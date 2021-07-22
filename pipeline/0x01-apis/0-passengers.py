#!/usr/bin/env python3
"""requets get"""
import requests


def availableShips(passengerCount):
    """method that returns the list of ships that can
    hold a given number of passengers"""
    r = requests.get('https://swapi-api.hbtn.io/api/starships/')
    data = r.json()
    ships = []
    nextt = 'https://swapi-api.hbtn.io/api/starships/'
    while nextt is not None:
        data = requests.get(nextt).json()
        for item in data['results']:
            passengers = item['passengers'].replace(',', '')
            if passengers == 'n/a' or passengers == 'unknown':
                passengers = 0
            if int(passengers) >= passengerCount:
                ships.append(item['name'])
        nextt = data['next']
    return ships
