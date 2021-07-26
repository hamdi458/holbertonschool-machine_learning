#!/usr/bin/env python3
"""requets get"""
import requests


def sentientPlanets():
    """method that returns the list of names of
    the home planets of all sentient species."""
    r = 'https://swapi-api.hbtn.io/api/species'
    planet = []

    while r is not None:
        data = requests.get(r).json()
        for species in data['results']:
            if (species['designation'] == 'sentient'
                or species['classification'] == 'sentient') and\
                     species['homeworld'] is not None:
                hw = requests.get(species['homeworld']).json()
                planet.append(hw['name'])
        r = data['next']
    return planet
