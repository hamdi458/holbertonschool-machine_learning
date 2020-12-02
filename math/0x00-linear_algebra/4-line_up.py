#!/usr/bin/env python3
""" add 2 listes"""


def add_arrays(arr1, arr2):
    """ add"""
    liste = []
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        liste.append(arr1[i] + arr2[i])
    return liste
