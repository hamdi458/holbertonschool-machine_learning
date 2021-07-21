#!/usr/bin/env python3
"""
create a pd.DataFrame from a dictionary
"""
import pandas as pd

dic = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']}
index = ['A', 'B', 'C', 'D']
df = pd.DataFrame(dic, index=index)
