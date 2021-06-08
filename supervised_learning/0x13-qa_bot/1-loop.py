#!/usr/bin/env python3
"""script that takes in input from the user
with the prompt Q: and prints A: as a response.
If the user inputs exit, quit, goodbye, or bye, case
insensitive, print A: Goodbye and exit.
"""
w = ['exit', 'quit', 'goodbye', 'bye']
while 1:
    scanf = input('Q: ')
    if scanf.lower() in w:
        print('A: Goodbye')
        exit(0)
    else:
        print('A: ')
