#!/usr/bin/env python3
"""Multi-reference Question Answering
"""
quest_ans = __import__('0-qa').question_answer
seman_sear = __import__('3-semantic_search').semantic_search


def question_answer(coprus_path):
    """answers questions from multiple reference texts:"""
    w = ['exit', 'quit', 'goodbye', 'bye']

    while 1:
        scanf = input('Q: ')
        if scanf.lower() in w:
            print('A: Goodbye')
            exit(0)
        else:
            reference = seman_sear(coprus_path, scanf)
            answer = quest_ans(scanf, reference)
            if answer is None or answer == '':
                print('A: Sorry, I do not understand your question.')
            else:
                print('A: {}'.format(answer))

