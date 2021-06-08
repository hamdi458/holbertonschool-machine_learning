#!/usr/bin/env python3
""""""
scanf_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """that answers questions from a reference text:
    reference is the reference text
    If the answer cannot be found in the
        reference text, respond with Sorry,
        I do not understand your question."""

    w = ['exit', 'quit', 'goodbye', 'bye']

    while 1:
        scanf = input('Q: ')
        scanf = scanf.lower()

        if scanf in w:
            print('A: Goodbye')
            exit(0)
        else:
            answer = scanf_answer(scanf, reference)
            if answer is None or answer == '':
                print('A: Sorry, I do not understand your question.')
            else:
                print('A: {}'.format(answer))
