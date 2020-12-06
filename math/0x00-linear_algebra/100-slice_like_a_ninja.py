#!/usr/bin/env python3
"""function np_slice that slices a matrix along specific axes"""


def np_slice(matrix, axes={}):
    start = list(axes.keys())[0]
    stop = max(list(axes.keys()))
    matrix_new = []
    for i in range(start, stop + 1):
        if (i in list(axes.keys())):
            list1 = axes[i]
            matrix_new.append(slice(*list1))
        else:
            matrix_new.append(slice(None, None, None))
    return(matrix[tuple(matrix_new)])
