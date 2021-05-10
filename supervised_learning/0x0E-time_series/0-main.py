#!/usr/bin/env python3
preprocess = __import__('preprocess_data').preprocessing

if __name__ == "__main__":
    csv_path = '../data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    tr, va , te= preprocess(csv_path)
    print(tr)
    print(va)
    print(te)