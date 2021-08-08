""" Data Preprocessing and Formatting """
import random


def split_txt(file, out1, out2, n, shuffle=True, seed=41):
    """
    Assumption:
        1. each line represents one data point
        2. n: randomly split out n examples
    """
    with open(file, 'r') as in_fp:
        in_lines = in_fp.readlines()

    if shuffle:
        random.seed(seed)
        random.shuffle(in_lines)

    data1, data2 = in_lines[:n], in_lines[n:]
    assert(len(data1)+len(data2)==len(in_lines))

    with open(out1, 'wt', encoding='utf-8') as out_fp:
        out_fp.write('\n'.join(data1))
    with open(out2, 'wt', encoding='utf-8') as out_fp:
        out_fp.write('\n'.join(data2))

    print("Split data saved")
    return
