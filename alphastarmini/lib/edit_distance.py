#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Edit distance: module for calculating levenshtein distance and hamming distance "

import time
import random

import Levenshtein

__author__ = "Ruo-Ze Liu"

debug = False


def levenshtein_recur(a, b):
    '''
    source: https://blog.finxter.com/how-to-calculate-the-levenshtein-distance-in-python/,
    It is simple and right but not efficient when len(a) or len(b) > 10. 
    '''

    if not a:
        return len(b)
    if not b:
        return len(a)
    return min(levenshtein_recur(a[1:], b[1:]) + (a[0] != b[0]),
               levenshtein_recur(a[1:], b) + 1,
               levenshtein_recur(a, b[1:]) + 1)


def hammingDist(s1, s2):
    '''
    source: https://www.cnblogs.com/lexus/p/3772389.html
    '''
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])


def test():
    levenshtein = levenshtein_recur

    Start = 0
    Stop = 565
    limit = 10
    list_1 = ''.join([chr(random.randrange(Start, Stop)) for iter in range(limit)])
    list_2 = ''.join([chr(random.randrange(Start, Stop)) for iter in range(limit)])
    print(list_1)
    print(list_2)

    print("distance between 'cat', 'chello'", levenshtein('cat', 'chello'))
    print("distance between '', 'chello'", levenshtein('', 'chello'))
    print("distance between 'cat', ''", levenshtein('cat', ''))
    print("distance between 'cat', 'chello'", levenshtein('cat', 'chello'))
    print("distance between 'cat', 'cate'", levenshtein('cat', 'cate'))
    print("distance between 'cat', 'ca'", levenshtein('cat', 'ca'))
    print("distance between 'cat', 'cad'", levenshtein('cat', 'cad'))

    begin = time.time()
    print("distance between list_1, list_2", Levenshtein.distance(list_1, list_2))
    end = time.time() 
    print(f"Total runtime of the Levenshtein.distance is {end - begin}")

    begin = time.time()
    print("distance between list_1, list_2", levenshtein_recur(list_1, list_2))
    end = time.time() 
    print(f"Total runtime of the levenshtein_recur is {end - begin}")

    print("hamming distance between 'cat', 'cad'", hammingDist('cat', 'cad'))
    print("hamming distance between 'cat', 'cad'", Levenshtein.hamming('cat', 'cad'))

    Start = 0
    Stop = 565
    limit = 100
    list_1 = ''.join([chr(random.randrange(Start, Stop)) for iter in range(limit)])
    list_2 = ''.join([chr(random.randrange(Start, Stop)) for iter in range(limit)])
    print(list_1)
    print(list_2)

    begin = time.time()
    print("hamming distance between list_1, list_2", Levenshtein.hamming(list_1, list_2))
    end = time.time() 
    print(f"Total runtime of the Levenshtein.hamming is {end - begin}")

    begin = time.time()
    print("hamming distance between list_1, list_2", hammingDist(list_1, list_2))
    end = time.time() 
    print(f"Total runtime of the hammingDist is {end - begin}")


if __name__ == '__main__':
    test()
