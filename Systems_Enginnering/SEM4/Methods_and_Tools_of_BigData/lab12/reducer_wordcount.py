#!/usr/bin/env python3

import sys

current = None
total = 0

for line in sys.stdin:
    word, count = line.strip().split("\t")
    count = int(count)

    if word != current:
        if current:
            print(f"{current}\t{total}")
        current = word
        total = count
    else:
        total += count

if current:
    print(f"{current}\t{total}")
