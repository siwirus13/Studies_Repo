#!/usr/bin/env python3

import sys
from collections import defaultdict

char_counts = defaultdict(int)

for line in sys.stdin:
    for char in line.strip():
        char_counts[char] += 1

for char, count in char_counts.items():
    print(f"{char}\t{count}")
