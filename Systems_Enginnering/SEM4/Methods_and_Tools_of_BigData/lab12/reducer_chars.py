#!/usr/bin/env python3


import sys

try:
    current = None
    total = 0

    for line in sys.stdin:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue  # pomiń błędne linie

        char, count = parts
        count = int(count)

        if char != current:
            if current:
                print(f"{current}\t{total}")
            current = char
            total = count
        else:
            total += count

    if current:
        print(f"{current}\t{total}")
except Exception as e:
    print(f"Reducer error: {e}", file=sys.stderr)

