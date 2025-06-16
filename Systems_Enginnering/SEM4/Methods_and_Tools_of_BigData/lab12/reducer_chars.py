import sys

current = None
total = 0

for line in sys.stdin:
    char, count = line.strip().split("\t")
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
