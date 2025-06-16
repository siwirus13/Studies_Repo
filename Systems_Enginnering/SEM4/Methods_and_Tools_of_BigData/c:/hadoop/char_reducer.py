import sys
from collections import defaultdict
char_count = defaultdict(int)
for line in sys.stdin:
    char, count = line.strip().split('	')
    char_count[char] += int(count)
for char, count in char_count.items():
    print(f'{char}	{count}')
