import sys
for line in sys.stdin:
    for char in line.strip():
        print(f'{char}	1')
