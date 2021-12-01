with open("documents.txt") as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

lines.sort()
for i in lines:
    print(i)