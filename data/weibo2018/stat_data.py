import csv
import numpy as np
path = 'train.txt'
length = []
with open(path, "r", encoding="utf8") as f:
  reader = csv.reader(f, delimiter=",")
  for line in reader:
    content = line[2]
    label = int(line[1])
    length.append(len(content))

print(np.average(length), np.max(length))