import csv
import numpy as np

path = 'train.csv'
length = []
with open(path, "r") as f:
  reader = csv.reader(f, delimiter="\t")
  for line in reader:
    content = line[0]
    label = int(line[1])
    length.append(len(content.split(' ')))

print(np.average(length), np.max(length))