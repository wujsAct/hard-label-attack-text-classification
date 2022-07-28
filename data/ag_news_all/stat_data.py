import csv
import numpy as np
path = 'train.csv'
length = []
length1 = []
for path in ['train.csv', 'test.csv']:
  with open(path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    for line in reader:
      # print(len(line))
      content = line[1] + " " + line[2]
      # print(len(line), len(content.split(" ")), len(line[2].split(" ")))
      label = int(line[0])
      length.append(len(content.split()))
      length1.append(len(line[1].split()))

# print(length)
print(np.sum(length), len(length))
print(np.mean(length), np.max(length))


print(np.sum(length1), len(length1))
print(np.mean(length1), np.max(length1))