import json
from collections import defaultdict
import numpy as np

length=[]
stat_types = defaultdict(int)
with open('train.json', "r", encoding="utf8") as file_:
  for line in file_:
    tt = json.loads(line)
    label = tt['label_desc']
    content = tt['sentence']
    stat_types[label] += 1
    length.append(len(content))

sorted_stat_types = sorted(stat_types.items(), key=lambda x:x[1], reverse=True)
ret=[]
for item in sorted_stat_types:
  ret.append(item[0])

print('、'.join(ret))
print(np.average(length), np.max(length))