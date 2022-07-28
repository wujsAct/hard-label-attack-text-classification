import json

f_ = json.load(open("C:/Users/junsh/Desktop/example/flawtolinetocode",'r'))

for key in f_:
  print(key)
  for flaw in f_[key]:
    print(flaw, f_[key][flaw])

