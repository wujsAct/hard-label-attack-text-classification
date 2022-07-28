from ltp import LTP


ltp = LTP()

seg, hidden = ltp.seg(["日本虎视眈眈武力夺岛美军向俄后院开火普京终不再忍"])
print(seg)
# print(hidden)
ner = ltp.ner(hidden)

print(ner)

