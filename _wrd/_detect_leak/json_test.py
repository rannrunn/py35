# coding: utf-8
import json
from io import StringIO

print(json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}]))

print(json.dumps("\"foo\bar"))

print(json.dumps({"c": (0, 1), "b": 0, "a": 0}, sort_keys=True))


io = StringIO()
json.dump(['streaming API'], io)
io.getvalue()

