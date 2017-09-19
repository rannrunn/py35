import json

data = {}
data['key1'] = 'value1'
data['key2'] = ["key2_1",{"key2_1_1":["value2_1_1"]}]
json_data = json.dumps(data)

print(json_data)
print(type(json_data)) # json 형식의 type은 str이다.

result = json.loads(json_data)

print(result["key2"])
print(type(result))