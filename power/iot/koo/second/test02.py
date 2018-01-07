s1 = 'hello'
s2 = 'hello,'
s3 = 'hello'
print(id(s1), id(s2), id(s3))

print(s1 == s2)

print(s1 is s2)



s3 = 'hello, world!'
s4 = 'hello, world!'
print(id(s3), id(s4))

print(s3 == s4)

print(s3 is s4)