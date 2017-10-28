import os

file = 'c:/png/8132D823.png'

# 파일명과 확장자만 얻어오기 : 8132D823.png
print(os.path.basename(file))

# 파일이 들어 있는 디렉토리 얻어오기 : c:/png
print(os.path.dirname(file))

# 파일 경로와 확장자 split 하기 : ('c:/png/8132D823', '.png')
print(os.path.splitext(file))

# 파일명과 확장자 split 하기 : ('8132D823', '.png')
print(os.path.splitext(os.path.basename(file)))
