# coding: utf-8

import json

import MySQLdb
import folium
import pandas as pd

# 데이터베이스 접속 정보 설정
con = MySQLdb.connect('192.168.0.7', 'root', '1111', 'SFIOT')
con.set_character_set('utf8')
cur = con.cursor(MySQLdb.cursors.DictCursor)
cur.execute('SET NAMES utf8;')
cur.execute('SET CHARACTER SET utf8;')
cur.execute('SET character_set_connection=utf8;')

# 데이터베이스에서 전주 위치 정보 SELECT
query = """SELECT POLE_LAT, POLE_LON, POLE_CPTZ_NO FROM NT_POLE_MNG
       """
cur.execute(query);
results = cur.fetchall()
df = pd.DataFrame(list(results))

# 폴리움 맵 초기화
map_osm = folium.Map(location=[36.337679, 127.780415], zoom_start=8)

# 데이터베이스에서 전주아이디, 위도, 경도를 읽어와 맵에 추가
for idx in range(len(df)):
    print(idx)
    folium.Marker([float(df['POLE_LAT'][idx]), float(df['POLE_LON'][idx])], popup=df['POLE_CPTZ_NO'][idx], icon=folium.Icon(color='red',icon='info-sign')).add_to(map_osm)


# 맵에 행정구분별 구획을 설정하기 위해 행정구획 정보가 저장된 json 파일 로딩
rfile = open('d:/skorea_provinces_geo_simple.json', 'r', encoding='euckr').read()
jsonData = json.loads(rfile)

# 맵에 행정구분별 구획 설정
folium.GeoJson(jsonData, name='json_data').add_to(map_osm)

# 파일 저장 위치 설정
map_osm.save('d:/map8.html')

