import folium
map_osm = folium.Map(location=[37.566345, 126.977893])

map_osm.save('d:/map1.html')


map_osm = folium.Map(location=[37.566345, 126.977893], zoom_start=17)
map_osm.save('d:/map2.html')




map_osm = folium.Map(location=[37.566345, 126.977893], zoom_start=17)
folium.Marker([37.566345, 126.977893], popup='서울특별시청', icon=folium.Icon(color='red',icon='info-sign')).add_to(map_osm)
folium.CircleMarker([37.5658859, 126.9754788], radius=100,color='#3186cc',fill_color='#3186cc', popup='덕수궁').add_to(map_osm)
map_osm.save('d:/map3.html')

