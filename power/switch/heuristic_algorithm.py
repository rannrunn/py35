import numpy as np

# circuit_breaker : 과전류 차단 안전장치

dict_section = {}
dict_lateral = {}
arr_lateral = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
arr_lateral_with_support = [1, 2, 3, 4, 5, 6, 7, 8]

# 한개의 feeder 와 한개의 서포팅 feeder 가 있다.
# circuit breaker에 대한 정보는 넣지 않는다.
dict_section['section_number'] = 1
dict_section['feeder_substation_id'] = 'SM64'
dict_section['feeder_capacity'] = 250
dict_section['supporting_feeder_substation_id'] = 'SU73'
dict_section['supporting_feeder_margin'] = 130
dict_section['feeder_tie_switch'] = 'SW9'

# 12개 lateral
dict_lateral['section_number'] = 1
dict_lateral['lateral_number'] = 1
dict_lateral['lateral_load'] = 50
dict_lateral['supporting_lateral_number'] = 13
dict_lateral['supporting_lateral_margin'] = 60
dict_lateral['supporting_lateral_substation_id'] = 'SM67'
dict_lateral['feeder_switch'] = 'SW10'
dict_lateral['supporting_lateral_switch'] = 'SW1'

list = []
for i in arr_lateral_with_support:
    lateral_number = i + 1
    dict_lateral['section_number'] = 1
    dict_lateral['lateral_number'] = 2
    #dict_lateral['lateral_load'] = 50
    dict_lateral['supporting_lateral_number'] = 13
    #dict_lateral['supporting_lateral_margin'] = 60
    dict_lateral['supporting_lateral_substation_id'] = 'SM67'
    dict_lateral['feeder_lateral_switch'] = 'SW10'
    dict_lateral['supporting_lateral_switch'] = 'SW1'


list.append(dict_lateral = {'section_number':'1', 'lateral_number':'1', 'lateral_load':'', 'supporting_lateral_number':'13', 'supporting_lateral_margin':'', 'supporting_lateral_substation_id':'SM67', 'feeder_lateral_switch':'SW10', 'supporting_lateral_switch':'SW1'})
list.append(dict_lateral = {'section_number':'1', 'lateral_number':'2', 'lateral_load':'', 'supporting_lateral_number':'13', 'supporting_lateral_margin':'', 'supporting_lateral_substation_id':'SM67', 'feeder_lateral_switch':'SW10', 'supporting_lateral_switch':'SW1'})
list.append(dict_lateral = {'section_number':'1', 'lateral_number':'3', 'lateral_load':'', 'supporting_lateral_number':'13', 'supporting_lateral_margin':'', 'supporting_lateral_substation_id':'SM67', 'feeder_lateral_switch':'SW10', 'supporting_lateral_switch':'SW1'})
list.append(dict_lateral = {'section_number':'1', 'lateral_number':'4', 'lateral_load':'', 'supporting_lateral_number':'13', 'supporting_lateral_margin':'', 'supporting_lateral_substation_id':'SM67', 'feeder_lateral_switch':'SW10', 'supporting_lateral_switch':'SW1'})
list.append(dict_lateral = {'section_number':'1', 'lateral_number':'5', 'lateral_load':'', 'supporting_lateral_number':'13', 'supporting_lateral_margin':'', 'supporting_lateral_substation_id':'SM67', 'feeder_lateral_switch':'SW10', 'supporting_lateral_switch':'SW1'})
list.append(dict_lateral = {'section_number':'1', 'lateral_number':'6', 'lateral_load':'', 'supporting_lateral_number':'13', 'supporting_lateral_margin':'', 'supporting_lateral_substation_id':'SM67', 'feeder_lateral_switch':'SW10', 'supporting_lateral_switch':'SW1'})
list.append(dict_lateral = {'section_number':'1', 'lateral_number':'7', 'lateral_load':'', 'supporting_lateral_number':'13', 'supporting_lateral_margin':'', 'supporting_lateral_substation_id':'SM67', 'feeder_lateral_switch':'SW10', 'supporting_lateral_switch':'SW1'})
list.append(dict_lateral = {'section_number':'1', 'lateral_number':'8', 'lateral_load':'', 'supporting_lateral_number':'13', 'supporting_lateral_margin':'', 'supporting_lateral_substation_id':'SM67', 'feeder_lateral_switch':'SW10', 'supporting_lateral_switch':'SW1'})


print(np.random.randint(10, 20, size=10))

print(dict)

# 전체 secstion은 supporting lateral margin과 supporting feerder margin으로 복구 가능해야 한다.
# lateral 이 supporting lateral margin 으로 복구되었을 때 여유 margin 은 다른 lateral에 영향을 미치지 않는다.
# supporting lataral로 복구 가능한 lateral을 제외한 lateral 의 load 합이 supporting feeder margin 보다 작아야 한다.
# lateral 9 ~ 12 까지의 load 합은 feeder capacity 보다 작아야 한다.



