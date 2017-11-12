import numpy as np


list_list = []
initial_list_dict = []

initial_list_dict.append({'type': 'feeder', 'id': 'F1', 'substation_id': 'SM64', 'rated_capacity': '450', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'feeder', 'id': 'F2', 'substation_id': 'SU73', 'rated_capacity': '450', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'feeder', 'id': 'F3', 'substation_id': 'SM67', 'rated_capacity': '450', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'feeder', 'id': 'F4', 'substation_id': 'SM70', 'rated_capacity': '450', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'feeder', 'id': 'F5', 'substation_id': 'ZH63', 'rated_capacity': '450', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'feeder', 'id': 'F6', 'substation_id': 'ZH59', 'rated_capacity': '450', 'load': '', 'capacity_margin': ''})

initial_list_dict.append({'type': 'group', 'id': 'A', 'feeder': 'F1', 'lateral':[], 'lateral_load_res':[], 'lateral_load_remain':[]})
initial_list_dict.append({'type': 'group', 'id': 'B', 'feeder': 'F1', 'lateral':[], 'lateral_load_res':[], 'lateral_load_remain':[]})
initial_list_dict.append({'type': 'group', 'id': 'C', 'feeder': 'F1', 'lateral':[], 'lateral_load_res':[], 'lateral_load_remain':[]})

initial_list_dict.append({'type': 'lateral', 'id': 'LAT1', 'feeder': 'F1', 'lateral_tie_switch': 'SW1', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT2', 'feeder': 'F1', 'lateral_tie_switch': 'SW2', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT3', 'feeder': 'F1', 'lateral_tie_switch': 'SW3', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT4', 'feeder': 'F1', 'lateral_tie_switch': 'SW4', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT5', 'feeder': 'F1', 'lateral_tie_switch': 'SW5', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT6', 'feeder': 'F1', 'lateral_tie_switch': 'SW6', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT7', 'feeder': 'F1', 'lateral_tie_switch': 'SW7', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT8', 'feeder': 'F1', 'lateral_tie_switch': 'SW8', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT9', 'feeder': 'F1', 'lateral_tie_switch': '', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT10', 'feeder': 'F1', 'lateral_tie_switch': '', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT11', 'feeder': 'F1', 'lateral_tie_switch': '', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT12', 'feeder': 'F1', 'lateral_tie_switch': '', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT13', 'feeder': 'F3', 'lateral_tie_switch': 'SW1', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT14', 'feeder': 'F3', 'lateral_tie_switch': 'SW2', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT15', 'feeder': 'F3', 'lateral_tie_switch': 'SW3', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT16', 'feeder': 'F4', 'lateral_tie_switch': 'SW4', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT17', 'feeder': 'F4', 'lateral_tie_switch': 'SW5', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT18', 'feeder': 'F5', 'lateral_tie_switch': 'SW6', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT19', 'feeder': 'F6', 'lateral_tie_switch': 'SW7', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})
initial_list_dict.append({'type': 'lateral', 'id': 'LAT20', 'feeder': 'F6', 'lateral_tie_switch': 'SW8', 'rated_capacity': '100', 'load': '', 'capacity_margin': ''})

initial_list_dict.append({'type': 'switch', 'switch_type': 'feeder_tie_switch', 'id': 'SW9', 'point':{'1': 'SM64', '2': 'SU73'}})
initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW1', 'point':{'1': 'LAT1', '2': 'LAT13'}})
initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW2', 'point':{'1': 'LAT2', '2': 'LAT14'}})
initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW3', 'point':{'1': 'LAT3', '2': 'LAT15'}})
initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW4', 'point':{'1': 'LAT4', '2': 'LAT16'}})
initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW5', 'point':{'1': 'LAT5', '2': 'LAT17'}})
initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW6', 'point':{'1': 'LAT6', '2': 'LAT18'}})
initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW7', 'point':{'1': 'LAT7', '2': 'LAT19'}})
initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW8', 'point':{'1': 'LAT8', '2': 'LAT20'}})

initial_list_dict.append({'type': 'branching_point', 'id': '07' , 'section_type': 'feeder', 'section_id': 'F1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '17' , 'section_type': 'feeder', 'section_id': 'F1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '32' , 'section_type': 'feeder', 'section_id': 'F1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '33' , 'section_type': 'feeder', 'section_id': 'F1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '41' , 'section_type': 'feeder', 'section_id': 'F1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '42' , 'section_type': 'feeder', 'section_id': 'F1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '48' , 'section_type': 'feeder', 'section_id': 'F1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})

initial_list_dict.append({'type': 'branching_point', 'id': '03' , 'section_type': 'lateral', 'section_id': 'LAT1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '02' , 'section_type': 'lateral', 'section_id': 'LAT1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '18' , 'section_type': 'lateral', 'section_id': 'LAT2', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '19' , 'section_type': 'lateral', 'section_id': 'LAT2', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '22' , 'section_type': 'lateral', 'section_id': 'LAT2', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '27' , 'section_type': 'lateral', 'section_id': 'LAT3', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '28' , 'section_type': 'lateral', 'section_id': 'LAT3', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '30' , 'section_type': 'lateral', 'section_id': 'LAT3', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '34' , 'section_type': 'lateral', 'section_id': 'LAT4', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '37' , 'section_type': 'lateral', 'section_id': 'LAT4', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '38' , 'section_type': 'lateral', 'section_id': 'LAT4', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '09' , 'section_type': 'lateral', 'section_id': 'LAT5', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '10' , 'section_type': 'lateral', 'section_id': 'LAT5', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '14' , 'section_type': 'lateral', 'section_id': 'LAT5', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '43' , 'section_type': 'lateral', 'section_id': 'LAT6', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '44' , 'section_type': 'lateral', 'section_id': 'LAT6', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '46' , 'section_type': 'lateral', 'section_id': 'LAT6', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '60' , 'section_type': 'lateral', 'section_id': 'LAT7', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '61' , 'section_type': 'lateral', 'section_id': 'LAT7', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '62' , 'section_type': 'lateral', 'section_id': 'LAT7', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '49' , 'section_type': 'lateral', 'section_id': 'LAT8', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '50' , 'section_type': 'lateral', 'section_id': 'LAT8', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '52' , 'section_type': 'lateral', 'section_id': 'LAT8', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})

initial_list_dict.append({'type': 'branching_point', 'id': '56' , 'section_type': 'lateral', 'section_id': 'LAT9', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '59' , 'section_type': 'lateral', 'section_id': 'LAT10', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '58' , 'section_type': 'lateral', 'section_id': 'LAT11', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
initial_list_dict.append({'type': 'branching_point', 'id': '20' , 'section_type': 'lateral', 'section_id': 'LAT12', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''}) # id가 18 이었으나 id가 중복되어 20으로 변경

# 스위치 조작 순서
initial_list_dict.append({'type':'switch_operation_order', 'result':[]})
# output
initial_list_dict.append({'type':'output', 'seq':'', 'result':{'LAT1':'', 'LAT2':'', 'LAT3':'', 'LAT4':'', 'LAT5':'', 'LAT6':'', 'LAT7':'', 'LAT8':''}})

print(np.random.randint(10, 20, size=10))

print(dict)

# 전체 secstion은 supporting lateral margin과 supporting feerder margin으로 복구 가능해야 한다.
# lateral 이 supporting lateral margin 으로 복구되었을 때 여유 margin 은 다른 lateral에 영향을 미치지 않는다.
# supporting lataral로 복구 가능한 lateral을 제외한 lateral 의 load 합이 supporting feeder margin 보다 작아야 한다.
# lateral 9 ~ 12 까지의 load 합은 feeder capacity 보다 작아야 한다.

# F1에 속한 레터럴의 로드 생성
# 레터럴에 속한 분기점(branching point)의 총로드는 레터럴의 정격용량(rated capacity)를 넘을 수 없다.
# 모든 레터럴에 대한 로드의 총 합은 피더에서 피더의 정격용량(rated capacity)를 넘을 수 없다.
def setLateral(list_dict):

    # 반복을 위한 bool 설정 : 모든 분기점에 대한 load가 생성되었을 때 각 레터럴의 총로드가 100 이하이고 피더의 총로드가 450 이하일 경우 bool == False
    bool_while = True

    while bool_while:

        # 피더의 총로드
        total_feeder_load = 0

        # dict LIST 검색 중단 여부 설정
        bool_for = True

        # 모든 레터럴 중
        for i in range(len(list_dict)):
            dict = list_dict[i]
            if dict['type'] == 'lateral' and dict['feeder'] == 'F1':
                lateral_id = dict['id']
                # 레터럴의 총로드
                lateral_load = 0
                for j in range(len(list_dict)):
                    # branching point 의 json
                    json_bp = list_dict[j]
                    if json_bp['type'] == 'branching_point' and json_bp['section_id'] == lateral_id:
                        # 1에서 33 사이의 수 중에서 난수 생성
                        load = np.random.randint(1, 24)
                        lateral_load += load
                        total_feeder_load += load
                        #print('load:', load)

                        json_bp['load'] = load

                        list_dict[j] = json_bp

                    # 조건이 만족되지 않았을 경우 for문 종료
                    # 피더의 총로드가 450을 초과하거나 레터럴의 총로드가 100을 초과할 경우
                    if total_feeder_load > 450 or lateral_load > 100:
                        bool_for = False
                        break

                #print('lateral_id:', lateral_id)
                #print('total_lateral_load:', total_lateral_load)

                # 레털의 총로드 저장
                dict['load'] = lateral_load
                dict['capacity_margin'] = int(dict['rated_capacity']) - lateral_load
                list_dict[i] = dict

            # 조건이 만족되지 않았을 경우 for문 종료
            if bool_for == False:
                break

        # F1에 총로드와 여유 용량을 저장
        for i in range(len(list_dict)):
            dict = list_dict[i]
            if dict['type'] == 'feeder' and dict['id'] == 'F1':
                dict['load'] = total_feeder_load
                dict['capacity_margin'] = int(dict['rated_capacity']) - total_feeder_load

            list_dict[i] = dict

        #print('total_feeder_load:', total_feeder_load)

        # for문이 이상 없이 종료되었을 경우 while문은 종료
        if bool_for == True:
            bool_while = False

    print('setLateralLoad end')

    return list_dict

# 서포팅 피더의 여유 용량 설정
def setSupportingFeeder(list_dict):

    idx, dict = getDict(list_dict, [['type', 'feeder'], ['id', 'F2']], [])
    # 서포팅 피더의 로드를 220에서 270 사이로 랜덤하게 생성
    load = np.random.randint(50, 250)
    dict['load'] = load
    # 서포팅 피더의 여유 용량 계산
    dict['capacity_margin'] = int(dict['rated_capacity']) - load

    list_dict[idx] = dict

    return list_dict

# 서포팅 레터럴의 로드 설정
def setSupportingLateral(list_dict):
    for i in range(len(list_dict)):
        dict = list_dict[i]
        if dict['type'] == 'lateral' and dict['feeder'] != 'F1':
            load = 0
            for i in range(3):
                load += np.random.randint(1, 33)

            dict['load'] = load
            dict['capacity_margin'] = int(dict['rated_capacity']) - load

            list_dict[i] = dict

    return list_dict

# 그룹 설정
def setGroup(list_dict):
    idx_group_a, dict_group_a = getDict(list_dict, [['type', 'group'], ['id', 'A']], [])
    idx_group_b, dict_group_b = getDict(list_dict, [['type', 'group'], ['id', 'B']], [])
    idx_group_c, dict_group_c = getDict(list_dict, [['type', 'group'], ['id', 'C']], [])
    for i in range(len(list_dict)):
        dict = list_dict[i]
        if dict['type'] == 'lateral' and dict['feeder'] == 'F1':
            if dict['lateral_tie_switch'] != '':
                # 서포팅 레터럴의 index와 dictionary를 가져온다.
                idx_supporting_lateral, dict_supporting_lateral = getDict(list_dict, [['type', 'lateral'], ['lateral_tie_switch', dict['lateral_tie_switch']]], [['id', dict['id']]])

                # 서포팅 레터럴은 레터럴과 같은 레터럴 타이 스위치에 연결되어 있고 id가 다르다.
                supporting_lateral_margin = int(dict_supporting_lateral['capacity_margin'])
                load = int(dict['load'])

                if load > supporting_lateral_margin:
                    dict_group_a['lateral'].append(dict['id'])
                else:
                    dict_group_b['lateral'].append(dict['id'])
            else:
                dict_group_c['lateral'].append(dict['id'])

    return list_dict

# index와 dictionary 데이터를 리턴
def getDict(list_dict, list_if_true, list_if_false):
    dict = {}
    for idx in range(len(list_dict)):
        dict = list_dict[idx]
        bool = True

        # list_if_yes 와 같은 데이터가 있을 경우 True
        for key, value in list_if_true:
            if key in dict.keys() and dict[key] != value:
                bool = False

        # list_if_no 와 다른 데이터가 있을 경우 True
        for key, value in list_if_false:
            if key not in dict.keys() or (key in dict.keys() and dict[key] == value):
                bool = False

        # bool이 True일 경우 return
        if bool == True:
            return idx, dict
    return -1, dict

# 그룹 B 의 총로드 계산
def getTotalGroupBLoad(list_dict):
    result = 0
    # 그룹 B dict
    idx_group_b, dict_group_b = getDict(list_dict, [['type', 'group'], ['id', 'B']], [])
    # 그룹 B에 속하는 lateral
    list_group_b = dict_group_b['lateral']
    # 그룹 B에 속하는 lateral의 총 로드 계산
    for id in list_group_b:
        idx_lateral, dict_lateral = getDict(list_dict, [['type', 'lateral'], ['id', id]], [])
        load = dict_lateral['load']
        result += load
    return result

def getTotalGroupBResLoad(list_dict):
    pass

def algorithm_one():
    pass

def algorithm_two():
    pass

def algorithm_three():
    pass

# 순서도
def flowChart(list_dict):

    idx_feeder, dict_feeder = getDict(list_dict, [['type', 'feeder'], ['id', 'F1']], [])
    idx_supporting_feeder, dict_supporting_feeder = getDict(list_dict, [['type', 'feeder'], ['id', 'F2']], [])
    supporting_feeder_margin = dict_supporting_feeder['capacity_margin']
    total_lateral_load = dict_feeder['load']
    total_group_b_load = getTotalGroupBLoad(list_dict)
    total_group_b_res_laod = getTotalGroupBResLoad(list_dict)

    print(supporting_feeder_margin)
    print(total_lateral_load)
    print(total_group_b_load)
    print(total_group_b_res_laod)
    print(list_dict)

    # 분기 1 사전 계산

    # 분기 1
    if supporting_feeder_margin > total_lateral_load:
        return

    # 분기 2
    if supporting_feeder_margin > total_group_b_load:
        return

    # 블록 3

    # 분기 4
    if supporting_feeder_margin > total_group_b_load:
        return

    # 분기 5
    if supporting_feeder_margin > total_group_b_load - total_group_b_res_laod:
        return

    # 블록 6

    # 분기 7
    if supporting_feeder_margin > total_group_b_load - total_group_b_res_laod:
        return



    return list_dict

    pass

def main():

    data_cnt = 1

    # 생성할 데이터의 개수를 range에 설정
    for i in range(data_cnt):

        list_dict = initial_list_dict

        # 레터럴의 로드 및 여유 용량 설정
        list_dict = setLateral(list_dict)

        # 서포팅 피더의 여유 용량 설정
        list_dict = setSupportingFeeder(list_dict)

        # 서포팅 레터럴의 로드 및 여유 용량 설정
        list_dict = setSupportingLateral(list_dict)

        # 레터럴을 그룹에 분배
        # 그룹의 로드 등의 계산은 순서도가 실행될 때 계산한다. (알고리즘의 일관성을 위해)
        list_dict = setGroup(list_dict)

        # 순서도 실행
        # 1. 결과 생성
        # 2. 결과에는 순서도 블록 및 선택 알고리즘이 들어 간다.
        list_dict = flowChart(list_dict)

        # list 에 추가
        # 1. 9번에 도달한 데이터는 list에 추가하지 않는다.
        list_list.append(list_dict)

        # list를 csv나 다른 파일로 추출하는 것은 하지 않는다.(다시 불러오기 힘들기 때문)

        print(list_dict)

    pass

if __name__ == '__main__':
    main()
