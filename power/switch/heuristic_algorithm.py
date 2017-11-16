import numpy as np
import os

def initial(initial_list_dict):

    # 피더
    initial_list_dict.append({'type': 'feeder', 'id': 'F1', 'substation_id': 'SM64', 'rated_capacity': '450', 'load': '', 'capacity_margin': ''})
    initial_list_dict.append({'type': 'feeder', 'id': 'F2', 'substation_id': 'SU73', 'rated_capacity': '450', 'load': '', 'capacity_margin': ''})
    initial_list_dict.append({'type': 'feeder', 'id': 'F3', 'substation_id': 'SM67', 'rated_capacity': '450', 'load': '', 'capacity_margin': ''})
    initial_list_dict.append({'type': 'feeder', 'id': 'F4', 'substation_id': 'SM70', 'rated_capacity': '450', 'load': '', 'capacity_margin': ''})
    initial_list_dict.append({'type': 'feeder', 'id': 'F5', 'substation_id': 'ZH63', 'rated_capacity': '450', 'load': '', 'capacity_margin': ''})
    initial_list_dict.append({'type': 'feeder', 'id': 'F6', 'substation_id': 'ZH59', 'rated_capacity': '450', 'load': '', 'capacity_margin': ''})

    # 그룹
    initial_list_dict.append({'type': 'group', 'id': 'A', 'feeder': 'F1', 'lateral':[], 'lateral_load_res':[], 'lateral_load_remain':[]})
    initial_list_dict.append({'type': 'group', 'id': 'B', 'feeder': 'F1', 'lateral':[], 'lateral_load_res':[], 'lateral_load_remain':[]})
    initial_list_dict.append({'type': 'group', 'id': 'C', 'feeder': 'F1', 'lateral':[], 'lateral_load_res':[], 'lateral_load_remain':[]})

    # 레터럴
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT1', 'feeder': 'F1', 'feeder_branching_point': '7', 'lateral_tie_switch': 'SW1', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':['3', '2'], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT2', 'feeder': 'F1', 'feeder_branching_point': '17', 'lateral_tie_switch': 'SW2', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':['18', '19', '22'], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT3', 'feeder': 'F1', 'feeder_branching_point': '17', 'lateral_tie_switch': 'SW3', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':['27', '28', '30'], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT4', 'feeder': 'F1', 'feeder_branching_point': '33', 'lateral_tie_switch': 'SW4', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':['34', '37', '38'], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT5', 'feeder': 'F1', 'feeder_branching_point': '33', 'lateral_tie_switch': 'SW5', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':['9', '10', '14'], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT6', 'feeder': 'F1', 'feeder_branching_point': '42', 'lateral_tie_switch': 'SW6', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':['43', '44', '46'], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT7', 'feeder': 'F1', 'feeder_branching_point': '48', 'lateral_tie_switch': 'SW7', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':['60', '61', '62'], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT8', 'feeder': 'F1', 'feeder_branching_point': '48', 'lateral_tie_switch': 'SW8', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':['49', '50', '52'], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT9', 'feeder': 'F1', 'feeder_branching_point': '7', 'lateral_tie_switch': '', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':[], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT10', 'feeder': 'F1', 'feeder_branching_point': '32', 'lateral_tie_switch': '', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':[], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT11', 'feeder': 'F1', 'feeder_branching_point': '41', 'lateral_tie_switch': '', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':[], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT12', 'feeder': 'F1', 'feeder_branching_point': '42', 'lateral_tie_switch': '', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':[], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT13', 'feeder': 'F3', 'feeder_branching_point': '', 'lateral_tie_switch': 'SW1', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':[], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT14', 'feeder': 'F3', 'feeder_branching_point': '', 'lateral_tie_switch': 'SW2', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':[], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT15', 'feeder': 'F3', 'feeder_branching_point': '', 'lateral_tie_switch': 'SW3', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':[], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT16', 'feeder': 'F4', 'feeder_branching_point': '', 'lateral_tie_switch': 'SW4', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':[], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT17', 'feeder': 'F4', 'feeder_branching_point': '', 'lateral_tie_switch': 'SW5', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':[], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT18', 'feeder': 'F5', 'feeder_branching_point': '', 'lateral_tie_switch': 'SW6', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':[], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT19', 'feeder': 'F6', 'feeder_branching_point': '', 'lateral_tie_switch': 'SW7', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':[], 'branching_point_res':[]})
    initial_list_dict.append({'type': 'lateral', 'id': 'LAT20', 'feeder': 'F6', 'feeder_branching_point': '', 'lateral_tie_switch': 'SW8', 'rated_capacity': '100', 'load': '', 'capacity_margin': '', 'branching_point':[], 'branching_point_res':[]})

    # 스위치
    initial_list_dict.append({'type': 'switch', 'switch_type': 'feeder_tie_switch', 'id': 'SW9', 'point':{'1': 'SM64', '2': 'SU73'}})
    initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW1', 'point':{'1': 'LAT1', '2': 'LAT13'}})
    initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW2', 'point':{'1': 'LAT2', '2': 'LAT14'}})
    initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW3', 'point':{'1': 'LAT3', '2': 'LAT15'}})
    initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW4', 'point':{'1': 'LAT4', '2': 'LAT16'}})
    initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW5', 'point':{'1': 'LAT5', '2': 'LAT17'}})
    initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW6', 'point':{'1': 'LAT6', '2': 'LAT18'}})
    initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW7', 'point':{'1': 'LAT7', '2': 'LAT19'}})
    initial_list_dict.append({'type': 'switch', 'switch_type': 'lateral_tie_switch', 'id': 'SW8', 'point':{'1': 'LAT8', '2': 'LAT20'}})

    # 브랜칭 포인트 (피더에 포함된)
    initial_list_dict.append({'type': 'branching_point', 'id': '7' , 'section_type': 'feeder', 'section_id': 'F1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '17' , 'section_type': 'feeder', 'section_id': 'F1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '32' , 'section_type': 'feeder', 'section_id': 'F1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '33' , 'section_type': 'feeder', 'section_id': 'F1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '41' , 'section_type': 'feeder', 'section_id': 'F1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '42' , 'section_type': 'feeder', 'section_id': 'F1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '48' , 'section_type': 'feeder', 'section_id': 'F1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})

    # 브랜칭 포인트 (그룹 A 또는 B)
    initial_list_dict.append({'type': 'branching_point', 'id': '3' , 'section_type': 'lateral', 'section_id': 'LAT1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '2' , 'section_type': 'lateral', 'section_id': 'LAT1', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '18' , 'section_type': 'lateral', 'section_id': 'LAT2', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '19' , 'section_type': 'lateral', 'section_id': 'LAT2', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '22' , 'section_type': 'lateral', 'section_id': 'LAT2', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '27' , 'section_type': 'lateral', 'section_id': 'LAT3', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '28' , 'section_type': 'lateral', 'section_id': 'LAT3', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '30' , 'section_type': 'lateral', 'section_id': 'LAT3', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '34' , 'section_type': 'lateral', 'section_id': 'LAT4', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '37' , 'section_type': 'lateral', 'section_id': 'LAT4', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '38' , 'section_type': 'lateral', 'section_id': 'LAT4', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '9' , 'section_type': 'lateral', 'section_id': 'LAT5', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
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

    # 브랜칭 포인트 (그룹 C)
    initial_list_dict.append({'type': 'branching_point', 'id': '56' , 'section_type': 'lateral', 'section_id': 'LAT9', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '59' , 'section_type': 'lateral', 'section_id': 'LAT10', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '58' , 'section_type': 'lateral', 'section_id': 'LAT11', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''})
    initial_list_dict.append({'type': 'branching_point', 'id': '20' , 'section_type': 'lateral', 'section_id': 'LAT12', 'switch':{'1': '', '2': '', '3': '', '4': ''}, 'load': ''}) # id가 18 이었으나 id가 중복되어 20으로 변경

    # 결과
    initial_list_dict.append({'type':'result', 'seq':'', 'flow_chart_result':'', 'switch_operation_order':[], 'ann_input':[0, 0, 0, 0, 0, 0, 0, 0, 0], 'ann_output':[0, 0, 0, 0, 0, 0, 0, 0]})

    return initial_list_dict


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
                        laod = 0
                        # 레터럴 1은 두 개 브랜칭 포인트 밖에 없으므로 최대값이 다른 브랜칭 포인트에 비해 크게 생성한다.
                        if lateral_id == 'LAT1':
                            # 난수 생성
                            load = np.random.randint(branching_point_lateral_1_load_min, branching_point_lateral_1_load_max)
                        else:
                            # 난수 생성
                            load = np.random.randint(branching_point_other_lateral_load_min, branching_point_other_lateral_load_max)

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

    return list_dict

# 서포팅 피더의 여유 용량 설정
def setSupportingFeeder(list_dict):

    idx, dict = getDict(list_dict, [['type', 'feeder'], ['id', 'F2']], [])
    # 서포팅 피더의 로드를 50에서 250 사이로 랜덤하게 생성
    load = np.random.randint(supporting_feeder_load_min, supporting_feeder_load_max)
    dict['load'] = load
    # 서포팅 피더의 여유 용량 계산
    dict['capacity_margin'] = int(dict['rated_capacity']) - load

    list_dict[idx] = dict

    return list_dict

# 서포팅 레터럴의 부하 및 여유 용량 설정
def setSupportingLateral(list_dict):
    for idx in range(len(list_dict)):
        dict = list_dict[idx]
        if dict['type'] == 'lateral' and dict['feeder'] != 'F1':
            load = 0
            for _ in range(3):
                # 5에서 33 까지의 난수를 생성
                # 첫번 째 인자를 5보다 더 큰 값을 사용할 경우 서포팅 레터럴로 중단된 레터럴의 부하를 복구하기 힘든 경우가 많아진다.
                load += np.random.randint(branching_point_supporting_lateral_load_min, branching_point_supporting_lateral_load_max)

            dict['load'] = load
            dict['capacity_margin'] = int(dict['rated_capacity']) - load

            list_dict[idx] = dict

    return list_dict


# 파일을 읽어 레터럴의 로드 및 여유 용량 설정
def setLateralFromLine(list_dict, line):

    list_line_branching_point = line.split('[')[1].replace(']','').split(',')

    list_branching_point_id = ['3','2','18','19','22','27','28','30','34','37','38','9','10','14','43','44','46','60','61','62','49','50','52','56','59','58','20']
    for idx in range(len(list_line_branching_point)):
        item = list_line_branching_point[idx]
        idx_dict, dict_branching_point = getDictBranchingPoint(list_dict, list_branching_point_id[idx])

        # 브랜칭 포인트의 load 를 집어 넣는다.
        dict_branching_point['load'] = item
        list_dict[idx_dict] = dict_branching_point

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
                    load = int(json_bp['load'])
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

    return list_dict


# 파일을 읽어 서포팅 피더의 여유 용량 설정
def setSupportingFeederFromLine(list_dict, line):
    idx_supporting_feeder, dict_supporting_feeder = getDict(list_dict, [['type', 'feeder'], ['id', 'F2']], [])
    dict_supporting_feeder['capacity_margin'] = int(line.split('[')[2].replace(']','').split(',')[0])
    list_dict[idx_supporting_feeder] = dict_supporting_feeder
    return list_dict


# 파일읅 읽어 서포팅 레터럴의 로드 및 여유 용량 설정
def setSupportingLateralFromLine(list_dict, line):
    list_supporting_lateral_id = ['LAT13', 'LAT14', 'LAT15', 'LAT16', 'LAT17', 'LAT18', 'LAT19', 'LAT20']
    list_supporting_lateral = line.split('[')[3].replace(']','').split(',')
    for idx in range(len(list_supporting_lateral_id)):
        item = list_supporting_lateral_id[idx]
        idx_supporting_lateral, dict_supporting_lateral = getDictLateral(list_dict, item)
        dict_supporting_lateral['capacity_margin'] = int(list_supporting_lateral[idx])
        list_dict[idx_supporting_lateral] = dict_supporting_lateral
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
                # 서포팅 레터럴의 dictionary를 가져온다.
                dict_supporting_lateral = getDictSupportingLateral(list_dict, dict['id'])[1]

                # 서포팅 레터럴은 레터럴과 같은 레터럴 타이 스위치에 연결되어 있고 id가 다르다.
                supporting_lateral_margin = int(dict_supporting_lateral['capacity_margin'])
                load = int(dict['load'])

                if load <= supporting_lateral_margin:
                    dict_group_a['lateral'].append(dict['id'])
                else:
                    dict_group_b['lateral'].append(dict['id'])
            else:
                dict_group_c['lateral'].append(dict['id'])

    list_dict[idx_group_a] = dict_group_a
    list_dict[idx_group_b] = dict_group_b
    list_dict[idx_group_c] = dict_group_c

    print('그룹 A 딕셔너리:', dict_group_a)
    print('그룹 B 딕셔너리:', dict_group_b)
    print('그룹 C 딕셔너리:', dict_group_c)

    return list_dict

# index와 dictionary 데이터를 리턴
def getDict(list_dict, list_if_true, list_if_false):
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
    return -1, {}

# 그룹의 총로드 계산
def getTotalGroupLoad(list_dict, group_id):
    result = 0
    # 그룹의 딕셔너리
    idx_group, dict_group = getDict(list_dict, [['type', 'group'], ['id', group_id]], [])
    # 그룹에 속하는 lateral
    list_group = dict_group['lateral']
    # 그룹에 속하는 lateral의 총 로드 계산
    for id in list_group:
        idx_lateral, dict_lateral = getDict(list_dict, [['type', 'lateral'], ['id', id]], [])
        load = dict_lateral['load']
        result += load
    return result

# 서포팅 레터럴로 복구가 가능한 부하의 총계를 계산
def getTotalGroupBResLoad(list_dict):
    total_group_b_res_load = 0
    # 그룹 B dict
    idx_group_b, dict_group_b = getDict(list_dict, [['type', 'group'], ['id', 'B']], [])
    # 그룹 B에 속하는 lateral
    list_group_b = dict_group_b['lateral']
    # 그룹 B에 속하는 lateral의 총 로드 계산
    for id in list_group_b:
        idx_lateral, dict_lateral = getDict(list_dict, [['type', 'lateral'], ['id', id]], [])
        dict_supporting_lateral = getDictSupportingLateral(list_dict, dict_lateral['id'])[1]
        supporting_lateral_margin = dict_supporting_lateral['capacity_margin']
        list_res_branching_point = []
        load_res = 0
        # 레터럴 딕셔너리에서 브랜칭 포인트 리스트를 역순으로 검색 (서포팅 레터럴에 가까운 브랜칭 포인트부터 가져오기 위해)
        for idx in range(len(dict_lateral['branching_point']) -1, -1, -1):
            # 레터럴 딕셔너리에 들어있는 브랜칭 포인트를 리스트로 가져옴
            list_branching_point = dict_lateral['branching_point']
            # 브랜칭 포인트 리스트에서 브랜칭 포인트 아이디를 가져옴
            branching_point_id = list_branching_point[idx]
            # 브랜칭 포인트를 가지고 브랜칭 포인트 딕셔너리를 가져옴
            idx_branching_point, dict_branching_point = getDict(list_dict, [['type', 'branching_point'], ['id', branching_point_id]], [])
            # 브랜칭 포인트 딕셔너리에서 브랜칭 포인트의 부하를 가져옴
            load_branching_point = int(dict_branching_point['load'])
            # 서포팅 레터럴의 여유 용량에서 브랜칭 포인트 들의 총 부하가 0보다 클 경우에만 내용 실행
            if supporting_lateral_margin - (load_res + load_branching_point) >= 0:
                # 브랜칭 포인트의 부하를 복구 가능한 부하(레터럴에 적용)에 더해준다.
                load_res += load_branching_point
                # 브랜칭 포인트의 부하를 복구 가능한 부하(그룹에 적용)에 더해준다.
                total_group_b_res_load += load_branching_point
                # 브랜칭 포인트의 id를 복구 가능한 브랜칭 포인트 리스트(레터럴에 적용)에 추가한다.
                list_res_branching_point.append(branching_point_id)
            else:
                break

        dict_lateral['branching_point_res'] = list_res_branching_point

    return total_group_b_res_load


# 삽입 정렬을 이용해 list 의 레터럴을 부하에 따라 오름차순으로 정렬한다.
def sortListAsLoad_insertionSort(list_dict, list):
    for size in range(1, len(list)):
        key = list[size]
        val = int(getDict(list_dict, [['type', 'lateral'], ['id', list[size]]], [])[1]['load'])
        i = size
        while i > 0:
            val_pre = int(getDict(list_dict, [['type', 'lateral'], ['id', list[i - 1]]], [])[1]['load'])
            if val_pre > val:
                list[i] = list[i - 1]
                i -= 1
            else :
                break

        list[i] = key

    return list


# 그룹의 리스트를 오름차순으로 정렬한다.
def setSortGroupList(list_dict, group_id):

    dict_group = getDict(list_dict, [['type', 'group'], ['id', group_id]], [])[1]
    list_group_lateral = dict_group['lateral']
    list_sort_group_lateral = sortListAsLoad_insertionSort(list_dict, list_group_lateral)

    print('오름차순으로 정렬된 그룹', group_id, ':', list_sort_group_lateral)
    list_val = []
    for idx in range(len(list_sort_group_lateral)):
        list_val.append(int(getDict(list_dict, [['type', 'lateral'], ['id', list_sort_group_lateral[idx]]], [])[1]['load']))
    print('오름차순으로 정렬된 그룹', group_id, '의 값:', list_val)

    return list_sort_group_lateral


# 브랜칭 포인트의 딕셔너리를 가져온다.
def getDictBranchingPoint(list_dict, branching_point_id):
    return getDict(list_dict, [['type', 'branching_point'], ['id', branching_point_id]], [])


# 레터럴을 이용해 레터럴의 딕셔너리를 가져온다.
def getDictLateral(list_dict, lateral_id):
    # 레터럴 타이 스위치를 가져온다.
    return getDict(list_dict, [['type', 'lateral'], ['id', lateral_id]], [])


# 레터럴에 대응하는 서포팅 레터럴의 딕셔너리를 가져온다.
def getDictSupportingLateral(list_dict, lateral_id):
    # 레터럴 타이 스위치를 가져온다.
    lateral_tie_switch = getDict(list_dict, [['type', 'lateral'], ['id', lateral_id]], [])[1]['lateral_tie_switch']
    # 레터럴 타이 스위치가 같고 레터럴과 id가 다른며 레터럴 타이 스위치가 ''가 아닌 것이 서포팅 레터럴이다.
    return getDict(list_dict, [['type', 'lateral'], ['lateral_tie_switch', lateral_tie_switch]], [['id', lateral_id], ['lateral_tie_switch', '']])


# 브랜칭 포인트의 부하를 가져온다.
def getBranchingPointLoad(list_dict, branching_point_id):
    return int(getDict(list_dict, [['type', 'branching_point'], ['id', branching_point_id]], [])[1]['load'])


# 레터럴의 부하를 가져온다.
def getLateralLoad(list_dict, lateral_id):
    return int(getDict(list_dict, [['type', 'lateral'], ['id', lateral_id]], [])[1]['load'])


# 서포팅 레터럴의 부하를 가져온다.
def getSupportingLateralLoad(list_dict, lateral_id):
    # 레터럴 타이 스위치를 가져온다.
    lateral_tie_switch = getDict(list_dict, [['type', 'lateral'], ['id', lateral_id]], [])[1]['lateral_tie_switch']
    # 레터럴 타이 스위치가 같고 레터럴과 id가 다른며 레터럴 타이 스위치가 ''가 아닌 것이 서포팅 레터럴이다.
    dict_supporting_lateral = getDict(list_dict, [['type', 'lateral'], ['lateral_tie_switch', lateral_tie_switch]], [['id', lateral_id], ['lateral_tie_switch', '']])[1]
    # 서포팅 레터럴의 부하를 가져온다.
    return int(dict_supporting_lateral['load'])


# 삼항 연산자 : a if test else b
# 서포팅 레터럴로 복구 가능한 부하를 가져온다.
def getLateralResLoad(list_dict, lateral_id):
    # 레터럴의 부하
    load_lateral = getLateralLoad(list_dict, lateral_id)
    # 서포팅 레터럴의 부하
    load_supporting_lateral = getSupportingLateralLoad(list_dict, lateral_id)
    # 복구 가능한 레터럴의 부하
    return load_lateral if load_supporting_lateral - load_lateral >= 0 else load_supporting_lateral


# 서포팅 레터럴로 복구가 불가능한 부하를 가져온다.
def getLateralRemainLoad(list_dict, lateral_id):
    # 레터럴 타이 스위치를 가져온다.
    lateral_tie_switch = getDict(list_dict, [['type', 'lateral'], ['id', lateral_id]], [])[1]['lateral_tie_switch']
    # 레터럴 타이 스위치가 같고 레터럴과 id가 다른며 레터럴 타이 스위치가 ''가 아닌 것이 서포팅 레터럴이다.
    dict_supporting_lateral = getDict(list_dict, [['type', 'lateral'], ['lateral_tie_switch', lateral_tie_switch]], [['id', lateral_id], ['lateral_tie_switch', '']])[1]
    # 서포팅 레터럴의 부하를 가져온다.
    load_supporting_lateral = int(dict_supporting_lateral['load'])
    # 서포팅 레터럴로 복구 가능한 부하
    load_res = getLateralResLoad(list_dict, lateral_id)
    # 서포팅 레터럴로 복구가 불가능한 부하
    return load_supporting_lateral - load_res


# 순서도 알고리즘 1
def algorithm_one(list_dict, supporting_feeder_margin, total_group_c_load, total_group_b_load):
    flow_chart_result = 'block 1 yes'
    switch_operation_order = []
    ann_output = [0, 0, 0, 0, 0, 0, 0, 0]
    switch_operation_order.append('SW9 close')

    ann_output = normalizationOutput(ann_output)

    return flow_chart_result, switch_operation_order, ann_output


# 순서도 알고리즘 2
def algorithm_two(list_dict, supporting_feeder_margin, total_group_c_load, total_group_b_load):
    flow_chart_result = 'block 2 yes'
    switch_operation_order = []
    ann_output = [0, 0, 0, 0, 0, 0, 0, 0]

    # 서포팅 피더의 마진에서 그룹 C 의 부하와 그룹 B의 부하를 차감한다.
    capacity_margin = supporting_feeder_margin - total_group_c_load - total_group_b_load

    # 그룹 A 를 오름차순으로 정렬한다.
    list_sort_group_a = setSortGroupList(list_dict, 'A')

    # 서포팅 피더로 전력을 공급 받는 레터럴
    list_lateral = []

    # 서포팅 피더로 전력을 공급 받는 레터럴 리스트 생성
    switch_operation_order.append('SW9 close')
    for idx in range(len(list_sort_group_a)):
        lateral_id = list_sort_group_a[idx]
        # 부하를 가져온다.
        load = int(getDict(list_dict, [['type', 'lateral'], ['id', lateral_id]], [])[1]['load'])
        if capacity_margin - load >= 0:
            capacity_margin -= load
            list_lateral.append(lateral_id)

    # 스위치 조작 순서 및 ANN OUTPUT 생성
    for idx in range(len(list_sort_group_a)):
        lateral_id = list_sort_group_a[idx]
        if list_sort_group_a[idx] not in list_lateral:
            dict_lateral = getDictLateral(list_dict, lateral_id)[1]
            switch_operation_order.append(dict_lateral['lateral_tie_switch'] + ' close')
            switch_operation_order.append('branching point ' + dict_lateral['feeder_branching_point'] + ' - ' + dict_lateral['id'] + ' switch open')
            ann_index = int(lateral_id[3:]) - 1
            ann_output[ann_index] = 1

    ann_output = normalizationOutput(ann_output)

    return flow_chart_result, switch_operation_order, ann_output


# 순서도 알고리즘 3 (아직 완성 안 됨)
def algorithm_three(list_dict, supporting_feeder_margin, total_group_c_load, total_group_b_load, total_group_b_res_laod):
    flow_chart_result = 'block 1 yes'
    switch_operation_order = []
    ann_output = [0, 0, 0, 0, 0, 0, 0, 0]

    # 그룹 B 를 오름차순으로 정렬한다.
    list_sort_group_b = setSortGroupList(list_dict, 'B')

    # 서포팅 피더의 마진에서 그룹 C의 부하를 뺀다.
    capacity_margin = supporting_feeder_margin - total_group_c_load

    # 서포팅 피더로 전력을 공급 받는 레터럴
    lateral_그룹B중_서포팅피더로전력을공급받는레터럴 = []
    lateral_그룹B중_전력을반반공급받는레터럴 = []
    # 논문에는 언급이 없음
    lateral_그룹A중_서포팅피더로전력을공급받는 = []

    # 서포팅 피더로 전력을 공급 받는 레터럴 리스트 생성
    for idx in range(len(list_sort_group_b)):
        lateral_id = list_sort_group_b[idx]
        # 부하를 가져온다.
        load_lateral = getLateralLoad(list_dict, lateral_id)
        # 복구된 부하의합
        load_sum = 0
        # 남은 여유 용량에서 (현재 레터럴의 부하 + 남은 잔여 부하)를 뺐을 때 0과 같거나 클 경우
        if 1 == 1:
            pass
        # 남은 여유 용량에서 (현재 레터럴의 부하 + 남은 잔여 부하)를 뺐을 때 0보다 작은 경우
        else:
            pass

    switch_operation_order.append('SW9 close')

    ann_output = normalizationOutput(ann_output)

    return flow_chart_result, switch_operation_order, ann_output


# 순서도
def flowChart(list_dict):

    idx_feeder, dict_feeder = getDict(list_dict, [['type', 'feeder'], ['id', 'F1']], [])
    idx_supporting_feeder, dict_supporting_feeder = getDict(list_dict, [['type', 'feeder'], ['id', 'F2']], [])
    idx_result, dict_result = getDict(list_dict, [['type', 'result']], [])
    supporting_feeder_margin = int(dict_supporting_feeder['capacity_margin'])
    total_lateral_load = int(dict_feeder['load'])
    total_group_a_load = getTotalGroupLoad(list_dict, 'A')
    total_group_b_load = getTotalGroupLoad(list_dict, 'B')
    total_group_c_load = getTotalGroupLoad(list_dict, 'C')
    total_group_b_res_laod = getTotalGroupBResLoad(list_dict)

    print('총 부하:', total_lateral_load)
    print('그룹 A의 총 부하:', total_group_a_load)
    print('그룹 B의 총 부하:', total_group_b_load)
    print('그룹 C의 총 부하:', total_group_c_load)
    print('서포팅 피더의 여유 용량:', supporting_feeder_margin)
    print('서포팅 피더의 여유 용량 - 그룹 C의 총 부하:', supporting_feeder_margin - total_group_c_load)
    print('서포팅 피더의 여유 용량 - 그룹 C의 총 부하 - 그룹 B의 총 부하 :', supporting_feeder_margin - total_group_c_load - total_group_b_load)
    print('그룹 B 중 서포팅 레터럴로 복구 가능한 브랜칭 포인트의 총 부하:', total_group_b_res_laod)

    flow_chart_result = ''
    switch_operation_order = []
    ann_output = [0, 0, 0, 0, 0, 0, 0, 0] # LAT1 ~ LAT8 까지 순서대로 출력

    # 결과 딕셔너리 : {'type':'result', 'seq':'', 'flow_chart_result':'', 'switch_operation_order':[], 'ann_output':{'LAT1':'', 'LAT2':'', 'LAT3':'', 'LAT4':'', 'LAT5':'', 'LAT6':'', 'LAT7':'', 'LAT8':''}}

    # 분기 1 사전 계산

    # 분기 1
    print('분기 1 시작')
    if supporting_feeder_margin >= total_lateral_load:
        flow_chart_result, switch_operation_order, ann_output = algorithm_one(list_dict, supporting_feeder_margin, total_group_c_load, total_group_b_load)
        dict_result['flow_chart_result'] = flow_chart_result
        dict_result['switch_operation_order'] = switch_operation_order
        dict_result['ann_output'] = ann_output
        idx_result = getDict(list_dict, [['type', 'result']], [])[0]
        list_dict[idx_result] = dict_result
        return list_dict

    print('분기 2 시작')
    # 분기 2
    if supporting_feeder_margin >= total_group_c_load + total_group_b_load:
        flow_chart_result, switch_operation_order, ann_output = algorithm_two(list_dict, supporting_feeder_margin, total_group_c_load, total_group_b_load)
        dict_result['flow_chart_result'] = flow_chart_result
        dict_result['switch_operation_order'] = switch_operation_order
        dict_result['ann_output'] = ann_output
        idx_result = getDict(list_dict, [['type', 'result']], [])[0]
        list_dict[idx_result] = dict_result
        return list_dict


    # 뒤에 알고리즘이 완료되지 않은 것도 ANN에는 뒤에 알고리즘이 필요하지 않는 것도 있어서 아래와 같이 리턴
    ann_output = [9, 9, 9, 9, 9, 9, 9, 9]
    dict_result['flow_chart_result'] = flow_chart_result
    dict_result['switch_operation_order'] = switch_operation_order
    dict_result['ann_output'] = ann_output
    list_dict[idx_result] = dict_result
    return list_dict


    print('블록 3 시작')
    # 블록 3

    print('분기 4 시작')
    # 분기 4
    if supporting_feeder_margin >= total_group_c_load + total_group_b_load:
        flow_chart_result, switch_operation_order, ann_output = algorithm_two(list_dict, supporting_feeder_margin, total_group_c_load, total_group_b_load)
        dict_result['flow_chart_result'] = flow_chart_result
        dict_result['switch_operation_order'] = switch_operation_order
        dict_result['ann_output'] = ann_output
        idx_result = getDict(list_dict, [['type', 'result']], [])[0]
        list_dict[idx_result] = dict_result
        return list_dict

    print('분기 5 시작')
    # 분기 5
    if supporting_feeder_margin >= total_group_c_load + total_group_b_load - total_group_b_res_laod:
        flow_chart_result, switch_operation_order, ann_output = algorithm_three(list_dict, supporting_feeder_margin, total_group_c_load, total_group_b_load, total_group_b_res_laod)
        dict_result['flow_chart_result'] = flow_chart_result
        dict_result['switch_operation_order'] = switch_operation_order
        dict_result['ann_output'] = ann_output
        idx_result = getDict(list_dict, [['type', 'result']], [])[0]
        list_dict[idx_result] = dict_result
        return list_dict

    print('블록 6 시작')
    # 블록 6

    print('분기 7 시작')
    # 분기 7
    if supporting_feeder_margin >= total_group_c_load + total_group_b_load - total_group_b_res_laod:
        flow_chart_result, switch_operation_order, ann_output = algorithm_three(list_dict, supporting_feeder_margin, total_group_c_load, total_group_b_load, total_group_b_res_laod)
        dict_result['flow_chart_result'] = flow_chart_result
        dict_result['switch_operation_order'] = switch_operation_order
        dict_result['ann_output'] = ann_output
        idx_result = getDict(list_dict, [['type', 'result']], [])[0]
        list_dict[idx_result] = dict_result
        return list_dict

    return list_dict


def setData(list_dict):
    input_branching_point = ''
    input_supporting_feeder = ''
    input_supporting_lateral = ''

    list_branching_point_id = ['3','2','18','19','22','27','28','30','34','37','38','9','10','14','43','44','46','60','61','62','49','50','52','56','59','58','20']
    list_supporting_lateral_id = ['LAT13','LAT14','LAT15','LAT16','LAT17','LAT18','LAT19','LAT20']

    input_branching_point += '['
    for item in list_branching_point_id:
        idx, dict = getDictBranchingPoint(list_dict, item)
        input_branching_point += str(dict['load']) + ", "
    input_branching_point = input_branching_point[:-2] + ']'

    input_supporting_feeder = '[' + str(getDict(list_dict, [['type', 'feeder'], ['id', 'F2']], [])[1]['capacity_margin']) + ']'

    input_supporting_lateral += '['
    for item in list_supporting_lateral_id:
        idx, dict = getDictLateral(list_dict, item)
        input_supporting_lateral += str(dict['load']) + ", "
    input_supporting_lateral = input_supporting_lateral[:-2] + ']'


    return input_branching_point, input_supporting_feeder, input_supporting_lateral


# 인풋 데이터 정규화
def normalizationInput(ann_input):
    min = ann_input[0]
    max = ann_input[0]

    for idx in range(len(ann_input)):
        if max < ann_input[idx]:
            max = ann_input[idx]

    for idx in range(len(ann_input)):
        if min > ann_input[idx]:
            min = ann_input[idx]

    for idx in range(len(ann_input)):
        ann_input[idx] = (0.8 * ((ann_input[idx] - min)/(max - min))) + 0.1

    return ann_input


# 아웃풋 데이터 정규화
def normalizationOutput(ann_output):
    for idx in range(len(ann_output)):
        ann_output[idx] = 0.9 if ann_output[idx] == 1 else 0.1
    return ann_output


# ANN INPUT 생성
def setAnnInput(list_dict):

    list_ann_input_lateral = ['LAT1', 'LAT2', 'LAT3', 'LAT4', 'LAT5', 'LAT6', 'LAT7', 'LAT8']

    ann_input = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    idx_supporting_feeder, dict_supporting_feeder = getDict(list_dict, [['type', 'feeder'], ['id', 'F2']], [])
    supporting_feeder_margin = int(dict_supporting_feeder['capacity_margin'])
    total_group_b_load = getTotalGroupLoad(list_dict, 'B')
    total_group_c_load = getTotalGroupLoad(list_dict, 'C')

    dict_group_a = getDict(list_dict, [['type', 'group'], ['id', 'A']], [])[1]['lateral']
    dict_group_b = getDict(list_dict, [['type', 'group'], ['id', 'B']], [])[1]['lateral']

    # ANN INPUT (서포팅 피더 마진 - 그룹 B - 그룹 C)
    # 정규화
    ann_input[8] = supporting_feeder_margin - total_group_c_load - total_group_b_load

    # ANN INPUT LATERAL 설정
    for idx in range(len(list_ann_input_lateral)):
        lateral_id = list_ann_input_lateral[idx]
        load = getLateralLoad(list_dict, lateral_id)
        if lateral_id in dict_group_a:
            ann_input[idx] = load
        if lateral_id in dict_group_b:
            ann_input[idx] = 0

    ann_input = normalizationInput(ann_input)

    return ann_input


#####
# 서포팅 피더의 부하 최소값과 최대값
supporting_feeder_load_min = 100
supporting_feeder_load_max = 250

#####
# 브랜칭 포인트(레터럴1에 있는) 최소값과 최대값
branching_point_lateral_1_load_min = 1
branching_point_lateral_1_load_max = 40

#####
# 브랜칭 포인트(레터럴2,3,4,5,6,7,8에 있는) 최소값과 최대값
branching_point_other_lateral_load_min = 1
branching_point_other_lateral_load_max = 24

#####
# 브랜칭 포인트(서포팅 레터럴에 있는) 최소값과 최대값
branching_point_supporting_lateral_load_min = 10
branching_point_supporting_lateral_load_max = 33


# 실행 전 해야 할 일
# 1. C:\dat 폴더 생성
# 2. C:\dat\input 폴더 생성
def main():

    list_list = []

    # 데이터를 파일에서 읽기 : 1
    # 데이터를 소스에서 생성하기 : 2
    flag = '1'

    if flag == '1':

        path_dir = 'C:/dat/'

        file_list = os.listdir(path_dir)

        for item in file_list:

            file_name = item

            file_path = path_dir + file_name

            print(os.path.dirname(file_path))

            # C:\dat 바로 밑에 있는 파일만 가져오기 위해 다른 폴더 들은 패쓰
            if os.path.dirname(file_path) != 'C:/dat':
                continue

            # 파일이 아니라 디렉터리인 경우 패쓰
            if os.path.isdir(file_path):
                continue

            print(file_path)

            file_extension = file_name.split('.')[1]

            if os.path.exists(file_path) and file_path.split('.')[1] == 'dat' and file_name.find('output') == -1:
                f = open(file_path, 'r')
                # 아웃풋 리스트
                list_output = []
                # 라인의 개수
                data_cnt = 1
                while True:
                    line = f.readline()
                    print()
                    print()
                    print('라인넘버', data_cnt)
                    print(line)
                    if not line:
                        print('종료')
                        break
                    list_dict = []
                    list_dict = initial(list_dict)

                    # 레터럴의 로드 및 여유 용량 설정
                    list_dict = setLateralFromLine(list_dict, line)
                    # 서포팅 피더의 여유 용량 설정
                    list_dict = setSupportingFeederFromLine(list_dict, line)
                    # 서포팅 레터럴의 로드 및 여유 용량 설정
                    list_dict = setSupportingLateralFromLine(list_dict, line)



                    print('중간 딕셔너리 리스트:', list_dict)

                    # 레터럴을 그룹에 분배
                    # 그룹 별 로드 등을 계산하는 것은 순서도가 실행될 때 한다. (알고리즘의 일관성을 위해)
                    list_dict = setGroup(list_dict)

                    # 순서도 실행
                    # 1. 결과 생성
                    # 2. 결과에는 순서도 블록 및 선택 알고리즘이 들어 간다.
                    list_dict = flowChart(list_dict)

                    # list 에 추가
                    # 1. 9번에 도달한 데이터는 list에 추가하지 않는다.
                    list_list.append(list_dict)

                    # result 딕셔너리를 가져와 seq 에 data_cnt 를 집어 넣는다.
                    idx_result, dict_result = getDict(list_dict, [['type', 'result']], [])
                    dict_result['seq'] = data_cnt
                    list_dict[idx_result] = dict_result

                    print('최종 딕셔너리 리스트:', list_dict)
                    print('결과:', getDict(list_dict, [['type', 'result']], [])[1])

                    print('정규화된 Input:', setAnnInput(list_dict))

                    list_output.append(line + str(setAnnInput(list_dict)) + str(getDict(list_dict, [['type', 'result']], [])[1]['ann_output']))

                    # seq에 넣을 data_cnt 에 1 추가
                    data_cnt += 1

                output_file = open(path_dir + file_name + '_output' + '.' + file_extension, 'w')

                for item in list_output:
                    output_file.write("%s\n" % item)

    elif flag == '2':

        list_output = []

        #####
        # 생성할 데이터의 개수
        total_data_cnt = 2000

        data_cnt = 1

        # 생성할 데이터의 개수를 range에 설정
        for idx in range(total_data_cnt):

            list_dict = []

            list_dict = initial(list_dict)



            # 레터럴의 로드 및 여유 용량 설정
            list_dict = setLateral(list_dict)
            # 서포팅 피더의 여유 용량 설정
            list_dict = setSupportingFeeder(list_dict)
            # 서포팅 레터럴의 로드 및 여유 용량 설정
            list_dict = setSupportingLateral(list_dict)



            # 레터럴을 그룹에 분배
            # 그룹 별 로드 등을 계산하는 것은 순서도가 실행될 때 한다. (알고리즘의 일관성을 위해)
            list_dict = setGroup(list_dict)

            # 순서도 실행
            # 1. 결과 생성
            # 2. 결과에는 순서도 블록 및 선택 알고리즘이 들어 간다.
            list_dict = flowChart(list_dict)

            # list 에 추가
            # 1. 9번에 도달한 데이터는 list에 추가하지 않는다.
            list_list.append(list_dict)

            # result 딕셔너리를 가져와 seq 에 data_cnt 를 집어 넣는다.
            idx_result, dict_result = getDict(list_dict, [['type', 'result']], [])
            dict_result['seq'] = data_cnt
            list_dict[idx_result] = dict_result

            print('최종 딕셔너리 리스트:', list_dict)
            print('결과:', getDict(list_dict, [['type', 'result']], [])[1])

            input_branching_point, input_supporting_feeder, input_supporting_lateral = setData(list_dict)
            list_output.append(input_branching_point + input_supporting_feeder + input_supporting_lateral)
            # seq에 넣을 data_cnt 에 1 추가
            data_cnt += 1

        for item in list_output:
            print(item)

        #####
        # 아웃풋 파일 경로
        output_file = open('C:/dat/input/input' + str(total_data_cnt) + '.' + 'dat', 'w')

        for item in list_output:
            output_file.write("%s\n" % item)

if __name__ == '__main__':
    main()




