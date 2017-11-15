import pandas as pd

# value 를 가져온다.
def getDictValue(dict, key):
    return dict[key] if key in dict else ''

# 확인요망
# DB 에서 구간에 속한 지점을 가져온다.
def getSector(cur, dict):
    list_col = ['STEP_IN_1', 'STEP_IN_2', 'STEP_IN_3', 'STEP_IN_4', 'STEP_IN_5', 'STEP_OUT_1']
    list_return = []
    query = """
                SELECT STEP_IN_1, STEP_IN_2, STEP_IN_3, STEP_IN_4, STEP_IN_5, STEP_OUT_1  
                FROM SEC_TB
                WHERE 1 = 1
                AND PIPE_BLK = '%s'
            """ % (getDictValue(dict, 'sector'))
    cur.execute(query)
    tuple = cur.fetchall()
    if cur.rowcount == 1:
        for item in list_col:
            if tuple[0][item] != '' and tuple[0][item] is not None:
                list_return.append(tuple[0][item])
    return list_return

# DB에서 지점의 데이터를 가져온다.
def getLocationData(cur, dict, tag_name):
    query = """
                SELECT DATE_FORMAT(LOG_TIME, '%%Y-%%m-%%d %%H:%%i:%%s') AS LOG_TIME, TAG_VAL  
                FROM %s
                WHERE 1 = 1
                AND TAG_NAME = '%s'
                AND LOG_TIME BETWEEN DATE_FORMAT('%s', '%%y%%m%%d%%H%%i%%s') AND DATE_FORMAT('%s', '%%y%%m%%d%%H%%i%%s')                
            """ % (getDictValue(dict, 'table'), tag_name, getDictValue(dict, 'time_start'), getDictValue(dict, 'time_end'))
    cur.execute(query)
    tuple = cur.fetchall()
    return pd.DataFrame(list(tuple))