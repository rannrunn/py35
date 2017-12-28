import pandas as pd
import os

def uvc_null(pole_id):
    os.chdir('D:\\dev\\IoT_data\\IoT_2차기간')
    data=pd.read_csv(pole_id+'.csv')
    uvc=int((pd.isnull(data['uvc']).sum() / data.shape[0]) * 100)
    return uvc

if __name__ == '__main__':
    os.chdir("D:\\dev\\IoT_data")
    poleId = pd.read_csv('2차기간_pole_id.csv')
    dict = {}
    list_uvc=[]
    list_pole=poleId['pole_id']
    for pole_id in poleId['pole_id']:
        print(pole_id)
        uvc=uvc_null(pole_id)
        list_uvc.append(uvc)
        # dict[pole_id] = uvc
    dict['pole_id']=list_pole
    dict['uvc_null']=list_uvc
    df = pd.DataFrame(dict)
    df.to_csv('test1.csv', encoding='utf-8')
