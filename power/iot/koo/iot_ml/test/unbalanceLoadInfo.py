import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import dbConnection
import MySQLdb
import modelUnbalanceLoad as mul
import datetime

class UnbalanceLoadInfo():
    def __init__(self):
        print('머신시작1-1')
        # DB connection
        self.con = dbConnection.getConnection()
        print('머신시작1-1-1')
        self.con.set_character_set('utf8')
        self.cur = self.con.cursor(MySQLdb.cursors.DictCursor)
        self.cur.execute('SET NAMES utf8;')
        self.cur.execute('SET CHARACTER SET utf8;')
        self.cur.execute('SET character_set_connection=utf8;')
        print('머신시작1-1-5')
        self.path_dir = 'F:\\IOT\\data\\2'
        self.model = mul.ModelUnbalanceLoad()
        print('머신시작1-2')

    # SELECT
    def selecPoleTemp(self, sensor_id, time_start, time_end):
        query = """SELECT TIME_ID, TEMP
                    FROM TB_IOT_POLE_SECOND s1
                    WHERE SENSOR_ID = '%s'
                    AND TIME_ID BETWEEN '%s' AND '%s'
                    """ % (sensor_id, time_start, time_end)
        self.cur.execute(query)
        results = self.cur.fetchall()
        df = pd.DataFrame(list(results))
        return df


    # SELECT
    def selectSensorList(self, pole_id):
        query = """SELECT POLE_ID, SENSOR_ID, PART_NAME
                    FROM TB_IOT_POLE_INFO s1
                    WHERE POLE_ID = '%s'
                    AND IFNULL(POLE_ID, '') != ''
                    AND IFNULL(SENSOR_ID, '') != ''
                    AND IFNULL(PART_NAME, '') != ''
                    """ % (pole_id)
        self.cur.execute(query)
        results = self.cur.fetchall()
        df = pd.DataFrame(list(results))
        return df

    def getMonthlyInfoDB(self, pole_id, time_start, time_end):
        print(time_start)
        print(time_end)
        index_start = datetime.datetime.strptime(time_start[:19], '%Y-%m-%d %H:%M:%S')
        index_end = datetime.datetime.strptime(time_end[:19], '%Y-%m-%d %H:%M:%S')
        time_start_predict = datetime.datetime(index_start.year, index_start.month, index_start.day, 0, 0, 0)
        time_end_predict = time_start_predict + datetime.timedelta(1) - datetime.timedelta(seconds=1)

        unbalanceCount = 0
        unbalanceClass = 0

        cnt = 0
        while time_start_predict < index_end:
            time_start_predict += datetime.timedelta(days=1)
            time_end_predict += datetime.timedelta(days=1)
            unbalanceCount += self.getDailyInfo(pole_id, time_start_predict, time_end_predict)
            # print('unbalanceCount:', unbalanceCount)
            cnt += 1
            # print('cnt:', cnt)

        if unbalanceCount > 10:
            unbalanceClass = 2
        elif unbalanceCount > 5:
            unbalanceClass = 1

        return unbalanceCount, unbalanceClass


    def getDailyInfoDB(self, pole_id, time_start, time_end):

        df = self.getData(pole_id, time_start, time_end)
        arr = self.setInput(df)
        unbalanceCount = 0
        # print('arr:', len(arr))
        if len(arr) == 72:
            # print('예측시작')
            unbalanceCount = self.model.predict(arr)

        return unbalanceCount

    def getTotalInfoCSV(self, df, pole_id, date):
        df = df[df['POLE_ID'] == pole_id]

        index_start = datetime.datetime.strptime('2017-06-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        index_end = datetime.datetime.strptime('2017-12-31 23:59:59', '%Y-%m-%d %H:%M:%S')
        time_start_predict = datetime.datetime(index_start.year, index_start.month, index_start.day, 0, 0, 0)
        time_end_predict = time_start_predict + datetime.timedelta(1) - datetime.timedelta(seconds=1)

        unbalanceCount = 0
        unbalanceClass = 0

        cnt = 0
        while time_start_predict < index_end:
            unbalanceCountDaily, unbalanceClassDaily = self.getDailyInfoCSV(df, pole_id, str(time_start_predict)[0:10])
            time_start_predict += datetime.timedelta(days=1)
            time_end_predict += datetime.timedelta(days=1)
            # print('unbalanceCount:', unbalanceCount)
            unbalanceCount += unbalanceCountDaily
            cnt += 1

        if unbalanceCount > 70:
            unbalanceClass = 2
        elif unbalanceCount > 50:
            unbalanceClass = 1

        print('pole_id:', pole_id, 'unbalanceCount:', unbalanceCount, 'unbalanceClass:', unbalanceClass)

        return unbalanceCount, unbalanceClass

    def getMonthlyInfoCSV(self, df, pole_id, date):
        df = df[df['POLE_ID'] == pole_id]

        index_start = datetime.datetime.strptime('2017-07-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        index_end = datetime.datetime.strptime('2017-07-31 23:59:59', '%Y-%m-%d %H:%M:%S')
        time_start_predict = datetime.datetime(index_start.year, index_start.month, index_start.day, 0, 0, 0)
        time_end_predict = time_start_predict + datetime.timedelta(1) - datetime.timedelta(seconds=1)

        unbalanceCount = 0
        unbalanceClass = 0

        cnt = 0
        while time_start_predict < index_end:
            unbalanceCountDaily, unbalanceClassDaily = self.getDailyInfoCSV(df, pole_id, str(time_start_predict)[0:10])
            time_start_predict += datetime.timedelta(days=1)
            time_end_predict += datetime.timedelta(days=1)
            # print('unbalanceCount:', unbalanceCount)
            unbalanceCount += unbalanceCountDaily
            cnt += 1

        if unbalanceCount > 10:
            unbalanceClass = 2
        elif unbalanceCount > 5:
            unbalanceClass = 1

        print('pole_id:', pole_id, 'unbalanceCount:', unbalanceCount, 'unbalanceClass:', unbalanceClass)

        return unbalanceCount, unbalanceClass

    def getDailyInfoCSV(self, df, pole_id, date):
        df = df[df['POLE_ID'] == pole_id]
        df = df[df['DATE'] == date]
        return df['UNBALANCE_FLAG'].values[0], 1

    def setInput(self, df):
        arr = np.array([])
        for item in df.columns.values:
            arr = np.append(arr, df[item].values)
        for idx in range(3 - len(df.columns.values)):
            arr = np.append(arr, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        return arr


    def getData(self, pole_id, time_start, time_end):
        start = time.time()

        df_result = pd.DataFrame(columns=['TIME_ID'])
        df_result.set_index('TIME_ID', inplace=True)

        df_sensor = self.selectSensorList(pole_id)
        for idx in range(len(df_sensor)):
            sensor_id = str(df_sensor['SENSOR_ID'][idx])
            part_name = str(df_sensor['PART_NAME'][idx])
            if part_name == 'nan' or part_name != '변압기 본체':
                continue
            df_temp = self.selecPoleTemp(sensor_id, time_start, time_end)
            if len(df_temp) == 0:
                continue
            df_temp = df_temp[['TIME_ID', 'TEMP']]
            df_temp['TEMP'] = df_temp['TEMP'].astype(float)
            df_temp.columns = ['TIME_ID', sensor_id]
            df_temp.set_index('TIME_ID', inplace=True)
            df_temp.index = pd.to_datetime(df_temp.index)

            df_temp = df_temp.resample('60T').mean()
            df_result = pd.merge(df_result, df_temp, left_index=True, right_index=True, how='outer')

        # mask = (df['TIME_ID'] >=  '') & (df['TIME_ID'] <= '')
        # df = df.loc[mask],ㅣㅣㅣㅣㅣ

        print(time.time() - start)

        return df_result


    def getListPole(self):
        return  [
            '8132X291',
            '8132W782',
            '8232P471',
            '8232R383',
            '8232R152',
            '8132W212',
            '8132W832',
            '8232P531',
            '8132W952',
            '8132Z961',
            '8132X914',
            '8132Q911',
            '8132W231',
            '8132W122',
            '8132X921',
            '8132X152',
            '8132X122',
            '8132W621',
            '8132W981',
            '8132X601'
        ]


    def writePredictCsv(self):

        time_start = '2017-06-01 00:00:00'
        time_end = '2017-12-31 23:59:59'

        list_pole = self.getListPole()

        obj = UnbalanceLoadInfo()

        df_result = pd.DataFrame(columns=['DATE', 'POLE_ID', 'UNBALANCE_FLAG'])

        for pole_id in list_pole:
            index_start = datetime.datetime.strptime(time_start[:19], '%Y-%m-%d %H:%M:%S')
            index_end = datetime.datetime.strptime(time_end[:19], '%Y-%m-%d %H:%M:%S')
            time_start_predict = datetime.datetime(index_start.year, index_start.month, index_start.day, 0, 0, 0)
            time_end_predict = time_start_predict + datetime.timedelta(1) - datetime.timedelta(seconds=1)

            df = obj.getData(pole_id, time_start, time_end)

            cnt = 0
            while time_start_predict < index_end:

                df_data = df.loc[time_start_predict:time_end_predict,:]

                arr = obj.setInput(df_data)
                unbalanceFlag = 0
                # print('arr:', len(arr))
                if len(arr) == 72:
                    # print('예측시작')
                    unbalanceFlag = obj.model.predict(arr)
                print(time_start_predict)
                df_result = df_result.append({'DATE': time_start_predict, 'POLE_ID':pole_id, 'UNBALANCE_FLAG':unbalanceFlag}, ignore_index=True)

                time_start_predict += datetime.timedelta(days=1)
                time_end_predict += datetime.timedelta(days=1)

                print('unbalanceFlag:', unbalanceFlag)
                cnt += 1
                # print('cnt:', cnt)

        # print(df_result)
        df_result.set_index('DATE', inplace=True)
        df_result.to_csv('predict_all.csv')


if __name__ == '__main__':

    obj = UnbalanceLoadInfo()
    # obj.writePredictCsv()

    date = '2017-07-01'

    file_name = './predict_all.csv'

    list_pole = obj.getListPole()

    df = pd.read_csv(file_name)

    for pole_id in list_pole:
        #unbalanceCount, unbalanceClass = obj.getTotalInfoCSV(df, pole_id, date)
        unbalanceCount, unbalanceClass = obj.getMonthlyInfoCSV(df, pole_id, date)
        # unbalanceCount, unbalanceClass = obj.getDailyInfoCSV(df, pole_id, date)






