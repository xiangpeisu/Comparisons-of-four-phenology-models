import pandas as pd
import copy
import datetime
import gzip
import os
from datetime import *

import joblib
#import eli5
# import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
# from sktime.regression.interval_based import TimeSeriesForestRegressor
from sktime.regression.kernel_based import RocketRegressor
from sklearn.metrics import r2_score
from sklearn import metrics
import datetime
# from sktime.regression.deep_learning import TapNetRegressor

station_path = r'D:\E\data\2001-2022气象数据'
file_path = r'D:\E\data\2001-2022气象数据\2001'
tif_path = 'D:\\E\\data\\grass_yield\\MCD12Q2\\MCD12Q2处理\\'
MCD_path = 'D:\\E\\data\\grass_yield\\MCD12Q2\\MCD12Q2v6_1\\'
article_path = r'E:\saved_model2'

def StatDa2Rast():
    """
    从isd-history2024.csv文件中筛选出经纬度在97~128，35~54、时间跨度在2001~2022年的站点编号
    :return: 站点编号stationID
    """
    # loaddata
    data = pd.read_csv(station_path + '\\' + 'isd-history2024.csv')
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)    
    # print(data.head(5))
    
    Max_Lon, Min_Lon, Max_Lat, Min_Lat = 128,97,54,37
    Max_year, Min_year = 2022, 2001

    # 根据经纬度筛选出所需数据
    neilonbool1 = data['LON'] > Min_Lon
    neilonbool2 = data['LON'] < Max_Lon
    neilonbool = neilonbool1 & neilonbool2

    neilatbool1 = data['LAT'] > Min_Lat
    neilatbool2 = data['LAT'] < Max_Lat
    neilatbool = neilatbool1 & neilatbool2
    neilatlonbool = neilonbool & neilatbool

    # 根据起始、结束时间筛选出所需数据
    # 对data['BEGIN']整体使用lambda函数
    begin = data['BEGIN'].apply(lambda x: str(x)[0:4]) #data['BEGIN']值形如20020220，str(x)[0:4]取其前4个代表年份的字符
    # print(begin)
    end = data['END'].apply(lambda x: str(x)[0:4])
    beginbool = begin.apply(lambda x: int(x)) <= Min_year
    endbool = end.apply(lambda x: int(x)) >= Max_year
    datebool = beginbool & endbool   

    bool = neilatlonbool & datebool
    data['STATION_ID'] = data['USAF']
    stationID = (data['STATION_ID'][bool]).apply(lambda x: str(x)+'99999')
    print(stationID.head(5))
    data['LONGITUDE'] = data['LON']
    stationLon = data['LONGITUDE'][bool]
    # print(stationLon.head(5))
    data['LATITUDE'] = data['LAT']
    stationLat = data['LATITUDE'][bool]
    # 海拔高度
    data['ELEVATION'] = data['ELEV(M)']
    stationElev =  data['ELEVATION'][bool]
    # 站点名
    data['STATION NAME'] = data['STATION NAME']
    stationName = data['STATION NAME'][bool]

    stationInfo = pd.concat([stationID, stationName, stationLon, stationLat, stationElev], axis=1, ignore_index=False)
    # print(stationInfo.head(5))
    # print(len(stationID), len(stationInfo)) #244个无重复站点
    # print(type(stationID.values[0]),type(stationInfo))
    stationInfo.drop_duplicates(inplace=True, ignore_index=True)
    # print(len(stationInfo))

    # 站点54577099999数据缺失严重
    mask = (stationID == 54577099999)
    # print(mask)
    stationID.drop(stationID[mask].index, inplace=True)
    stationInfo.drop(stationInfo[stationInfo['STATION_ID'] == 54577099999].index, inplace=True)
    stationInfo.reset_index(drop=True, inplace=True)
    stationID.reset_index(drop=True, inplace=True)
    # print(len(stationID),len(stationInfo))
    stationInfo.to_csv(station_path + '\\' + 'neimenggu_stations3.csv')
    return stationID, stationInfo

def un_gz(file_name):
    """ungz zip file"""
    f_name = file_name.replace(".gz", "")
    # 获取文件的名称，去掉
    g_file = gzip.GzipFile(file_name)
    # 创建gzip对象
    open(f_name, "wb+").write(g_file.read())
    # gzip对象用read()打开后，写入open()建立的文件里。
    g_file.close()  # 关闭gzip对象

def dataproce(f_name, f_path, stationInfo):
    """
    :param f_name: 气象数据文件路径+名
    :return: meter_arr,numpy_array格式，该文件中的气象数据
             meter_data3,DataFrame格式
    """
    # print(f_name)
    Meter = pd.read_csv(f_path + '\\' + f_name)
    # print(Meter)
    if len(Meter) < 300:
        meter_data3 = pd.DataFrame()
        return meter_data3
    Meter_list = []
    # 完整显示列
    pd.set_option('display.max_columns', None)
    # print(Meter.columns)
    # 用map函数对列表每个元素作相同操作
    # split()默认对连续的空格、换行符、制表符数为分隔符
    line_temp = list(map(lambda x: x.split(), Meter.columns))
    # print(line_temp[0], line_temp)
    # 用pd.DataFrame(line_temp[0])将列表line_temp[0]转成dataframe，
    # .T是使列表中每个元素为1列
    meter_data1 = pd.DataFrame(line_temp[0]).T
    # print(meter_data1.shape)
    meter_data1.columns = ['year', 'month', 'day', 'hour', 'temp', 'dew-pointtemperature', 'pressure', 'winddirection',
                           'windspeed', 'cloudamount',
                           'hour1ofrainfall', 'hour6ofrainfall']
    # print(meter_data1)

    # print(Meter.values,type(Meter.values[0][0]))
    line_temp = list(map(lambda x: x[0].split(), Meter.values))
    # print(line_temp)
    meter_data2 = pd.DataFrame(line_temp)
    # print(meter_data2, type(meter_data2))
    meter_data2.columns = ['year', 'month', 'day', 'hour', 'temp', 'dew-pointtemperature', 'pressure', 'winddirection',
                           'windspeed', 'cloudamount',
                           'hour1ofrainfall', 'hour6ofrainfall']
    # print(meter_data2)

    # 将meter_data1和meter_data2合并
    data = [meter_data1, meter_data2]
    meter_data = pd.concat(data)
    # 不保留旧索引（drop=True），不生成新dataframe（inplace=True）
    meter_data.reset_index(drop=True, inplace=True)
    # print(meter_data)
    # 将填充值-9999替换为nan
    meter_data = meter_data.replace(to_replace='-9999', value=np.nan)
    # print(meter_data.isnull().sum())

    meter_data[['temp', 'dew-pointtemperature', 'pressure', 'winddirection', 'windspeed', 'cloudamount',
                'hour1ofrainfall', 'hour6ofrainfall']] = meter_data[
        ['temp', 'dew-pointtemperature', 'pressure', 'winddirection', 'windspeed', 'cloudamount',
         'hour1ofrainfall', 'hour6ofrainfall']].apply(pd.to_numeric)
    # 对数据进行清洗、插补等预处理
    # 用每日均值代替nan,若还有nan:线性插值：
    # print(meter_data['temp'], meter_data['pressure'], meter_data['cloudamount'], meter_data['hour1ofrainfall'])
    #判断meter_data['year']是否数值型，输出结果为True
    # print(np.isreal(meter_data['year'].values))
    #输出每日温度均值
    # print(meter_data.groupby(['year', 'month', 'day'])['temp'].transform('mean'))

    for col in ['temp', 'dew-pointtemperature', 'pressure', 'windspeed', 'cloudamount']:
        #用每日均值替代缺失值nan
        meter_data[col] = meter_data[col].fillna(meter_data.groupby(['year', 'month', 'day'])[col].transform('mean'))        
        if meter_data[col].isna().sum() != 0:
            #若某天全为nan，则用线性插值替代
            meter_data[col] = meter_data[col].interpolate()
    #查询是否无nan
    nan_values = meter_data.isna().any()  # 测试meter_data中是否有nan
    # print(nan_values)
    ## 统计meter_data每行null值个数
    # print(meter_data.isnull().sum())
    # print( meter_data['temp'], meter_data['pressure'], meter_data['cloudamount'], meter_data['hour1ofrainfall'])
    # 若缺整月数据，或某月数据不足30天，则舍弃该数据，返回处理下个数据文件
    # print(meter_data['month'].nunique())
    if meter_data['month'].nunique()<12:
        meter_data3 = pd.DataFrame()
        return meter_data3

    # 向dataframe中加入站点id、经纬度、海拔信息
    # stationID, stationInfo = StatDa2Rast()
    stationid = f_name.replace('-', '')
    stationid = stationid[0:-4]
    stationid = int(stationid)
    # print(stationid)
    # 找出stationid所在的行号rou_index
    # print(stationInfo[stationInfo.STATION_ID == stationid].index.tolist()[0])
    row_index = stationInfo[stationInfo.STATION_ID == stationid].index.tolist()[0]
    # print(row_index,type(row_index))
    lon = stationInfo['LONGITUDE'][row_index]
    lat = stationInfo['LATITUDE'][row_index]
    elev = stationInfo['ELEVATION'][row_index]
    # print(stationInfo,stationInfo['LONGITUDE'][row_index])
    # print(stationid, lon, lat, elev)
    meter_data['STATION_ID'] = stationid
    meter_data['LONGITUDE'] = lon
    meter_data['LATITUDE'] = lat
    meter_data['ELEVATION'] = elev

    # 统计每日最高温度、最低温度、平均温度、每日气压、每日降水量、每日风速
    pd.set_option('display.unicode.east_asian_width', True)  # 设置列名对齐
    # 分别将温度、气压、降水量、风速提取出来形成新的数据方便分析
    Temp_data = meter_data[
        ['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month', 'day', 'hour', 'temp']].copy()
    VaporPressure = meter_data[
        ['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month', 'day', 'hour', 'pressure']].copy()
    precipitation = meter_data[
        ['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month', 'day', 'hour6ofrainfall']].copy()
    windspeed = meter_data[
        ['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month', 'day', 'windspeed']].copy()
    cloudamount = meter_data[
        ['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month', 'day', 'cloudamount']].copy()
    dew_pointtemperature = meter_data[
        ['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month', 'day', 'dew-pointtemperature']].copy()

    # #将dataframe中的值由str转成float
    # #禁止SettingWithCopyWarning警告：用Temp_data = meter_data[['年','月','日','温度']]后，
    # # Temp_data实际是一个指向meter_data对应列的“标签”，在内存中并没有为Temp_data分配新的地址用于存储希望截取的数据。
    # #当对Temp_data的“温度”列操作时，python在风中凌乱啊“Temp_data他自己没有数据的啊，他指向了meter_data而已，
    # # 我怎么敢直接修改meter_data的数据呢？！”，所以用meter_data给Temp_data赋值时，加一个.copy()
    # Temp_data.loc[:,'temp'] = Temp_data.loc[:,'temp'].astype(float)
    # VaporPressure.loc[:,'pressure'] = VaporPressure.loc[:,'pressure'].astype(float)
    # precipitation.loc[:,'6hour of rainfall'] = precipitation.loc[:,'6hour of rainfall'].astype(float)
    # windspeed.loc[:,'wind speed'] = windspeed.loc[:,'wind speed'].astype(float)

    # 用groupby对数据中的列分类统计
    # 每日最高温度、最低温度、平均温度(对小时列也求了max,min,mean)
    # 数据说明文档中表示原始数据中温度、露点温度、气压、风速、降雨量的换算系数为10，所以要对原始数据中的对应数据除以10，进行换算。
    Temp_max = (Temp_data.groupby(['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month', 'day'],
                                  as_index=False).max())
    # print(Temp_max)
    Temp_max.rename(columns={'temp': 'Temp_max'}, inplace=True)
    Temp_max['Temp_max'] = Temp_max['Temp_max'] / 10
    # print(Temp_max)
    Temp_min = (Temp_data.groupby(['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month', 'day'],
                                  as_index=False).min())
    # print(Temp_min.head())
    Temp_min.rename(columns={'temp': 'Temp_min'}, inplace=True)
    Temp_min['Temp_min'] = Temp_min['Temp_min'] / 10
    # print(Temp_min.head())
    Temp_mean = (Temp_data.groupby(['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month', 'day'],
                                   as_index=False).mean(numeric_only=True))
    # print(Temp_mean.head())
    Temp_mean.rename(columns={'temp': 'Temp_mean'}, inplace=True)
    Temp_mean['Temp_mean'] = Temp_mean['Temp_mean'] / 10
    # print(Temp_mean.head())

    VaporP_mean = (VaporPressure.groupby(['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month', 'day'],
                                         as_index=False).mean(numeric_only=True))
    # print(VaporP_mean.head())
    VaporP_mean.rename(columns={'pressure': 'VaporP_mean'}, inplace=True)
    VaporP_mean['VaporP_mean'] = VaporP_mean['VaporP_mean'] / 10
    # print(VaporP_mean.head())
    precipitation_sum = (
        precipitation.groupby(['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month', 'day'],
                              as_index=False).sum())
    # print(precipitation_sum.head())
    precipitation_sum.rename(columns={'hour6ofrainfall': 'precipitation_sum'}, inplace=True)
    precipitation_sum['precipitation_sum'] = precipitation_sum['precipitation_sum'] / 10
    # print(precipitation_sum.head(),precipitation_sum.isnull().sum())
    windspeed_mean = (windspeed.groupby(['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month', 'day'],
                                        as_index=False).mean())
    # print(windspeed_mean.head())
    windspeed_mean.rename(columns={'windspeed': 'windspeed_mean'}, inplace=True)
    windspeed_mean['windspeed_mean'] = windspeed_mean['windspeed_mean'] / 10
    # print(windspeed_mean.head())
    cloudamount = (cloudamount.groupby(['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month', 'day'],
                                       as_index=False).mean())
    # print(cloudamount.head())
    cloudamount['cloudamount'] = cloudamount['cloudamount'] / 10

    dew_pointtemperature = (dew_pointtemperature.groupby(['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month', 'day'],
                                       as_index=False).mean())
    dew_pointtemperature['dew-pointtemperature'] = dew_pointtemperature['dew-pointtemperature']/10

    # 将得到的气象数据整合到一个dataframe中返回
    meter_data3 = pd.concat([Temp_max, Temp_min['Temp_min'], Temp_mean['Temp_mean'], VaporP_mean['VaporP_mean'],
                             precipitation_sum['precipitation_sum'], windspeed_mean['windspeed_mean'],
                             cloudamount['cloudamount'], dew_pointtemperature['dew-pointtemperature']], axis=1)
    # print(meter_data3.head(),meter_data3.isnull().sum())
    # print(meter_data3)

    # 若当年数据个数!=365，补齐到365
    # 1.其余只相差几天的，线性插补
    # 2.若为闰年:2月28日的数据用28日和29日的均值替换; 删去2月29日
    # print(f_name)
    year = int(f_name[-4:])
    # print(year)
    # 1.其余只相差几天的，线性插补
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] #每月包含的天数
    if len(meter_data3) != 365:
        # 若这年不足365天，先线性插补，再查还有哪月少
        for col in ['Temp_max', 'Temp_min', 'Temp_mean', 'VaporP_mean', 'windspeed_mean',
                    'cloudamount', 'dew-pointtemperature']:
            meter_data3[col] = meter_data3[col].astype(float).interpolate()
        # print(f_name,year,len(meter_data3))
        if len(meter_data3) != 365:
            for month in range(1, 13):
            # for month in range(4,5):
                mask_month = (meter_data3['month'].apply(pd.to_numeric) == month)
                # print(meter_data3[mask_month])
                if len(meter_data3[mask_month]) < days[month - 1]:
                    # print(month,len(meter_data3[mask_month]),days[month-1])
                    # 若某月天数不足，查缺失哪天，补齐
                    # print(meter_data3[mask_month]['day'].apply(pd.to_numeric))
                    for date in range(1, days[month - 1] + 1):
                    # # for date in range(28, days[month - 1] + 1):
                        # 若缺失date,则算出date前一日的index，在index+1处插入新行
                        if date not in list(meter_data3[mask_month]['day'].apply(pd.to_numeric)):
                            #若缺失date不是1号，则需计算date前一日的index
                            if date != 1:
                                mask_date = mask_month & (meter_data3['day'].apply(pd.to_numeric) == date-1)
                                index = meter_data3[mask_date].index[0]
                                index = index - 1
                            # 若缺失date是1号，但不是1月1号，则需计算前一月最后一日index
                            elif (date == 1) and (month > 1):
                                mask_date = (meter_data3['month'].apply(pd.to_numeric) == month-1) & \
                                            (meter_data3['day'].apply(pd.to_numeric) == days[month-2])
                                index = meter_data3[mask_date].index[0]
                                index = index + 1
                            #若缺失date是1月1日，则index = 0
                            if (date == 1) and (month == 1):
                                index = 0
                            # print(month,date)
                            medavalist = copy.deepcopy(list(meter_data3.iloc[index]))
                            # print(medavalist)
                            medavalist[-11] = '%02d' % month
                            medavalist[-10] = '%02d' % date
                            medavalist[-8:] = [np.nan] * 8
                            # print(medavalist)
                            # print(month, date)
                            # print(index)
                            columns = meter_data3.columns
                            # print(meter_data3.values,type(meter_data3.values))
                            meter_data3 = pd.DataFrame(np.insert(meter_data3.values, index, values=medavalist, axis=0))
                            meter_data3.columns = columns
                            meter_data3.reset_index(drop=True, inplace=True)
                            mask_month = (meter_data3['month'].apply(pd.to_numeric) == month)
        # 2.若为闰年:2月28日的数据用28日和29日的均值替换; 删去2月29日
        if year % 4 == 0:
            mask0 = (meter_data3['month'] == '02')
            meter_data0 = meter_data3.loc[mask0, 'day']
            if '29' in meter_data0:
                for col in ['Temp_max', 'Temp_min', 'Temp_mean', 'VaporP_mean', 'precipitation_sum',
                            'dew-pointtemperature']:
                    mask1 = (meter_data3['month'] == '02') & (meter_data3['day'] == '28')
                    meter_data1 = meter_data3.loc[mask1, col].values[0]
                    # print(meter_data1)
                    mask2 = (meter_data3['month'] == '02') & (meter_data3['day'] == '29')
                    meter_data2 = meter_data3.loc[mask2, col].values[0]
                    dataval = (meter_data1 + meter_data2) / 2
                    # 2月28日的数据用28日和29日的均值替换
                    meter_data3.loc[mask1, col] = dataval
                    # print(dataval)
                    # print(meter_data3.loc[mask1,col].values[0])
                # 删去2月29日：
                meter_data3.drop(meter_data3[mask2].index, inplace=True)
                # print(len(meter_data3))
        for col in ['Temp_max', 'Temp_min', 'Temp_mean', 'VaporP_mean',
                                        'windspeed_mean',
                                        'cloudamount', 'dew-pointtemperature']:
            meter_data3[col] = meter_data3[col].astype(float).interpolate()
        # print(meter_data3[col].isnull().sum())
    print(meter_data3)
    return meter_data3
def ExtrMCD12Q2():
    """
    input:
    output：  将各站点的站点编号、经纬度、海拔、返青期_年份、枯黄期_年份做成dataframe格式
    """
    phenology = ['Greenup', 'Dormancy']
    # stationID, stationInfo = StatDa2Rast()
    # “neimenggu_stations2.csv”是在arcGIS中将内蒙古shp边界范围内的点选出来，共244个点
    stationInfo = pd.read_csv(station_path + '\\' + 'neimenggu_stations2.csv')
    # print(stationID,stationInfo)
    # print(len(stationInfo))
    stationInfo.reset_index(drop=True, inplace=True)
    # stationID.reset_index(drop=True, inplace=True)
    pd.set_option('display.max_columns', None)
    # print(stationID,stationInfo)
    for f in phenology:
        for year in range(2001, 2023): #MCD12Q2提取了2001-2022的tiffile数据
            tifile = 'MCD12Q2.A'+ str(year) + '001' + '.' + f + '.Num_Modes_01.tif'
            filename = MCD_path + '\\' + tifile
            # print(filename)
            dataset = gdal.Open(filename)  # 打开文件
            tiff_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
            originX = tiff_geotrans[0]
            originY = tiff_geotrans[3]
            pixelWidth = tiff_geotrans[1]  #水平分辨率，0.005°
            pixelHeight = tiff_geotrans[5]
            band = dataset.GetRasterBand(1)
            # print(originX,originY,pixelWidth,pixelHeight)
            greenup, dormancy = [], []
            for i in range(0, len(stationInfo)):   # 根据len(stationInfo),内蒙古境内共244个站点
                # print(stationInfo.iloc[i, 3])
                lon = stationInfo.iloc[i, 2]
                lat = stationInfo.iloc[i, 3]
                # print(lon,lat)
                # 计算某一坐标对应像素与左上角的像素的相对位置，按像素数计算，计算公式如下：
                xOffset = int((lon - originX) / pixelWidth)
                yOffset = int((lat - originY) / pixelHeight)
                # 读取（lon，lat）点的返青期值
                data = band.ReadAsArray(xOffset, yOffset, 1, 1)  #只读了1个站点位置的点
                # print(len(data))
                # print(data)
                if f == 'Greenup':
                    greenup.append(data[0, 0])
                else:
                    dormancy.append(data[0, 0])
            # print(len(greenup))
            # print(greenup)
            column = f + '_' + str(year)
            if f == 'Greenup':
                stationInfo[column] = greenup
            else:
                stationInfo[column] = dormancy
            # print(stationInfo,column)
    # print(stationInfo)
    stationInfo.to_csv(MCD_path + "\\StationInfo1.csv", index=False)
    return stationInfo
def ExtrMeteoData():
    """
     将STATION_ID  LONGITUDE   LATITUDE  ELEVATION 年 月 日 日最高温...    日风速 Greenup_2001 Dormancy_2001这种dataframe
     数据合并成dataframe数据dataset中
    :return:dataset存入MCD_path+"\\StationMetePheno.csv"中
    """
    stationInfo = pd.read_csv(MCD_path + "\\StationInfo1.csv")
    # stationInfo = ExtrMCD12Q2()
    # print(stationInfo)
    stationID = stationInfo['STATION_ID']
    # print(len(stationInfo))
    meter_data = pd.DataFrame()
    times = 0
    dataset = pd.DataFrame()
    # 创建列表，存放标签，最后转成np.array
    y_Greenup, y_Dormancy = [], []
    for root, dirs, files in os.walk(station_path, topdown=True):
        # 在station_path文件夹下的子文件夹名放入dirs中
        if root == station_path:
            dir = dirs[0:-3]   # dir = ['2001', '2002', ...., '2022']
        # print(dir)
        else:
            roots = root.replace(station_path + '\\', "") #roots为子文件夹名
            # print(roots)
            # 保证只取“2001”~“2023”这几个子文件夹的文件
            if roots in dir:
                # print(roots,type(roots),'\n')
                # 所需站点（stationID）的气象数据文件名为“statfilename”
                stationname = stationID.apply(lambda x: str(x))
                statfilename = stationname.apply(lambda x: x[:6] + '-' + x[6:] + '-' + roots + '.gz')
                # print(statfilename)

                # 遍历每年每个所需站点的气象数据
                for metefile in statfilename:
                    # 若该站点该年度有数据
                    if metefile in files:
                        #     print((metefile),root,type(root))
                        # 解压该数据文件
                        un_gz(root + '\\'+metefile)

                        # 提取气象数据
                        file_name = metefile.replace(".gz", "")
                        if file_name == '470140-99999-2007'\
                                or file_name == '306370-99999-2001'\
                                or file_name == '299980-99999-2020':
                            continue
                        # print(file_name,metefile)
                        fname = file_name.replace('-', '')
                        fname = fname[:-4]
                        # print(type(fname),type(stationInfo['STATION_ID']))
                        # f_name = root + '\\' + file_name
                        print(file_name)
                        meter_data = dataproce(file_name, root, stationInfo)
                        if meter_data.empty:
                            continue
                        # print(meter_data)
                        # print(type(stationInfo))
                        # print(stationInfo.loc[stationInfo['STATION_ID'] == int(fname)]['Greenup_2001'])
                        # print(file_name)
                        for f in ['Greenup', 'Dormancy']:
                            index = stationInfo.loc[stationInfo['STATION_ID'] == int(fname)].index.tolist()[0] #fname是某站点某年的数据文件名
                            # print(index)
                            # 往meter_data里加Greenup和Dormancy
                            # meter_data[f] = stationInfo.at[index, f + '_' + roots]

                            # # 若 Greenup或Dormancy的值不是32767（无效值），则将此值添加进y_Greenup和y_Dormancy
                            # if ((f == 'Greenup') and (stationInfo.at[index, f + '_' + roots] != 32767)):
                            #     # print(stationInfo.at[index, f + '_' + roots])
                            #     y_Greenup.append(stationInfo.at[index, f + '_' + roots])
                            # elif ((f == 'Dormancy') and (stationInfo.at[index, f + '_' + roots] != 32767)):
                            #     y_Dormancy.append(stationInfo.at[index, f + '_' + roots])
                            if f == 'Greenup':
                                # print(stationInfo.at[index, f + '_' + roots])
                                ygreenup = stationInfo.at[index, f + '_' + roots]
                                print(meter_data)
                                # if metefile=='306270-99999-2001.gz':
                                #     print(meter_data)
                                print(file_name,ygreenup,type(ygreenup))
                                meter_data['Greenup'] = ygreenup

                                y_Greenup.append(stationInfo.at[index, f + '_' + roots])
                            elif f == 'Dormancy':
                                ydormancy = stationInfo.at[index, f + '_' + roots]
                                meter_data['Dormancy'] = ydormancy
                                y_Dormancy.append(stationInfo.at[index, f + '_' + roots])   #某站点某年的Greenup
                        dataset = pd.concat([dataset, meter_data], ignore_index=True) #气象数据
                        # print(dataset,'\n',fname)


                        # print(dataset)
                        # dataset1 = pd.DataFrame(dataset)
                        # print(dataset1)
            #             times = times+1
            #             if times == 3:
            #                 break
            # break

    dataset.to_csv(MCD_path + "\\StationMetePheno1.csv", index=False)
    return y_Greenup, y_Dormancy

def DataPreproc1():
    """
    为sklearn的randomforest和时序准备好数据：1。除去Greenup和Dormancy的值为32767所在年份数据；2.将Greenup和Dormancy的值由距离1970-01-01转成DOY；
                                        3.将meter_data中年月日的值也转成当年的DOY；4.计算GDD(生长度日)、GPD(类似GDD的降水累计)
    :return: MCD_path = 'D:\\E\\data\\grass_yield\\MCD12Q2\\MCD12Q2v6_1\\'
            MCD_path + \\RFinData1.csv：带GDD、GPD的气象、Greenup、Dormancy数据
            MCD_path + \\RFinSeries1.csv：仅气象、Greenup、Dormancy数据
    """
    data = pd.read_csv(MCD_path + "\\StationMetePheno1.csv")
    pheno = pd.read_csv(MCD_path + "\\StationPheno1.csv")
    pd.set_option('display.max_columns', None)
    # print(data.head(1000))
    # print(data.info(),'\n',data.describe())
    # print(data.head(), data.shape)
    # stationID, stationInfo = StatDa2Rast()
    # print(stationID)
    # data cleaning
    # 清除掉包含“32767”的数据点
    data.drop('hour', axis=1, inplace=True)
    mask_Greenup = data['Greenup'] == 32767
    mask_Dormancy = data['Dormancy'] == 32767
    data.drop(data[mask_Greenup].index, inplace=True)
    # data_Dormancy = data.drop(data.loc[mask_Dormancy].index)
    data.reset_index(drop=True, inplace=True)
    # mask = (data['year'] == 2016) & (data['STATION_ID'] == 53083099999)
    # print(data[mask].empty)

    # 提取不同站点各年物候期做标签
    station_Greenup = pd.concat([data['STATION_ID'], data['year'], data['Greenup']], axis=1)
    station_Greenup.drop_duplicates(inplace=True)
    station_Greenup.reset_index(drop=True, inplace=True)
    station_Dormancy = pd.concat([data['STATION_ID'], data['year'], data['Dormancy']], axis=1)
    station_Dormancy.drop_duplicates(inplace=True)
    station_Dormancy.reset_index(drop=True, inplace=True)
    # print(data.head(),station_Greenup.head())
    # print(station_Greenup)
    # print(station_Greenup.iloc[333, [0,1,-1]], station_Greenup.iloc[736, [0,1,-1]])

    # 查找标签中的异常值
    # 将MCD12Q2的物候期（距离1970-01-01的天数）转成年积日，没删除Greenup和Dormancy中值为32767的数据
    year1 = '1970-1-1'
    # p表示parse，表示分析的意思，所以strptime是给定一个时间字符串和分析模式，返回一个时间对象
    d0 = datetime.datetime.strptime(year1, '%Y-%m-%d')
    for index, row in  station_Greenup.iterrows():
        year2 = row['year']
        # print(year2)
        d1 = datetime.datetime(year2, 1, 1)
        # 计算year2年第1天距离1970.01.01的天数
        interval = d1 - d0
        # 计算年积日
        row['Greenup'] = row['Greenup']-interval.days
    # print(len(station_Greenup)) # 有1882个样本点
        # print(row['STATION_ID'],row['year'],row['Greenup'])
    # for index, row in station_Dormancy.iterrows():
    #     year2 = row['year']
    #     d1 = datetime.datetime(year2, 1, 1)
    #     # 计算year2年第1天距离1970.01.01的天数
    #     interval = d1 - d0
    #     #计算年积日
    #     row['Dormancy'] = row['Dormancy'] - interval.days

    # # 打印箱线图，查找异常点：(看着没有)
    # bp = plt.boxplot(station_Greenup['Greenup'], patch_artist=True, widths=0.3, showmeans=False,
    #                      medianprops={'lw': 1, 'color': 'black'},
    #                      zorder=1,
    #                      # 设置异常点属性，如点的形状、填充色和点的大小
    #                      flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 8})
    #
    # bp1 = plt.boxplot(station_Dormancy['Dormancy'], patch_artist=True, widths=0.3, showmeans=False,
    #                       medianprops={'lw': 1, 'color': 'black'},
    #                       zorder=1,
    #                       # 设置异常点属性，如点的形状、填充色和点的大小
    #                       flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 8})
    # plt.show()

    # # 处理异常值
    # # 返青期的两个<80和>200异常点直接删除
    # mask = (station_Greenup['Greenup'] > 200) | (station_Greenup['Greenup'] < 80)
    # #记录离群值所在的index
    # index_outlier = station_Greenup[mask].index
    # # print(index_outlier)
    # # 实例个数
    # pointnum = len(station_Greenup)
    # #drop掉离群值
    # station_Greenup.drop(station_Greenup[mask].index, inplace=True)
    # station_Greenup.reset_index(drop=True, inplace=True)
    # bp = plt.boxplot(station_Greenup['Greenup'], patch_artist=True, widths=0.3, showmeans=False,
    #                  medianprops={'lw': 1, 'color': 'black'},
    #                  zorder=1,
    #                  # 设置异常点属性，如点的形状、填充色和点的大小
    #                  flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 3})
    #
    # plt.show()

    # print(data.head())
    # 计算每日GDD, 基温取5°C,从1月1日开始计算 和GDP和[[@Estimate2022Masago]]
    #遍历data的每行，逐行计算GDD、GPD和DOY;DOY of Greenup和Dormancy
    data['GDD'], data['GPD'],data['Gdoy'],data['Ddoy'] = 0,0,0,0
    data['DOY'] = 1
    year1 = '1970-1-1'
    # p表示parse，表示分析的意思，所以strptime是给定一个时间字符串和分析模式，返回一个时间对象
    d0 = datetime.datetime.strptime(year1, '%Y-%m-%d')
    # print(data['DOY'],data['GDD'],data)
    doy = 0
    for index, row in  data.iterrows():
        year2 = int(row['year'])
        d1 = datetime.datetime(year2, 1, 1)
        # 计算year2年第1天距离1970.01.01的天数
        interval = d1 - d0
        # 每年doy从1起，每日（行）+1
        doy = doy + 1
        if ((row['month']==1) and (row['day']==1)):
            sumtemp,sumprecip = 0,0
            doy = 1
        latitude = row['LATITUDE']
        sin1 = 2*math.pi*doy/365-1.39
        tan1 = math.tan(0.409*math.sin(sin1))
        tan2 = -math.tan(latitude*math.pi/180) #角度转弧度参与运算
        photoperiod = (24/math.pi) * math.acos(tan2*tan1)
        # photoperiod = (24/math.pi) * math.acos((-math.tan(latitude*math.pi/180))*(math.tan(0.409*math.sin(2*math.pi*doy/365-1.39))))
        sumtemp = sumtemp + max(row['Temp_mean']-5, 0)     #基温取5°C，所以要-5，-5后和0比较，取较大值
        # print(row['Temp_mean']-5)
        sumprecip = sumprecip + row['precipitation_sum']
        # row['PHO'] = photoperiod
        # row['GDD'] = sumtemp
        # row['GPD'] = sumprecip
        # row['DOY'] = doy
        # row['Gdoy'] = row['Greenup']-interval.days
        # row['Ddoy'] = row['Dormancy']-interval.days

        # print(row)
        # iterrows迭代修改是临时的，原数据改不了，需添加下面这句
        data.loc[index,'PHO'] = photoperiod
        data.loc[index,'GDD'] = sumtemp
        data.loc[index,'GPD'] = sumprecip
        data.loc[index,'DOY'] = doy
        data.loc[index,'Gdoy'] = row['Greenup']-interval.days
        data.loc[index, 'Ddoy'] = row['Dormancy']-interval.days
    # print(data)

    # 按[天数上值,特征]准备sklearn randomforest的数据
    metdata = [data['STATION_ID'],data['LONGITUDE'],data['LATITUDE'],data['ELEVATION'],
               data['year'],data['DOY'],data['Temp_mean'],data['VaporP_mean'],data['GDD'],data['GPD'],
               data['precipitation_sum'],data['dew-pointtemperature'],data['windspeed_mean'], data['PHO'],
               data['cloudamount'],data['Gdoy'],data['Ddoy']]
    meter_data = pd.concat(metdata,axis=1, ignore_index=False)
    meter_data.to_csv(MCD_path +'\\experiment2'+ "\\RFinData1.csv", index=False)
    # # 按原始数据的[天数上值，特征]准备sklearn randomforest的数据，为时序算法提供对照
    # metseries = [data['STATION_ID'],data['LONGITUDE'],data['LATITUDE'],data['ELEVATION'],
    #            data['year'],data['DOY'],data['Temp_mean'],data['precipitation_sum'],data['dew-pointtemperature'],data['Gdoy'],data['Ddoy']]
    # metseries = pd.concat(metseries, axis=1, ignore_index=False)
    # metseries.to_csv(MCD_path + '\\RFinSeries1.csv', index=False)

def DataPreproc2():
    """
    准备sklearn和sktime格式的输入数据：
    1.除去数据中的NaN值；
    2.保证2年以上的连续数据，且每年数据为365天
    3.由DataTimeRange()完成，本函数未完成：将原始 RFinData1.csv中每个站点的前一年平均返青期起截取365天数据，形成参与建模的时序气象数据，前一年的Gdoy要修改成当年Gdoy值，
    时序数据的时间区间应该是前一年的后半年+当年的前2月的值，不能直接用全年数据，因为只有本年前2月的数据对物候返青期有影响：

    :return:
    """
    meter_data = pd.read_csv(MCD_path +'experiment2'+ "\\RFinData1.csv")
    metseries = pd.read_csv(MCD_path + "RFinSeries1.csv")
    pd.set_option('display.max_columns', None)
    # print(meter_data)
    # # 测试meter_data中是否有nan
    # meterdata_nanvalues = meter_data.isna().any()
    # metseries_nanvalues = metseries.isna().any()
    # print(meterdata_nanvalues, metseries_nanvalues)

    # 1.清洗数据：除去NaN：将NaN的值用线性插值代替
    meter_data = meter_data.interpolate()
    # # 测试meter_data中是否还有nan
    # meterdata_nanvalues = meter_data.isna().any()
    # metseries_nanvalues = metseries.isna().any()
    # print(meterdata_nanvalues)


    # 2.1 遍历每个站点，提取有连续年份的数据（每个站点的前一年平均返青期起截取365天数据，形成参与建模的时序气象数据）
    # 2.2 遍历每年数据，保证为365天
    stationID = meter_data['STATION_ID']
    stationID.drop_duplicates(inplace=True)
    stationID.reset_index(drop=True, inplace=True)

    timeseries = pd.DataFrame()
    for id in stationID:
        stationID_data = meter_data[(meter_data.loc[:, 'STATION_ID'] == id)]
        station_year = stationID_data['year'].copy()
        station_year.drop_duplicates(inplace=True)
        station_year.reset_index(drop=True, inplace=True)
        # print(station_year,type(station_year))
        for year in list(station_year):
            # 判断series中是否包含某值用series.unique()
            # if year + 1 in station_year.unique():
            if year+1 in list(station_year):
                # print(id, year,year+1)
                meter_data1 = stationID_data[stationID_data.loc[:,'year'] == year]
                meter_data1.reset_index(drop=True, inplace=True)
                if len(meter_data1)!=365:
                    meter_data1 = meter_data1.iloc[0:365,:]
                # print(type(meter_data1))
                timeseries = pd.concat([timeseries,meter_data1], ignore_index=True)
                # print(timeseries)
                meter_data2 = stationID_data[stationID_data.loc[:,'year'] == year+1]
                meter_data2.reset_index(drop=True, inplace=True)
                if len(meter_data2) != 365:
                    meter_data2 = meter_data2.iloc[0:365, :]
                timeseries = pd.concat([timeseries,meter_data2], ignore_index=True)
        timeseries.drop_duplicates(inplace=True)

        #检验 timeseries 中的年份是否符合具有连续2年以上的要求
        # timeseries_year = timeseries['year'].copy()
        # timeseries_year.drop_duplicates(inplace=True)
        # timeseries_year.reset_index(drop=True, inplace=True)
        # print(id,timeseries_year)

    timeseries.reset_index(drop=True, inplace=True)
    timeseries.to_csv(MCD_path +'\\experiment2' + "\\SklearnInData1.csv", index=False)

