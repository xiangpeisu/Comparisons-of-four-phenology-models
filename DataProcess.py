import pandas as pd
class DataSource(object):
    ext: str
    def __init__(self, infilepath,outfilepath) -> None:
        if not infilepath.suffix == self.ext:
            raise ValueError("Invalid file format")
        self.infilepath = infilepath
        self.outfilepath = outfilepath
class StationID(DataSource):
    ext = '.csv'
    def extract(self,location,time):
        """
        从infilepath的文件中筛选出经度在：location['Min_Lon']~location['Max_Lon']，
        纬度在：location['Min_Lat']~location['Max_Lat']
        时间跨度在：time['Min_year']~time['Max_year']年的站点编号

        :param infilepath，outfilepath,
               location['Min_Lon':None,'Max_Lon':None,'Min_Lat':None,'Max_Lat':None]
               time['Min_year':None,'Max_year':None]
        :return: 站点编号stationID，站点情况stationInfo
        """
        # loaddata
        data = pd.read_csv(self.infilepath)
         # 根据经纬度筛选出所需数据
        neilonbool1 = data['LONGITUDE'] >= location['Min_Lon']         #101
        neilonbool2 = data['LONGITUDE'] <= location['Max_Lon']
        neilonbool = neilonbool1 & neilonbool2

        neilatbool1 = data['LATITUDE'] >= location['Min_Lat']
        neilatbool2 = data['LATITUDE'] <= location['Max_Lat']           #47
        neilatbool = neilatbool1 & neilatbool2
        neilatlonbool = neilonbool & neilatbool

        # 根据起始、结束时间筛选出所需数据
        # 对data['BEGIN_DATE']整体使用lambda函数
        begin = data['BEGIN_DATE'].apply(lambda x: x[0:4]) #data['BEGIN_DATE']值形如20020220，x[0:4]取其前4个代表年份的字符
        print(type(data['BEGIN_DATE']))
        end = data['END_DATE'].apply(lambda x: x[0:4])
        beginbool = begin.apply(lambda x: int(x)) <= time['Min_year']
        endbool = end.apply(lambda x: int(x)) >= time['Max_year']
        datebool = beginbool & endbool

        bool = neilatlonbool & datebool
        stationID = data['STATION_ID'][bool]
        stationLon = data['LONGITUDE'][bool]
        stationLat = data['LATITUDE'][bool]
        # 海拔高度
        stationElev = data['ELEVATION'][bool]

        stationInfo = pd.concat([stationID, stationLon, stationLat, stationElev], axis=1, ignore_index=False)
        # 屏蔽指定stationID的站点
        # 站点54577099999数据缺失严重
        # mask = (stationID == 54577099999)
        # stationID.drop(stationID[mask].index, inplace=True)
        # stationInfo.drop(stationInfo[stationInfo['STATION_ID'] == 54577099999].index, inplace=True)
        # stationInfo.reset_index(drop=True, inplace=True)
        # stationID.reset_index(drop=True, inplace=True)
        stationInfo.to_csv(self.outfilepath)
        return stationID, stationInfo

class MCD12Q2(DataSource):
    ext = '.tif'
    def extract_stationID(self) -> pd.DataFrame:
        """
        :param:
        :return:
        """
        # loaddata
        data = pd.read_csv(self.filepath)

class ISD_Lite(DataSource):
    ext = '.gz'
