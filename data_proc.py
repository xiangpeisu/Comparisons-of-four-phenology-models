def datapreproc(time):
    """
    修改SpitSeriesTrainTest(time)函数的随机抽取部分，从35个站点中随即均匀分别抽取167、335、502、670、837、1004、1172、1339、
    1507、1674年的数据（1674年总数据的10%~100%）
    应先统一提取test data，用于不同sample_size调参训练后的模型的评价指标R2、RMSE、MAE的计算，不同模型使用相同的test data！
    因为考察的是不同sample_size对模型的影响，因此test data抽取最大sample_size，再在最大sample_size数据中使用最小sample_size
    做两种情况的对比研究
    time:第time次运行，要重复运行5次
    :return:
    """
    # 1.加载数据
    print(time, ':datapreproc_learning_curve()')
    data = pd.read_csv(MCD_path + '\\experiment2' + "\\DataTimeRange.csv")
    pd.set_option('display.max_columns', None)
    # print(data.head(5))
    # print(len(data), len(data) / 365)  # 有1674年的数据

    # 2. 划分train data和test data，随机提取int((len(data) / 365) * 0.3个某整年数据，不参与训练而作test数据
    # 提取0.3的数据做test data,x_test仍然是dataframe格式
    ide_test = [int(i) for i in range(0, int((len(data) / 365)))]
    index_test = random.sample(ide_test, (int((len(data) / 365) * 0.3)))  # 从ide_test中随机抽取data*0.3总数的数据,并乱序
    # print(len(ide_test),len(index_test),type(index_test))
    # print(index_test)
    data_train = data.copy(deep=False)
    print("data_train.shape: ",data_train.shape)
    data_test = pd.DataFrame()
    num_test = int((len(data)/365)*0.3)
    print(num_test)
    for i in range(0, num_test):  #i从0到502个test data
        index_start = int(index_test[i]*365)
        index_end = int((index_test[i]+1)*365)
        data_test = pd.concat([data_test,data.iloc[index_start: index_end, :]],axis=0, ignore_index=False)
        # x_test.append(x.iloc[index_start: index_end, :])
        data_train = pd.concat([data.iloc[index_start: index_end, :],data_train])
        data_train.drop_duplicates(keep=False,inplace=True,ignore_index=False) #keep=False：删除所有重复项,inplace=True:修改原数据,ignore_index:不重新编号
    print(data_test.shape[0]/365,data_train.shape[0]/365)
        # 出错：某些行删不掉：x_train.drop(x_train.index[index_start:index_end], inplace=True)
        # print(shape_x_train_1 - x_train.shape[0], len(x.iloc[index_start: index_end, :]),x_train.shape)
        # shape_x_train_1 = x_train.shape[0]

    # print(x_test.head(5))
    data_test.reset_index(drop=True, inplace=True) # drop=True:旧索引不会添加为列，直接重置为默认整数索引
    # print(data_test.head(5))
    data_train.reset_index(drop=True, inplace=True)
    data_test.to_csv(article_path + '\sample_size_test' + "\\data_test_pd_" + '4model_samplesize_test' + str(time) + '.csv',
                  index=False)
    data_train.to_csv(article_path + '\sample_size_test' + "\\data_train_pd_" + '4model_samplesize_test' + str(time) + '.csv',
                  index=False)
    print(data_train.shape[0]/365,data_test.shape[0]/365,data.shape[0]/365)
    # 将pd转成array，再保存
    data_test_array_forest = data_test.values
    data_train_array_forest = data_train.values
    np.savetxt(article_path + '\sample_size_test' + "\\data_test_array_" + '4model_samplesize_test' + str(time) +'.csv',
               data_test_array_forest)
    np.savetxt(article_path + '\sample_size_test' + "\\data_train_array_" + '4model_samplesize_test' + str(time) + '.csv',
               data_train_array_forest)

    # 加载划分好的train data和test data
    data_train = pd.read_csv(article_path + '\sample_size_test' + "\\data_train_pd_" + '4model_samplesize_test' + str(time) + '.csv')
    data_test = pd.read_csv(article_path + '\sample_size_test' + "\\data_test_pd_" + '4model_samplesize_test' + str(time) + '.csv')
    # print(data_train.shape, data_test.shape)
    # 生成不同sample size 的 train data 和 test data
    # 将test data 按10%、20%、30%、40%、50%、60%、70%、80%、90%、100%提取，准备做test data size对模型accuracy的影响test
    ide_train = [i for i in range(int(len(data_train) / 365))]  # 列出从0~1172(1674*0.7)的列表
    ide_test = [i for i in range(int(len(data_test) / 365))]  # 列出从0~502(1674*0.3)的列表
    # print(len(ide_train),len(ide_test))
    for ratio in np.linspace(10, 100, 10):  # np.linspace(start,end,totals_amount)
        # print("ratio:",ratio)
        samples = pd.DataFrame()
        num_train = int((len(data_train)/365) * (ratio/100))
        num_test = int((len(data_test)/365) * (ratio/100))
        # print(num_test,num_train)
        index_train = random.sample(ide_train, num_train)  # 从ide_train中随机抽取(ratio/100)总数的数据,并乱序
        index_test = random.sample(ide_test, num_test)  # 从ide_test中随机抽取(ratio/100)总数的数据,并乱序
        for i in range(0, num_train):  #i从0到1172个train data
            index_start = int(index_train[i]*365)
            index_end = int((index_train[i]+1)*365)
            samples = pd.concat([samples,data_train.iloc[index_start: index_end, :]],axis=0, ignore_index=False)
            # print(i,samples.shape)
        samples.reset_index(drop=True, inplace=True) # drop=True:旧索引不会添加为列，直接重置为默认整数索引
        data_train_pd = samples.copy(deep=False)
        # print(data_train_pd.shape[0]/365)
        data_train_pd.to_csv(article_path + '\sample_size_test' + "\\data_train_pd" + str(int(ratio / 10)) +'_' + str(time) + '.csv', index=False)
        samples = pd.DataFrame()
        # print(data_train_pd.shape, samples.shape)
        for i in range(0, int((len(data_test) / 365) * (ratio / 100))):  # i从0到1172个train data
            index_start = int(index_test[i] * 365)
            index_end = int((index_test[i] + 1) * 365)
            samples = pd.concat([samples, data_test.iloc[index_start: index_end, :]], axis=0, ignore_index=False)
        samples.reset_index(drop=True, inplace=True)  # drop=True:旧索引不会添加为列，直接重置为默认整数索引
        data_test_pd = samples.copy(deep=False)
        data_test_pd.to_csv(article_path + '\sample_size_test' + "\\data_test_pd" + str(int(ratio / 10)) + '_' + str(time) + '.csv', index=False)
        # print(data_test_pd.shape[0]/365,data_train_pd.shape[0]/365)

        # # 加载划分好的数据
        # data_train_pd = pd.read_csv(article_path + '\sample_size_test' + "\\data_train_pd" + str(int(ratio / 10)) +'_' + str(time) + '.csv')
        # data_test_pd = pd.read_csv(article_path + '\sample_size_test' + "\\data_test_pd" + str(int(ratio / 10)) + '_' + str(time) + '.csv')
        # 形成x和y
        Label_train = {"Gdoy": data_train_pd.pop("Gdoy")}  # 将data中的"Gdoy"列提出作为Label的值
        Label_test = {"Gdoy": data_test_pd.pop("Gdoy")}  # 将data中的"Gdoy"列提出作为Label的值
        # print(Label_train,type(Label_train))
        data_train_pd.pop("Ddoy")  # 去掉data中的"Ddoy"列
        data_test_pd.pop("Ddoy")
        x_train_pd, y_train_pd = data_train_pd, (pd.DataFrame(Label_train))
        x_test_pd, y_test_pd = data_test_pd, (pd.DataFrame(Label_test))
        # print(y_train_pd.shape, x_train_pd.shape, y_test_pd.shape, x_test_pd.shape)
        # 删除 STATION_ID 和 year列
        """
        将x_train_pd，y_train_pd，x_test_pd，y_test_pd
        整理成sklearn_RF要求格式 x_train_array_RF,y_train_array_RF,x_test_array_RF等
        整理成sktime格式的输入数据x_train_array，y_train_array,x_test_array，y_test_array，并保存成 *.npy文件
        最后得到的数据列为：LONGITUDE	LATITUDE	ELEVATION	DOY	Temp_mean	VaporP_mean	GDD	GPD	precipitation_sum
                        dew-pointtemperature	windspeed_mean	PHO	cloudamount，共13列
        """
        x_test_pd.pop('STATION_ID')
        x_test_pd.pop('year')
        x_train_pd.pop('STATION_ID')
        x_train_pd.pop('year')
        # 仅保留小数点后4位
        x_train_pd = x_train_pd.round(4)
        x_test_pd = x_test_pd.round(4)
        y_train_pd = y_train_pd.round(4)
        y_test_pd = y_test_pd.round(4)
        # 按RF要求从DataFrame转成nparray格式整理好后，再次保存
        x_train_array_RF = x_train_pd.values
        y_train_array_RF = y_train_pd.values
        x_test_array_RF = x_test_pd.values
        y_test_array_RF = y_test_pd.values
        np.savetxt(article_path + '\sample_size_test' + "\\x_train_array_RF_" + str(int(ratio / 10)) + '_' + str(time) + '.csv', x_train_array_RF)
        np.savetxt(article_path + '\sample_size_test' + "\\y_train_array_RF_" + str(int(ratio / 10)) + '_' + str(time) + '.csv', y_train_array_RF)
        np.savetxt(article_path + '\sample_size_test' + "\\x_test_array_RF_" + str(int(ratio / 10)) + '_' + str(time) + '.csv', x_test_array_RF)
        np.savetxt(article_path + '\sample_size_test' + "\\y_test_array_RF_" + str(int(ratio / 10)) + '_' + str(time) + '.csv', y_test_array_RF)
        # 整理成sktime的输入格式：3D nparray（样例，特征，时间）
        x_tem, y_tem = [], []
        # print(x_train_pd.shape[0]/365,y_train_pd.shape)
        for i in range(0, int(len(x_train_pd) / 365)):  # i取值0~int(len(x_train)/365，最后一个值是int(len(x_train)/365-1
        # for i in range(0, 2):
            x_tem.append(x_train_pd.iloc[365 * i: 365 * (i + 1), :].T.values) # .T即行列互换,13 row x 365 columns,列表元素是 nparray
            # print(x_train_pd.iloc[365 * i: 365 * (i + 1), :].T)
            # print(x_train_pd.iloc[365 * i: 365 * (i + 1), :])
            # print(x_train_pd.iloc[365 * i: 365 * (i + 1), :].values.T)
            # print(x_tem)
            # 每年的Gdoy提取一个值，长度从610645变成610645/365
            # print(y_train_pd.shape)
            y_tem.append(y_train_pd.iloc[i * 365])
            # print(type(y_train.iloc[i*365]))
        x_train_array = np.array(x_tem)
        # print(x_train_array.shape)
        y_train_array = np.array(y_tem)
        # print(y_train_array)
        y_train_array = y_train_array[:, 0]
        # print(type(y_train_array),y_train_array.shape)
        # print("x_y_train_array:", x_train_array.shape, y_train_array.shape, "x_y_test_array", x_test_pd.shape,
        #       y_test_pd.shape)
        x_tem, y_tem = [], []
        # print(len(x_test),len(y_test))
        for j in range(0, int(len(x_test_pd) / 365)):
            x_tem.append(x_test_pd.iloc[365 * j:365 * (j + 1), :].T.values)
            y_tem.append(y_test_pd.iloc[j * 365])
        x_test_array = np.array(x_tem)
        y_test_array = np.array(y_tem)
        y_test_array = y_test_array[:, 0]
        print("ratio_train:",ratio,"sktime:", x_test_array.shape, y_test_array.shape,x_train_array.shape,y_train_array.shape)
        # 存放3维以上array用 np.save ，读取用 np.load
        np.save(article_path + '\sample_size_test' + "\\x_train_array" + str(int(ratio / 10)) + '_' + str(time) + '.npy', x_train_array)
        np.save(article_path + '\sample_size_test' + "\\y_train_array" + str(int(ratio / 10)) + '_' + str(time) + '.npy', y_train_array)
        np.save(article_path + '\sample_size_test' + "\\x_test_array" + str(int(ratio / 10)) + '_' + str(time) + '.npy', x_test_array)
        np.save(article_path + '\sample_size_test' + "\\y_test_array" + str(int(ratio / 10)) + '_' + str(time) + '.npy', y_test_array)
        # # 3D 数据转换为 nested format
        # x_train_array = np.load(article_path + '\sample_size_test' + "\\x_train_array" + str(int(ratio / 10)) + '_' + str(0) + '.npy')
        # # y_train_array = np.load(article_path + '\sample_size_test' + "\\y_train_array" + str(int(ratio / 10)) + '_' + str(0) + '.npy')
        # x_test_array = np.load(article_path + '\sample_size_test' + "\\x_test_array" + str(int(ratio / 10)) + '_' + str(0) + '.npy')
        # # y_test_array = np.load(article_path + '\sample_size_test' + "\\y_test_array" + str(int(ratio / 10)) + '_' + str(0) + '.npy')
        # x_train_nested = convert_selfdownfromgit.from_3d_numpy_to_nested(x_train_array)
        # x_test_nested = convert_selfdownfromgit.from_3d_numpy_to_nested(x_test_array) # 将 x_test_array 转成nested并保存
        # np.save(article_path + '\sample_size_test' + "\\x_train_nested" + str(int(ratio / 10)) + '_' + str(0) + '.npy', x_train_nested)
        # np.save(article_path + '\sample_size_test' + "\\x_test_nested" + str(int(ratio / 10)) + '_' + str(0) + '.npy', x_test_nested)

  def train_test_split_timeseries(x, y,test_size):
    """

    :param x: sktime格式的输入: 3D nparray（样例，特征，时间）
    :param y:
    :param test_size:

    :return: x_train,x_cali,y_train,y_cali
    """
    # 2. 划分train data 和 calibration data，随机提取int((len(data) / 365) * 0.2个某整年数据，作 calibration 数据
    # 随机提取0.2的数据做calibration data
    num_cali = int(x.shape[0] * test_size)
    # print(int(x.shape[0]),num_cali)
    ide_cali = [int(i) for i in range(0, int(x.shape[0]))]
    index_cali = random.sample(ide_cali, num_cali)  # 从ide__cali中随机抽取test_size的数据,并乱序
    index_Bool =[True for i in range(0, int(x.shape[0]))] # 后面删除x中cali data时用
    x_train = x
    y_train = y
    x_cali = x[np.newaxis,index_cali[0]]
    # print(x_cali.shape)
    y_cali = y[index_cali[0]]
    index_Bool[0]=False
    # print("x_train.shape: ",x_train.shape,y_train.shape)
    for i in range(1, num_cali):  # 提取test_size个 calibration data
        index = int(index_cali[i])
        index_Bool[index] = False  # 用false记录提取的index
        # print(index)
        # print(x_train[index,:,:].shape)
        x_train_index = x_train[np.newaxis,index,:,:]
        # y_train_index = y_train[np.newaxis,index,:,:]
        x_cali = np.vstack((x_cali, x_train_index)) #返回新数组，原始数组不会被修改。
        y_cali = np.vstack((y_cali,y_train[index]))
        # print(x_train.shape,x.shape) # x_train变化不会影响x
    x_train = x[index_Bool]
    y_train = y[index_Bool]
    y_cali = np.squeeze(y_cali)
    # print(x_cali.shape,x_train.shape,y_cali.shape,y_train.shape,y.shape)
    return x_train,x_cali,y_train,y_cali
