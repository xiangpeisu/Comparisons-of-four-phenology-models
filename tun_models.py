def tun_RF(x_train,y_train,max_evals):
    """

    :param x_train:
    :param y_train:
    :param max_evals: 调参迭代次数
    :return:
    """
    # 1.将train data划分出calibration data
    # 划分数据 训练集与校验集
    TrainX, CaliX, TrainY, CaliY = train_test_split(x_train, y_train,
                                                    test_size=1 / 5,  # 指定数据中测试集所占比例
                                                    # Controls the shuffling applied to the data before applying the split.
                                                    # Pass an int for reproducible output across multiple function calls.
                                                    random_state=None,
                                                    shuffle=True
                                                    )

    def objective(params):
        """
        Objective function for hyperparameter optimization.

        Parameters:
        - params (dict): Dictionary containing hyperparameters to optimize.

        Returns:
        - float: Value of the objective function to minimize.
        """
        # Extract hyperparameters from the params dictionary
        n_estimators = int(params['n_estimators'])
        max_depth = int(params['max_depth'])
        min_samples_split = int(params['min_samples_split'])
        min_samples_leaf = int(params['min_samples_leaf'])

        # Create a Random Forest Regressor with the specified hyperparameters
        rf_regressor = RandomForestRegressor(n_estimators=n_estimators,
                                             max_depth=max_depth,
                                             min_samples_split=min_samples_split,
                                             min_samples_leaf=min_samples_leaf,
                                             random_state=42)

        # Train the Regressor on the training data
        rf_regressor.fit(TrainX, TrainY)
        # Compute the ROC AUC score on the validation data
        r2sco_Randomforest_test = r2_score(CaliY, rf_regressor.predict(CaliX))
        # Return 1 - r2sco_Randomforest_test because Hyperopt aims to minimize the objective function
        return 1 - r2sco_Randomforest_test

    # 定义一个参数空间
    # Define the search space for hyperparameters
    params_space = {
        'n_estimators': hyperopt.hp.choice('n_estimators', range(100,1000,50)),
        'max_depth': hyperopt.hp.randint('max_depth', 1,50),
        'min_samples_split': hyperopt.hp.choice('min_samples_split', range(2, 10)),
        'min_samples_leaf': hyperopt.hp.choice('min_samples_leaf', range(1, 10))
    }
    import warnings
    warnings.filterwarnings("ignore")
    # 定义超参数优化器
    trials = Trials()
    best = fmin(fn=objective,
                space=params_space,
                algo=tpe.suggest,
                max_evals = max_evals,
                trials=trials)
    # print("BestParameter: ", best)
    # The optimal parameters after training are obtained
    best_params = hyperopt.space_eval(params_space, best)
    # best_params['random_state'] = 42
    return best_params
def tun_KNN(x_train,y_train,max_evals):
    """
    Hyperopt调参，并计算test data 的predict值，R2，RMSE，MAE
    :param x_train:
    :param y_train:
    :param max_evals: 调参迭代次数
    :return: 最佳参数，test data的predict值，R2，RMSE，MAE
    """
    # 1.将train data划分出calibration data
    # 划分数据 训练集与校验集
    TrainX, CaliX, TrainY, CaliY = train_test_split_timeseries(x_train, y_train,
                                                    test_size=1 / 5,  # 指定数据中测试集所占比例
                                                    )

    def objective(params):
        """
        Objective function for hyperparameter optimization.

        Parameters:
        - params (dict): Dictionary containing hyperparameters to optimize.

        Returns:
        - float: Value of the objective function to minimize.
        """
        # Extract hyperparameters from the params dictionary
        n_neighbors = int(params["n_neighbors"])
        distance = params["distance"]
        weights = params["weights"]
        # leaf_size = int(params["leaf_size"])
        # colsample_bytree = int(params["colsample_bytree"])
        # min_split_loss = int(params["min_split_loss"])
        # reg_alpha = int(params["reg_alpha"])
        # Create a KNN Regressor with the specified hyperparameters
        knn_regressor = KNeighborsTimeSeriesRegressor(n_neighbors = n_neighbors,
                                                      distance = distance,
                                                      weights = weights)
        # Train the Regressor on the training data
        knn_regressor.fit(TrainX, TrainY)
        # Compute the R2 score on the validation data
        r2sco_Knn_test = r2_score(CaliY, knn_regressor.predict(CaliX))
        # Return 1 - r2sco_Knn_test because Hyperopt aims to minimize the objective function
        return 1 - r2sco_Knn_test

    # 定义一个参数空间
    # Define the search space for hyperparameters
    params_space = {
        'n_neighbors': hyperopt.hp.choice('n_neighbors', range(10, 100, 10)),  # 邻居数量
        'distance': hyperopt.hp.choice('distance',['euclidean', 'dtw']),  # 邻居数量
        'weights': hyperopt.hp.choice('weights',['uniform', 'distance'])  # 权重类型
    }
    import warnings
    warnings.filterwarnings("ignore")
    # 定义超参数优化器
    trials = Trials()
    best = fmin(fn=objective,
                space=params_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)
    # print("BestParameter: ", best)
    # The optimal parameters after training are obtained
    best_params = hyperopt.space_eval(params_space, best)
    # best_params['random_state'] = 42
    return best_params
  

def tun_model(time):
    """
    对不同模型用hyperopt调参
    :param time: 第time次运行，要重复5次
    :return:
    """
    max_evals = 150
    for ratio_train in np.linspace(10,100,10):
        print('time:',time,'ratio_train:', ratio_train)
        # 加载datapreproc(time)函数新划分出test和train的数据并转成sktime格式的time series数据
        x_train_nested = np.load(article_path + '\sample_size_test' + "\\x_train_nested" + str(int(ratio_train / 10)) + '_' + str(0) + '.npy',allow_pickle=True)
        x_train_array = np.load(article_path + '\sample_size_test' + "\\x_train_array" + str(int(ratio_train / 10)) + '_' +
                                str(time) + '.npy')
        y_train_array = np.load(article_path + '\sample_size_test' + "\\y_train_array" + str(int(ratio_train / 10)) + '_' +
                                str(time) + '.npy')
        # 调用Hyperopt调参函数：
        # best_params_knn = tun_KNN(x_train_array,y_train_array,max_evals)
        # print(best_params_knn)
        # dump(best_params_knn,article_path + '\sample_size_test' + '\\best_param_KNN_' + str(int(ratio_train / 10)) +'_'
        #      + str(time) + '.pkl')
        best_params_Rocket = tun_Rocket(x_train_array, y_train_array,max_evals)
        print(best_params_Rocket)
        dump(best_params_Rocket,article_path + '\sample_size_test' + '\\best_param_Rocket_' + str(int(ratio_train / 10)) + '_'
             + str(time) + '.pkl')
        # # 加载RF和XGBoost所用数据
        # x_train_array_RF = np.loadtxt(article_path + '\sample_size_test' + "\\x_train_array_RF_" + str(int(ratio_train / 10)) + '_' + str(
        #     time) + '.csv' )
        # y_train_array_RF = np.loadtxt(article_path + '\sample_size_test' + "\\y_train_array_RF_" + str(int(ratio_train / 10)) + '_' + str(
        #     time) + '.csv')
        # x_test_array_RF = np.loadtxt(
        #     article_path + '\sample_size_test' + "\\x_test_array_RF_" + str(int(ratio_train / 10)) + '_' + str(time) + '.csv')
        # y_test_array_RF = np.loadtxt(
        #     article_path + '\sample_size_test' + "\\y_test_array_RF_" + str(int(ratio_train / 10)) + '_' + str(time) + '.csv')
        # print(x_train_array_RF.shape[0]/365)
        # # 调用Hyperopt调参函数：
        # best_params_RF = tun_RF(x_train_array_RF, y_train_array_RF, max_evals)
        # dump(best_params_RF,
        #      article_path + '\sample_size_test' + '\\best_params_RF_' + str(int(ratio_train / 10)) + '_'
        #      + str(time) + '.pkl')
        # best_params_XGB = tun_XGBoost(x_train_array_RF, y_train_array_RF, max_evals)
        # dump(best_params_XGB,
        #      article_path + '\sample_size_test' + '\\best_params_XGB_' + str(int(ratio_train / 10)) + '_'
        #      + str(time) + '.pkl')
