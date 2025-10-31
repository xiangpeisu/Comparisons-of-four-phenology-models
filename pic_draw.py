def boxes_grouped_trainsize():
    """
    需设置纵轴标题，
    :return:
    """
    metrics_pd = pd.read_csv(article_path + '\sample_size_test' + "\\Result_trainsize"  + '.csv')
    # print(metrics_pd.head())
    #整理成 boxplot 需要格式
    model_metrics = {}
    start = 0
    for model in ['RF', 'XGBoost', 'KNN_TS', 'Rocket_TS']:
        for metric in ['R$^2$', 'RMSE', 'MAE']:
            value = []
            key = metric + '_' + model
            value = (metrics_pd['performance_ave'].iloc[start:start+10]).tolist() # 不含start+10
            # print(value)
            model_metrics[key] = copy.deepcopy(value)  # 如果赋值，则后面value改，这里的model_metrics[key]也会跟着改
            print(key, '\n', len(model_metrics[key]))
            # print(key,'\n',model_metrics[key])
            value_drop012 = value[3:]
            # print(type(model_metrics[key]),model_metrics[key])
            # print(key,':\n',metrics_pd.iloc[start:start + 10])
            for i in range(1,11):
                key1 = 'performance' + str(i)
                value.extend((metrics_pd[key1].iloc[start:start+10]).tolist())
                value_drop012.extend((metrics_pd[key1].iloc[start:start+10]).tolist()[3:])
            print(key,'\n',len(model_metrics[key]))
            model_metrics[key+'_10dup'] = value
            model_metrics[key+'_10dup_drop012'] = value_drop012
            start = start+10
    metrics_10dup = [
               [
                model_metrics['RMSE_KNN_TS_10dup'], model_metrics['RMSE_Rocket_TS_10dup'][0:]
                ,model_metrics['RMSE_RF_10dup'], model_metrics['RMSE_XGBoost_10dup']
              , model_metrics['MAE_KNN_TS_10dup'], model_metrics['MAE_Rocket_TS_10dup'][0:]
                   , model_metrics['MAE_RF_10dup'], model_metrics['MAE_XGBoost_10dup']
               ],
        [
           model_metrics['RMSE_KNN_TS_10dup_drop012'], model_metrics['RMSE_Rocket_TS_10dup_drop012']
            , model_metrics['RMSE_RF_10dup_drop012'], model_metrics['RMSE_XGBoost_10dup_drop012']
            , model_metrics['MAE_KNN_TS_10dup_drop012'], model_metrics['MAE_Rocket_TS_10dup_drop012']
            , model_metrics['MAE_RF_10dup_drop012'], model_metrics['MAE_XGBoost_10dup_drop012']
        ],
        [
           model_metrics['R$^2$_KNN_TS_10dup'], model_metrics['R$^2$_Rocket_TS_10dup'][0:]
           , model_metrics['R$^2$_RF_10dup'], model_metrics['R$^2$_XGBoost_10dup']
        ],
        [
           model_metrics['R$^2$_KNN_TS_10dup_drop012'], model_metrics['R$^2$_Rocket_TS_10dup_drop012']
            , model_metrics['R$^2$_RF_10dup_drop012'], model_metrics['R$^2$_XGBoost_10dup_drop012']
        ]
               ]
    metrics =   [ [
                    model_metrics['RMSE_RF'], model_metrics['RMSE_XGBoost']
                    , model_metrics['RMSE_KNN_TS'], model_metrics['RMSE_Rocket_TS'][0:]
                    # , model_metrics['RMSE_KNN_TS'], model_metrics['RMSE_KNN_TS']
                    , model_metrics['MAE_RF'], model_metrics['MAE_XGBoost']
                    , model_metrics['MAE_KNN_TS'], model_metrics['MAE_Rocket_TS'][0:]
                    #   , model_metrics['MAE_KNN_TS'], model_metrics['MAE_KNN_TS']
                   ],
    [
        model_metrics['RMSE_RF'][3:], model_metrics['RMSE_XGBoost'][3:]
        , model_metrics['RMSE_KNN_TS'][3:], model_metrics['RMSE_Rocket_TS'][3:]
        # , model_metrics['RMSE_KNN_TS'], model_metrics['RMSE_KNN_TS']
        , model_metrics['MAE_RF'][3:], model_metrics['MAE_XGBoost'][3:]
        , model_metrics['MAE_KNN_TS'][3:], model_metrics['MAE_Rocket_TS'][3:]
        #   , model_metrics['MAE_KNN_TS'], model_metrics['MAE_KNN_TS']
    ],
    [
        model_metrics['R$^2$_RF'], model_metrics['R$^2$_XGBoost']
        , model_metrics['R$^2$_KNN_TS'], model_metrics['R$^2$_Rocket_TS'][0:]
        # , model_metrics['R$^2$_KNN_TS'], model_metrics['R$^2$_KNN_TS']
    ],
    [
        model_metrics['R$^2$_RF'][3:], model_metrics['R$^2$_XGBoost'][3:]
        , model_metrics['R$^2$_KNN_TS'][3:], model_metrics['R$^2$_Rocket_TS'][3:]
        # , model_metrics['R$^2$_KNN_TS'], model_metrics['R$^2$_KNN_TS']
    ]
    ]
    # print(metrics_10dup[0])

    # 绘制箱线图
    # boxdata = metrics
    # name_fig = 'fig2_boxes_grouped.png'
    boxdata = metrics_10dup
    name_fig = 'fig2_boxes_10dup.png'
    fignums = ['a', 'b']
    # name_models = ['KNN for time series','Rocket for time series','RandForest','XGBoost']
    name_models = ['KNN_TS', 'Rocket_TS','RF', 'XGBoost']
    nrows_figs, ncols_figs = 2, 2
    fig_with = 14 # 整个图宽14cm
    # 多一行“图例”子图，多子图共用1个图例，将图例作为一个子图置于所有子图之上
    # 指定各子图相对大小:图例3列0.2行，其余子图1行1列
    widths = [1, 1]
    heights = [0.1, 0.9,0.9]
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    figs, axs = plt.subplots(nrows_figs + 1, ncols_figs, dpi=1200
                             , figsize=(fig_with / 2.54, fig_with * nrows_figs * (1 - 0.25) / (ncols_figs * 2.54) + fig_with * 0.05 / 2.54)
                             , constrained_layout=True, gridspec_kw=gs_kw)
    # 合并第0行的3列为一列做共用图例
    for ax in axs[0, 0:]:
        ax.remove()

    bp = axs[1,0].boxplot(boxdata[0], patch_artist=True, widths=0.3, positions=(1,1.4,1.8,2.2, 3.2,3.6,4.0,4.4),
                    # 多图层叠加时，控制画图层的先后：zorder=1，数字越大越后画
                    showmeans=False, medianprops={'lw': 1, 'color': 'black'}, zorder=1,
                    # 设置异常点属性，如点的形状、填充色和点的大小
                    flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 3})
    # for patch, color in zip(bp['boxes'], [color_list]):
    #     patch.set_facecolor(color)
    for patch, color in zip(bp['boxes'], [color_list[13], color_list[12], color_list[0], color_list[2]]):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # 替换横坐标x的值
    x_position = [1, 3.2]
    x_position_fmt = ["RMSE", "MAE"]
    axs[1, 0].text(2.3, 29000, 'Rocket')
    axs[1, 0].text(4.0, 6000, 'Rocket')
    # 第一个参数为显示位置，第二个参数为显示的值
    axs[1,0].set_xticks([i + 1.2 / 2 for i in x_position], x_position_fmt)
    axs[1,0].set_title('(a).',x=-0.05, y=-0.1)
    axs[1,0].set_ylabel('Model Performance')
    axs[1,0].grid(True,linestyle='--', linewidth=0.5)

    # print(len(boxdata))

    bp = axs[1, 1].boxplot(boxdata[1], patch_artist=True, widths=0.3, positions=(1, 1.4, 1.8, 2.2, 3.2, 3.6, 4.0, 4.4),
                           # 多图层叠加时，控制画图层的先后：zorder=1，数字越大越后画
                           showmeans=False, medianprops={'lw': 1, 'color': 'black'}, zorder=1,
                           # 设置异常点属性，如点的形状、填充色和点的大小
                           flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 3})
    # for patch, color in zip(bp['boxes'], color_list):
    #     patch.set_facecolor(color)
    colors = [color_list[13], color_list[12], color_list[0], color_list[2],color_list[13], color_list[12], color_list[0], color_list[2]]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # 替换横坐标x的值
    x_position = [1, 3.2]
    x_position_fmt = ["RMSE", "MAE"]
    # 第一个参数为显示位置，第二个参数为显示的值
    axs[1, 1].set_xticks([i + 1.2 / 2 for i in x_position], x_position_fmt)
    axs[1, 1].set_title('(b).', x=-0.05, y=-0.1)
    axs[1, 1].grid(True,linestyle='--', linewidth=0.5)

    bp = axs[2, 0].boxplot(boxdata[2], patch_artist=True, widths=0.13, positions=(1, 1.2, 1.4, 1.6),
                           # 多图层叠加时，控制画图层的先后：zorder=1，数字越大越后画
                           showmeans=False, medianprops={'lw': 1, 'color': 'black'}, zorder=1,
                           # 设置异常点属性，如点的形状、填充色和点的大小
                           flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 3})
    # for patch, color in zip(bp['boxes'], color_list):
    #     patch.set_facecolor(color)
    for patch, color in zip(bp['boxes'], [color_list[13], color_list[12], color_list[0], color_list[2]]):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # 替换横坐标x的值
    x_position = [1]
    x_position_fmt = ["R$2$"]
    y_position = [0,1,2,3,4]
    y_position_fmt = ['$-8*10^5$', '$-6*10^5$', '$-4*10^5$', '$-2*10^5$','0']
    axs[2, 0].text(1.65, -600000, 'Rocket')
    # 第一个参数为显示位置，第二个参数为显示的值
    axs[2, 0].set_xticks([i + 0.6 / 2 for i in x_position], x_position_fmt)
    # axs[2, 0].set_yticks([i for i in y_position],y_position_fmt)
    axs[2, 0].set_title('(c).', x=-0.05, y=-0.1)
    axs[2, 0].set_ylabel('Model Performance')
    axs[2, 0].grid(True,linestyle='--', linewidth=0.5)

    # print(boxdata[3])
    bp = axs[2, 1].boxplot(boxdata[3], patch_artist=True, widths=0.13, positions=(1, 1.2, 1.4, 1.6),
                           # 多图层叠加时，控制画图层的先后：zorder=1，数字越大越后画
                           showmeans=False, medianprops={'lw': 1, 'color': 'black'}, zorder=1,
                           # 设置异常点属性，如点的形状、填充色和点的大小
                           flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 3})
    # for patch, color in zip(bp['boxes'], color_list):
    #     patch.set_facecolor(color)
    for patch, color in zip(bp['boxes'], [color_list[13], color_list[12], color_list[0], color_list[2]]):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # 替换横坐标x的值
    x_position = [1]
    x_position_fmt = ["R$2$"]
    # 第一个参数为显示位置，第二个参数为显示的值
    axs[2, 1].set_xticks([i + 0.6 / 2 for i in x_position], x_position_fmt)
    axs[2, 1].set_title('(d).', x=-0.05, y=-0.1)
    axs[2, 1].grid(True,linestyle='--', linewidth=0.5)

    # 在第0行子图绘制图例：
    # 创建一个新的轴对象用于放置图例，位于图表右侧
    ax_legend = figs.add_axes([0.12, 0.85, 0.85, 0.05])  # [left, bottom, width, height]
    ax_legend.axis('off')  # 隐藏轴线
    # colors = color_list
    colors = [color_list[13], color_list[12], color_list[0], color_list[2]]
    labels = name_models
    handles = [patches.Patch(color=color, label=label,edgecolor='0') for color, label in zip(colors, labels)]
    legend_font = {"size": 10, 'family': "Arial"}
    # loc="lower left"指ax_legend的起始点(0.1,0.87)位于图例中的左下
    ax_legend.legend(handles=handles, ncol=4, frameon=False
                     # ,prop=legend_font
                     , mode='expand', loc="lower left")
    figs.tight_layout()
    # # bbox_to_anchor指定legend的(x,y,width,length),loc指bbox的(x,y)点相对图例的位置，prop可以传字体
    # # fig.legend(handles=handles, ncol=2, loc="outside upper center", frameon=False,
    # #                 prop=legend_font, mode='expand') #, bbox_to_anchor=(0., 1.02, 0.95, .105)
    # #                 plt.savefig("E://study//02_article//submitt//01_phenology//01_MLModel//picture//20241212//"+"Figure2_{:}".format(fignum)+"_scatter_{:}1.tiff".format(name_model), dpi=1200)
    plt.savefig(r"E:\study\02_article\submitt\01_phenology\01_MLModel\picture\20250711\\" + name_fig
                , dpi=1200)
    plt.show()
