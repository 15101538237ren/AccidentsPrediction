# -*- coding: utf-8 -*-
import sys,os,requests,urllib,json,math,pickle,datetime,simplejson,decimal
from  models import *
from import_data import unicode_csv_reader
import numpy as np #导入Numpy
from AccidentsPrediction.settings import BASE_DIR
from scipy.stats import spearmanr, pearsonr
import plotly.plotly
from util import second_format, date_format, get_work_day_data_for_train,get_holiday_and_tiaoxiu_data_for_train,\
    holiday_7_list, holiday_3_list,tiaoxiu_list,holiday_3_list_flatten,holiday_7_list_flatten,LAST_WEEK_KEY,YESTERDAY_KEY,LAST_N_HOUR_KEY,LABEL_KEY,get_conv_kernal_crespond_data
import pickle
reload(sys)
sys.setdefaultencoding('utf8')

OUT_FIGURES_DIR = os.path.join(BASE_DIR, 'static','figures')

work_day_dt_start = datetime.datetime.strptime("2016-01-12 00:00:00", second_format)
holiday_dt_start = datetime.datetime.strptime("2016-01-01 00:00:00", second_format)

LAST_N_HOUR_STR = "last_n_hour:"
YESTERDAY_STR = "yesterday:"
LAST_WEEK_STR = "last week:"
SPATIAL_LAYER = "outerlayer:"

# 计算皮尔逊相关系数r(d)
def calc_C_d_by_pearson_correlation(CpG_pairs):
    sum_pre = 0.0
    sum_post = 0.0
    for pair in CpG_pairs:
        sum_pre = sum_pre + pair[0]
        sum_post = sum_post + pair[1]
    length = len(CpG_pairs)
    mean_1 = sum_pre / length
    mean_2 = sum_post / length

    sum_up = 0.0
    sum_down_left = 0.0
    sum_down_right = 0.0
    for pair in CpG_pairs:
        X_i = pair[0]
        Y_i = pair[1]
        sum_up = sum_up + (X_i - mean_1) * (Y_i - mean_2)
        sum_down_left = sum_down_left + (X_i - mean_1) * (X_i - mean_1)
        sum_down_right = sum_down_right + (Y_i - mean_2) * (Y_i - mean_2)
    sum_down = math.sqrt(sum_down_left * sum_down_right)
    if sum_down == 0:
        return -1
    r_d = sum_up / sum_down
    return r_d

#获取data相关的空间数据
def get_spatial_related_data(data_shape,n_lng,n_lat,data_tmp, w, b, conv_param):
    out_conv= get_conv_kernal_crespond_data(np.array(data_tmp).reshape(data_shape), w, b, conv_param)
    out_shape = out_conv.shape
    zero_of_mat = np.zeros(out_shape)
    data_tmp_to_apd = []
    for w_i in range(n_lng):
        for h_j in range(n_lat):
            wh_id = w_i * n_lat + h_j
            it_to_rtn = 0
            item_of_now_conv = out_conv[0,0,h_j, w_i,:]
            if (item_of_now_conv > zero_of_mat).any():
                it_to_rtn = 1
            data_tmp_to_apd.append(it_to_rtn)
    return data_tmp_to_apd
#计算式将相关性
def calc_time_correlation(dt_list,count_limit,holiday_3_acc,holiday_7_acc,tiaoxiu_acc,work_day_acc,n,n_d,n_w,x_shape,n_lng,n_lat,w,b,conv_param):
    dt_cnt = 0
    #计算时间相关性的字典
    data_for_time_correlation_dict = {}
    keys_for_time_correlation = []
    for it in xrange(n, 0, -1):
        itt = it * -1
        keys_for_time_correlation.append(LAST_N_HOUR_STR+" "+str(itt))

    for it in xrange(1, n_d + 1):
        keys_for_time_correlation.append(YESTERDAY_STR+" "+str(it))
        itt = it * -1
        keys_for_time_correlation.append(YESTERDAY_STR+" "+str(itt))

    keys_for_time_correlation.append(YESTERDAY_STR+" 0")

    for it in xrange(1, n_w + 1):
        keys_for_time_correlation.append(LAST_WEEK_STR+" "+str(it))
        itt = it * -1
        keys_for_time_correlation.append(LAST_WEEK_STR+" "+str(itt))
    keys_for_time_correlation.append(LAST_WEEK_STR+" 0")

    for key in keys_for_time_correlation:
        data_for_time_correlation_dict[key] = []
    for dt_now in dt_list:
        dt_str = dt_now.strftime(second_format)
        dt_str_date = dt_now.strftime(date_format)

        if (dt_str_date in holiday_7_list[0]) or (dt_str_date in holiday_3_list[0]) or (dt_str_date == tiaoxiu_list[0]):
            continue

        dt_cnt += 1
        if dt_cnt < count_limit or count_limit < 0:

            special = 1

            if dt_str_date in holiday_3_list_flatten:
                data_now = holiday_3_acc[dt_str]

            elif dt_str_date in holiday_7_list_flatten:
                data_now = holiday_7_acc[dt_str]

            elif dt_str_date in tiaoxiu_list:
                data_now = tiaoxiu_acc[dt_str]

            else:
                data_now = work_day_acc[dt_str]
                special = 0

            data_last_week = data_now[LAST_WEEK_KEY] #n_w
            data_yesterday = data_now[YESTERDAY_KEY] #n_d
            data_last_hours = data_now[LAST_N_HOUR_KEY] #n
            data_labels = data_now[LABEL_KEY]

            data_nows = [int(item) for item in data_labels.content.split(",")]

            for it in range(n):
                itt = -n + it
                data_tmp = [int(item) for item in data_last_hours[it].content.split(",")]
                data_tmp = get_spatial_related_data(x_shape,n_lng,n_lat,data_tmp,w,b,conv_param)

                data_to_append = [[data_nows[item],data_tmp[item]] for item in range(len(data_tmp))]
                #append 一个list,之后分别计算每个list(每个对应昨天itt小时的的网格点中的所有数据)
                data_for_time_correlation_dict[LAST_N_HOUR_STR+" "+str(itt)].extend(data_to_append)

            data_ytmp = [int(item) for item in data_yesterday[n_d].content.split(",")]
            data_ytmp = get_spatial_related_data(x_shape,n_lng,n_lat,data_ytmp,w,b,conv_param)

            data_yto_append = [[data_nows[item],data_ytmp[item]] for item in range(len(data_ytmp))]
            data_for_time_correlation_dict[YESTERDAY_STR+" 0"].extend(data_yto_append)

            for it in range(n_d):
                key1 = -1 * (it + 1)
                data_tk1 = [int(item) for item in data_yesterday[it].content.split(",")]
                data_tk1 = get_spatial_related_data(x_shape,n_lng,n_lat,data_tk1,w,b,conv_param)

                data_yk1to_append = [[data_nows[item],data_tk1[item]] for item in range(len(data_tk1))]
                data_for_time_correlation_dict[YESTERDAY_STR + " " + str(key1)].extend(data_yk1to_append)

                key2 = -1 * key1
                idx_of_key2 = n_d + key2
                data_tk2 = [int(item) for item in data_yesterday[idx_of_key2].content.split(",")]
                data_tk2 = get_spatial_related_data(x_shape,n_lng,n_lat,data_tk2,w,b,conv_param)

                data_yk2to_append = [[data_nows[item], data_tk2[item]] for item in range(len(data_tk2))]
                data_for_time_correlation_dict[YESTERDAY_STR + " " + str(key2)].extend(data_yk2to_append)

            data_wtmp = [int(item) for item in data_last_week[n_w].content.split(",")]
            data_wto_append = [[data_nows[item],data_wtmp[item]] for item in range(len(data_wtmp))]
            data_for_time_correlation_dict[LAST_WEEK_STR + " 0"].extend(data_wto_append)

            for it in range(n_w):
                key1 = -1 * (it + 1)
                data_tk1 = [int(item) for item in data_last_week[it].content.split(",")]
                data_tk1 = get_spatial_related_data(x_shape,n_lng,n_lat,data_tk1,w,b,conv_param)

                data_wk1to_append = [[data_nows[item], data_tk1[item]] for item in range(len(data_tk1))]
                data_for_time_correlation_dict[LAST_WEEK_STR + " " + str(key1)].extend(data_wk1to_append)

                key2 = -1 * key1
                idx_of_key2 = n_w + key2
                data_tk2 = [int(item) for item in data_last_week[idx_of_key2].content.split(",")]
                data_tk2 = get_spatial_related_data(x_shape,n_lng,n_lat,data_tk2,w,b,conv_param)

                data_wk2to_append = [[data_nows[item], data_tk2[item]] for item in range(len(data_tk2))]
                data_for_time_correlation_dict[LAST_WEEK_STR + " " + str(key2)].extend(data_wk2to_append)
        print "finish %s" % dt_str
    data_for_time_correlation_tmp = {}

    for key in keys_for_time_correlation:
        # data_for_time_correlation_tmp[key] = [calc_C_d_by_pearson_correlation(item) for item in data_for_time_correlation_dict[key]]
        data_for_time_correlation_tmp[key] = calc_C_d_by_pearson_correlation(data_for_time_correlation_dict[key])
    print "finish all"

    for key,val in data_for_time_correlation_tmp.items():
        print "%s, %.4f" % (key,val)
    return data_for_time_correlation_tmp
def calc_spatial_correlation(dt_list,spatial_extent,count_limit,holiday_3_acc,holiday_7_acc,tiaoxiu_acc,work_day_acc,n,n_d,n_w,x_shape,n_lng,n_lat,b):
    #计算空间相关性的字典
    data_for_spatial_correlation_dict = {}

    for it in spatial_extent:
        data_for_spatial_correlation_dict[SPATIAL_LAYER+" "+str(it)] = [[], []]

    dt_cnt = 0
    for dt_now in dt_list:
        dt_str = dt_now.strftime(second_format)
        dt_str_date = dt_now.strftime(date_format)

        if (dt_str_date in holiday_7_list[0]) or (dt_str_date in holiday_3_list[0]) or (dt_str_date == tiaoxiu_list[0]):
            continue

        dt_cnt += 1
        if dt_cnt < count_limit or count_limit < 0:

            special = 1

            if dt_str_date in holiday_3_list_flatten:
                data_now = holiday_3_acc[dt_str]

            elif dt_str_date in holiday_7_list_flatten:
                data_now = holiday_7_acc[dt_str]

            elif dt_str_date in tiaoxiu_list:
                data_now = tiaoxiu_acc[dt_str]

            else:
                data_now = work_day_acc[dt_str]
                special = 0

            data_labels = data_now[LABEL_KEY]

            data_nows = [int(item) for item in data_labels.content.split(",")]
            for sp in spatial_extent:
                hw_sp = ww_sp = 1 + sp * 2
                w_shape_sp = (1, 1, hw_sp, ww_sp) #f,c,hw,ww
                w_sp = np.array([1.0 for i in range(hw_sp * ww_sp)]).reshape(w_shape_sp)
                conv_param_sp = {'stride': 1, 'pad': sp}

                out_conv= get_conv_kernal_crespond_data(np.array(data_nows).reshape(x_shape), w_sp, b, conv_param_sp)

                data_tmp_to_apd = []
                center_axis = (hw_sp * ww_sp + 1 ) / 2
                len_else = float(hw_sp * ww_sp - 1)
                for w_i in range(n_lng):
                    for h_j in range(n_lat):
                        tmp_counter = 0.0
                        item_of_now_conv = out_conv[0,0,h_j, w_i,:]
                        for idx, item in enumerate(item_of_now_conv):
                            if idx != center_axis - 1:
                                tmp_counter += item
                        it_to_rtn = tmp_counter/len_else
                        data_tmp_to_apd.append(it_to_rtn)
                data_for_spatial_correlation_dict[SPATIAL_LAYER+" "+str(sp)][0].extend([float(data_nows[itm]) for itm in range(len(data_nows))])
                data_for_spatial_correlation_dict[SPATIAL_LAYER+" "+str(sp)][1].extend([data_tmp_to_apd[itm] for itm in range(len(data_tmp_to_apd))])
        print "finish %s" % dt_str
    spatial_pearson_corr_dict = {}
    spatial_spearman_corr_dict = {}
    for key, val in data_for_spatial_correlation_dict.items():
        pearson_correlation_of_spatial = pearsonr(val[0],val[1])
        spatial_pearson_corr_dict[key] = pearson_correlation_of_spatial
        print "Pearson: %s, r: %.4f, p-val:%.8f" % (key, pearson_correlation_of_spatial[0],pearson_correlation_of_spatial[1])
        spearman_correlation_of_spatial = spearmanr(val[0],val[1])
        spatial_spearman_corr_dict[key] = spearman_correlation_of_spatial
        print "Spearman: %s, r: %.4f, p-val:%.8f" % (key, spearman_correlation_of_spatial[0],spearman_correlation_of_spatial[1])
    return [spatial_pearson_corr_dict,spatial_spearman_corr_dict]
#获取所有用于相关性分析的数据
def get_all_data_for_analysis(dt_start, dt_end, time_interval, n, n_d, n_w, **params):
    #经度网格数量
    n_lng = params["n_lng"]
    #纬度网格数量
    n_lat = params["n_lat"]
    #空间间隔长度
    d_len = params["d_len"]

    x_shape = (1, 1, n_lat, n_lng) #n,c,h,w
    hw = ww = 3
    w_shape = (1, 1, hw, ww) #f,c,hw,ww
    w = np.array([1.0 for i in range(hw * ww)]).reshape(w_shape)
    b = np.array([0])
    conv_param = {'stride': 1, 'pad': hw - 2}

    work_day_acc = get_work_day_data_for_train(work_day_dt_start, dt_end,time_interval, d_len, n, n_d, n_w)
    tiaoxiu_acc, holiday_3_acc, holiday_7_acc = get_holiday_and_tiaoxiu_data_for_train(holiday_dt_start, dt_end,time_interval, d_len, n, n_d, n_w)

    #获取区域功能矩阵
    region_functions = Region_Function.objects.filter(spatial_interval=d_len).order_by("region_type")
    region_matrix_dict = {}

    for r_f in region_functions:
        region_cnt_matrix = [int(item) for item in r_f.region_cnt_matrix.split(",")]
        region_matrix_dict[str(r_f.region_type)] = region_cnt_matrix

    dt_list = []
    dt_now = dt_start

    while dt_now < dt_end:
        dt_list.append(dt_now)
        dt_now += datetime.timedelta(minutes= time_interval)

    dt_cnt = 0
    count_limit = -1

    #空间外圈的圈数
    spatial_extent = [1, 2, 3]



    #计算时间相关性
    # time_correlation = calc_time_correlation(dt_list,count_limit,holiday_3_acc,holiday_7_acc,tiaoxiu_acc,work_day_acc,n,n_d,n_w,x_shape,n_lng,n_lat,w,b,conv_param)

    #计算空间相关性
    # [spatial_pearson_corr_dict,spatial_spearman_corr_dict] = calc_spatial_correlation(dt_list,spatial_extent,count_limit,holiday_3_acc,holiday_7_acc,tiaoxiu_acc,work_day_acc,n,n_d,n_w,x_shape,n_lng,n_lat,b)

    #计算天气严重程度相关性的字典
    weather_accidents={}

    #计算pm2.5严重程度相关性的字典
    air_accidents = {}

    #计算时间段相关性的字典
    time_seg_accidents = {}
    for dt_now in dt_list:
        dt_str = dt_now.strftime(second_format)
        dt_str_date = dt_now.strftime(date_format)


        if (dt_str_date in holiday_7_list[0]) or (dt_str_date in holiday_3_list[0]) or (dt_str_date == tiaoxiu_list[0]):
            continue

        if dt_str_date not in weather_accidents.keys():
            weather_accidents[dt_str_date] = {}
            weather_accidents[dt_str_date]["cnt"] = 0
        if dt_str_date not in air_accidents.keys():
            air_accidents[dt_str_date] = {}
            air_accidents[dt_str_date]["cnt"] = 0

        for ts in range(7):
            key_t = dt_str_date+" "+str(ts)
            if key_t not in time_seg_accidents.keys():
                time_seg_accidents[key_t] = {}
                time_seg_accidents[key_t]["cnt"] = 0
        
        dt_cnt += 1
        if dt_cnt < count_limit or count_limit < 0:

            special = 1

            if dt_str_date in holiday_3_list_flatten:
                data_now = holiday_3_acc[dt_str]

            elif dt_str_date in holiday_7_list_flatten:
                data_now = holiday_7_acc[dt_str]

            elif dt_str_date in tiaoxiu_list:
                data_now = tiaoxiu_acc[dt_str]

            else:
                data_now = work_day_acc[dt_str]
                special = 0

            data_labels = data_now[LABEL_KEY]
            weather_severity = int(round(float(data_now[LABEL_KEY].weather_severity)))
            air_pm25 = int(round(float(data_now[LABEL_KEY].pm25)/100.0)) * 100
            time_segment = data_now[LABEL_KEY].time_segment

            data_sum = np.array([int(item) for item in data_labels.content.split(",")]).sum()
            if "weather" not in weather_accidents[dt_str_date].keys():
                weather_accidents[dt_str_date]["weather"] = weather_severity
            weather_accidents[dt_str_date]["cnt"] += data_sum

            if "pm25" not in air_accidents[dt_str_date].keys():
                air_accidents[dt_str_date]["pm25"] = air_pm25
            air_accidents[dt_str_date]["cnt"] += data_sum
            
            if "time_segment" not in time_seg_accidents[dt_str_date+" "+str(time_segment)].keys():
                time_seg_accidents[dt_str_date+" "+str(time_segment)]["time_segment"] = time_segment
            time_seg_accidents[dt_str_date+" "+str(time_segment)]["cnt"] += data_sum

    accidents_of_weather_stat = {}
    for k,v in weather_accidents.items():
        if v["weather"] not in accidents_of_weather_stat.keys():
            accidents_of_weather_stat[v["weather"]] = [float(v["cnt"])]
        else:
            accidents_of_weather_stat[v["weather"]].append(float(v["cnt"]))
    accidents_of_weather = [[],[]]

    for k in range(5):
        mean_accidents_of_fk = np.array(accidents_of_weather_stat[k]).mean()
        print "%d: %.3f" % (k, mean_accidents_of_fk)
        accidents_of_weather[0].append(k)
        accidents_of_weather[1].append(mean_accidents_of_fk)
    print "weather Pearson Corr: %.4f, p-val: %.8f" % (pearsonr(accidents_of_weather[0], accidents_of_weather[1]))
    print "weather Spearman Corr: %.4f, p-val: %.8f" % (spearmanr(accidents_of_weather[0], accidents_of_weather[1]))

    accidents_of_air_stat = {}
    for k, v in air_accidents.items():
        if v["pm25"] not in accidents_of_air_stat.keys():
            accidents_of_air_stat[v["pm25"]] = [float(v["cnt"])]
        else:
            accidents_of_air_stat[v["pm25"]].append(float(v["cnt"]))
    accidents_of_air = [[], []]

    for k in xrange(0, 500, 100):
        mean_accidents_of_fk = np.array(accidents_of_air_stat[k]).mean()
        print "%d: %.3f" % (k, mean_accidents_of_fk)
        accidents_of_air[0].append(k)
        accidents_of_air[1].append(mean_accidents_of_fk)
    print "pm25 Pearson Corr: %.4f, p-val: %.8f" % (pearsonr(accidents_of_air[0], accidents_of_air[1]))
    print "pm25 Spearman Corr: %.4f, p-val: %.8f" % (spearmanr(accidents_of_air[0], accidents_of_air[1]))


    accidents_of_time_seg_stat = {}
    for k, v in time_seg_accidents.items():
        time_segment = int(k.split(" ")[1])
        if time_segment not in accidents_of_time_seg_stat.keys():
            accidents_of_time_seg_stat[time_segment] = [float(v["cnt"])]
        else:
            accidents_of_time_seg_stat[time_segment].append(float(v["cnt"]))
    accidents_of_time_seg = [[], []]

    for k in range(7):
        mean_accidents_of_fk = np.array(accidents_of_time_seg_stat[k]).mean()
        print "%d: %.3f" % (k, mean_accidents_of_fk)
    #     accidents_of_air[0].append(k)
    #     accidents_of_air[1].append(mean_accidents_of_fk)
    # print "Time Segment Pearson Corr: %.4f, p-val: %.8f" % (pearsonr(accidents_of_air[0], accidents_of_air[1]))
    # print "Time Segment Corr: %.4f, p-val: %.8f" % (spearmanr(accidents_of_air[0], accidents_of_air[1]))