# -*- coding: utf-8 -*-
import sys,os,requests,urllib,json,math,pickle,datetime,random
from  models import *
from import_data import unicode_csv_reader
import numpy as np #导入Numpy
import pickle
import keras
from keras.models import Model,load_model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.grid_search import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
reload(sys)
sys.setdefaultencoding('utf8')
## 六环
# min_lat = 39.696203
# max_lat = 40.181729
# min_lng = 116.099649
# max_lng = 116.718542

## 五环
min_lat = 39.764427
max_lat = 40.028983
min_lng = 116.214834
max_lng = 116.554975
x_pi = math.pi * 3000.0 / 180.0

DAWN = 0
MORNING_RUSH = 1
MORNING_WORKING = 2
NOON = 3
AFTERNOON_WORK = 4
AFTERNOON_RUSH = 5
NIGHT = 6
second_format = "%Y-%m-%d %H:%M:%S"
date_format = "%Y-%m-%d"
minute_format = "%Y-%m-%d %H:%M"
#节假日list: 2016-1-1 ~ 2017-2-28
holiday_str_list = ["2016-01-01","2016-01-02","2016-01-03","2016-02-07","2016-02-08","2016-02-09","2016-02-10","2016-02-11","2016-02-12","2016-02-13","2016-04-02","2016-04-03","2016-04-04","2016-04-30","2016-05-01","2016-05-02","2016-06-09","2016-06-10","2016-06-11","2016-09-15","2016-09-16","2016-09-17","2016-10-01","2016-10-02","2016-10-03","2016-10-04","2016-10-05","2016-10-06","2016-10-07","2016-12-31","2017-01-01","2017-01-02","2017-01-27","2017-01-28","2017-01-29","2017-01-30","2017-01-31","2017-02-01","2017-02-02"]
holiday_3_list_flatten = ["2016-01-01","2016-01-02","2016-01-03","2016-04-02","2016-04-03","2016-04-04","2016-04-30","2016-05-01","2016-05-02","2016-06-09","2016-06-10","2016-06-11","2016-09-15","2016-09-16","2016-09-17","2016-12-31","2017-01-01","2017-01-02"]

holiday_3_list = [["2016-01-01","2016-01-02","2016-01-03"],["2016-04-02","2016-04-03","2016-04-04"], ["2016-04-30","2016-05-01","2016-05-02"], ["2016-06-09","2016-06-10","2016-06-11"], ["2016-09-15","2016-09-16","2016-09-17"], ["2016-12-31","2017-01-01","2017-01-02"]]
holiday_7_list = [["2016-02-07","2016-02-08","2016-02-09","2016-02-10","2016-02-11","2016-02-12","2016-02-13"], ["2016-10-01","2016-10-02","2016-10-03","2016-10-04","2016-10-05","2016-10-06","2016-10-07"], ["2017-01-27","2017-01-28","2017-01-29","2017-01-30","2017-01-31","2017-02-01","2017-02-02"]]
holiday_7_list_flatten = ["2016-02-07","2016-02-08","2016-02-09","2016-02-10","2016-02-11","2016-02-12","2016-02-13","2016-10-01","2016-10-02","2016-10-03","2016-10-04","2016-10-05","2016-10-06","2016-10-07","2017-01-27","2017-01-28","2017-01-29","2017-01-30","2017-01-31","2017-02-01","2017-02-02"]

tiaoxiu_list = ["2016-02-06","2016-02-14","2016-06-12","2016-09-18","2016-10-08","2016-10-09","2017-01-22","2017-02-04"]
work_day_bounds = [["2016-01-04","2016-02-05"],["2016-02-15","2016-04-01"],["2016-04-05","2016-04-29"],["2016-05-03","2016-06-08"],["2016-06-13","2016-09-14"],["2016-09-19","2016-09-30"],["2016-10-10","2016-12-30"],["2017-01-03","2017-01-21"],["2017-01-23","2017-01-26"],["2017-02-03","2017-02-03"],["2017-02-05","2017-02-28"]]
hour_0 = " 00:00:00"
end_of_day = " 23:59:59"
TRAIN_DATA_KEY = "TRAIN_DATA"
LABEL_KEY = "LABEL"
LAST_N_HOUR_KEY = "LAST_N"
YESTERDAY_KEY = "YEST_ND"
LAST_WEEK_KEY = "LAST_WEEK"

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  N,C,H,W = x.shape
  F,_,HH,WW = w.shape
  S = conv_param['stride']
  P = conv_param['pad']
  Ho = 1 + (H + 2 * P - HH) / S
  Wo = 1 + (W + 2 * P - WW) / S
  x_pad = np.zeros((N,C,H+2*P,W+2*P))
  x_pad[:,:,P:P+H,P:P+W]=x
  #x_pad = np.pad(x, ((0,), (0,), (P,), (P,)), 'constant')
  out = np.zeros((N,F,Ho,Wo))

  for f in xrange(F):
    for i in xrange(Ho):
      for j in xrange(Wo):
        # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
        out[:,f,i,j] = np.sum(x_pad[:, :, i*S : i*S+HH, j*S : j*S+WW] * w[f, :, :, :], axis=(1, 2, 3))

    out[:,f,:,:]+=b[f]
  cache = (x, w, b, conv_param)
  return out, cache


#获取数据库中工作日时段的所有事故相关的数据
def get_work_day_data(data_bounds,time_interval, spatial_interval):
    work_day_accidents = {}
    work_day_accidents_arr = []
    for day_start, day_end in data_bounds:
        dt_start = datetime.datetime.strptime(day_start + hour_0, second_format)
        dt_end = datetime.datetime.strptime(day_end + end_of_day, second_format)

        # dt_now = dt_start
        # dt_now_end_of_day = datetime.datetime.strptime(day_start + end_of_day, "%Y-%m-%d %H:%M:%S")
        # datetime_list = []
        # time_delta = datetime.timedelta(hours= 24)
        # while dt_now < dt_end:
        #     datetime_list.append([dt_now, dt_now_end_of_day])
        #     dt_now += time_delta
        #     dt_now_end_of_day += time_delta
        # for dt_s, dt_e in datetime_list:
        accidents = Accidents_Array.objects.filter(time_interval= time_interval, spatial_interval= spatial_interval, create_time__range=[ dt_start, dt_end]).order_by("create_time")
        for accident in accidents:
            time_str = accident.create_time.strftime(second_format)
            work_day_accidents[time_str] = accident
            work_day_accidents_arr.append(accident)
    return work_day_accidents, work_day_accidents_arr
#获得所有时间对应的数据和对应的数组中的idx
def get_all_data_in_index(datetime_start, datetime_end,time_interval, spatial_interval):
    work_day_accidents = {}
    accidents = Accidents_Array.objects.filter(time_interval= time_interval, spatial_interval= spatial_interval, create_time__range=[ datetime_start, datetime_end]).order_by("create_time")

    for accident in accidents:
        time_str = accident.create_time.strftime(second_format)
        work_day_accidents[time_str] = accident
    return work_day_accidents
#获取工作日训练数据
#n:过去的几个小时
#n_d:昨天当前时刻前后的n_d个小时
#n_w:上周对应星期几的对应时间前后的n_w个小时
def get_work_day_data_for_train(time_interval, spatial_interval, n, n_d, n_w):
    work_day_accidents, work_day_accidents_arr = get_work_day_data(work_day_bounds,time_interval, spatial_interval)

    len_arr = len(work_day_accidents)
    print "len arr %d" % len_arr
    hour_delta = datetime.timedelta(hours=1)
    work_day_accidents_for_train = {}
    t_time_interval = int(60 / time_interval)
    #从1月12日开始生成训练数据
    for i in range(8 * 24 * t_time_interval, len(work_day_accidents_arr)):
        time_now = work_day_accidents_arr[i].create_time
        time_now_str = time_now.strftime(second_format)
        now_week_day = time_now.weekday()
        print "workday: %s" % time_now_str
        work_day_accidents_for_train[time_now_str] = {}

        #当前时刻的事故数据
        data_now = work_day_accidents_arr[i]
        work_day_accidents_for_train[time_now_str][LABEL_KEY] = data_now

        #上n个time_interval的事故数据
        time_minus_n = time_now - datetime.timedelta(minutes= time_interval * n)
        ts = time_minus_n
        ts_str = ts.strftime(second_format)

        #如果恰巧处在日期不连续的时间点,则直接取上几个数据
        if ts_str not in work_day_accidents.keys():
            last_n = i - n
            work_day_accidents_for_train[time_now_str][LAST_N_HOUR_KEY] = work_day_accidents_arr[last_n : i]
            # print "len_n: %d, len_ids: %d" % (n, len(work_day_accidents_arr[last_n : i]))
        else:
            #否则按照日期-n个time_interval来查找数据
            last_n_arr = []
            while ts < time_now:
                last_n_arr.append(work_day_accidents[ts_str])
                ts += datetime.timedelta(minutes= time_interval)
                ts_str = ts.strftime(second_format)
            work_day_accidents_for_train[time_now_str][LAST_N_HOUR_KEY] = last_n_arr
            # print "len_arr: %d" % len(last_n_arr)
        #print "Got last %d data for %s !" % (n, time_now_str)

        # n_d昨天相同时间前后n_d个time_interval的数据
        t_last_day_n_d_pre = time_now - datetime.timedelta(minutes= (24 * 60 + n_d * time_interval))
        t_last_day_n_d_post = time_now - datetime.timedelta(minutes = (24 * 60 - n_d * time_interval))

        t_pe = t_last_day_n_d_pre
        t_pe_str = t_pe.strftime(second_format)

        t_po = t_last_day_n_d_post
        t_po_str = t_po.strftime(second_format)

        #时间交界处
        if (t_pe_str not in work_day_accidents.keys()) or (t_po_str not in work_day_accidents.keys()):
            yest_nd_pre = i - 24 * t_time_interval - n_d
            yest_nd_post = i - 24 * t_time_interval + n_d
            work_day_accidents_for_train[time_now_str][YESTERDAY_KEY] = work_day_accidents_arr[yest_nd_pre : yest_nd_post + 1]
            #print "len_nd: %d, len_ids: %d" % (n_d, len(work_day_accidents_arr[yest_nd_pre : yest_nd_post + 1]))
        else:
            #否则按照昨天的-nd:+nd个time_interval来查找数据
            yest_nd_arr = []
            while t_pe <= t_po:
                yest_nd_arr.append(work_day_accidents[t_pe_str])
                t_pe += datetime.timedelta(minutes= time_interval)
                t_pe_str = t_pe.strftime(second_format)
            work_day_accidents_for_train[time_now_str][YESTERDAY_KEY] = yest_nd_arr
            #print "len_arr: %d" % len(yest_nd_arr)

        # n_w上周相同时间前后n_w小时的数据
        t_last_week_n_w_pre = time_now - datetime.timedelta(minutes= (7 * 24 * 60 + n_w * time_interval))
        t_last_week_n_w_post = time_now - datetime.timedelta(minutes= (7 * 24 * 60 - n_w * time_interval))

        tw_pe = t_last_week_n_w_pre
        tw_pe_str = tw_pe.strftime(second_format)

        tw_po = t_last_week_n_w_post
        tw_po_str = tw_po.strftime(second_format)
        last_week_nw_arr = []
        #时间交界处
        if (tw_pe_str not in work_day_accidents.keys()) or (tw_po_str not in work_day_accidents.keys()):
            #9是春节的最长休息时间
            for d_l in range(1, 35):
                last_week_time = time_now - datetime.timedelta(hours=d_l * 24)
                last_week_time_str = last_week_time.strftime(second_format)
                last_week_time_weekday = last_week_time.weekday()
                if last_week_time_weekday != now_week_day:
                    continue
                elif last_week_time_str not in work_day_accidents.keys():
                    continue
                else:
                    #星期既相等,时间数据又有
                    lw_t_pre = last_week_time - datetime.timedelta(minutes= n_w * time_interval)
                    lw_t_pre_str = lw_t_pre.strftime(second_format)
                    lw_t_post = last_week_time + datetime.timedelta(minutes= n_w * time_interval)
                    lw_t_post_str = lw_t_post.strftime(second_format)

                    if (lw_t_pre_str not in work_day_accidents.keys()) or (lw_t_post_str not in work_day_accidents.keys()):
                        continue
                    else:
                        while lw_t_pre <= lw_t_post:
                            last_week_nw_arr.append(work_day_accidents[lw_t_pre_str])
                            lw_t_pre += datetime.timedelta(minutes= time_interval)
                            lw_t_pre_str = lw_t_pre.strftime(second_format)
                        work_day_accidents_for_train[time_now_str][LAST_WEEK_KEY] = last_week_nw_arr
                        #print "len_arr_u: %d" % len(last_week_nw_arr)
                        break
        else:
            #否则按照上周的-nd:+nd个time_interval来查找数据
            while tw_pe <= tw_po:
                last_week_nw_arr.append(work_day_accidents[tw_pe_str])
                tw_pe += datetime.timedelta(minutes= time_interval)
                tw_pe_str = tw_pe.strftime(second_format)
            work_day_accidents_for_train[time_now_str][LAST_WEEK_KEY] = last_week_nw_arr
            #print "len_arr_d: %d" % len(last_week_nw_arr)
    return work_day_accidents_for_train
def prepare_lstm_data(out_pickle_file_path, dt_start, dt_end, time_interval, n, n_d, n_w, **params):

    region_type_list = range(1, 13)
    width = params["n_lng"]
    height = params["n_lat"]
    spatial_interval = params["d_len"]

    # n_time_steps
    n_time_steps = n + (n_d + n_w ) *2 + 2

    # 内层每一个样本点的每个时间点对应的数据维度
    data_dim = 10

    #卷积操作相关
    x_shape = (1, 1, height, width) #n,c,h,w
    out_shape = (1, 1, height * width)
    w_shape = (1, 1, 3, 3) #f,c,hw,ww
    w = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5,]).reshape(w_shape)
    b = np.array([0])
    conv_param = {'stride': 1, 'pad': 1}

    work_day_acc = get_work_day_data_for_train(time_interval, spatial_interval, n, n_d, n_w)
    holiday_dt_start = datetime.datetime.strptime("2016-01-01 00:00:00", second_format)
    tiaoxiu_acc, holiday_3_acc, holiday_7_acc = get_holiday_and_tiaoxiu_data_for_train(holiday_dt_start, dt_end,time_interval, spatial_interval, n, n_d, n_w)

    region_functions = Region_Function.objects.filter(spatial_interval=spatial_interval).order_by("region_type")
    region_matrix_dict = {}

    for r_f in region_functions:
        region_cnt_matrix = [int(item) for item in r_f.region_cnt_matrix.split(",")]
        region_matrix_dict[str(r_f.region_type)] = region_cnt_matrix

    zero_workday_data_list = []
    zero_workday_label_list = []
    # zero_workday_function_list = []

    zero_special_data_list = []
    zero_special_label_list = []
    # zero_special_function_list = []

    positive_data_list = []
    positive_label_list = []
    # positive_function_list = []

    dt_list = []
    dt_now = dt_start
    while dt_now < dt_end:
        dt_list.append(dt_now)
        dt_now += datetime.timedelta(minutes= time_interval)

    dt_cnt = 0
    for dt_now in dt_list:
        dt_str = dt_now.strftime(second_format)
        dt_str_date = dt_now.strftime(date_format)
        if (dt_str_date in holiday_7_list[0]) or (dt_str_date in holiday_3_list[0]) or (dt_str_date == tiaoxiu_list[0]):
            continue
        dt_cnt += 1


        if dt_str_date in holiday_3_list_flatten:
            data_now = holiday_3_acc[dt_str]
            special = 1
        elif dt_str_date in holiday_7_list_flatten:
            data_now = holiday_7_acc[dt_str]
            special = 1
        elif dt_str_date in tiaoxiu_list:
            data_now = tiaoxiu_acc[dt_str]
            special = 1
        else:
            data_now = work_day_acc[dt_str]
            special = 0

        data_last_week = data_now[LAST_WEEK_KEY]
        data_yesterday = data_now[YESTERDAY_KEY]
        data_last_hours = data_now[LAST_N_HOUR_KEY]
        data_labels = data_now[LABEL_KEY]

        data_merge = data_last_week + data_yesterday
        data_merge = data_merge + data_last_hours

        len_data_merge = len(data_merge)
        data_shape = (height * width, len_data_merge, data_dim)
        data_for_now = np.zeros(data_shape)

        for idx, data_i in enumerate(data_merge):
            extra_data = [int(data_i.highest_temperature), int(data_i.lowest_temperature), float(data_i.weather_severity), int(data_i.aqi), int(data_i.pm25), int(data_i.is_holiday), int(data_i.is_weekend), int(data_i.time_segment)]
            data_content = np.array([int(item) for item in data_i.content.split(",")]).reshape(x_shape)
            out_conv, _ = conv_forward_naive(data_content, w, b, conv_param)
            out_conv = out_conv.ravel()
            data_for_now[:, idx, 0] = [it for it in range(height * width)]
            data_for_now[:, idx, 1] = out_conv
            data_for_now[:, idx, 2: data_dim] = extra_data

        data_arr = [1 if int(item) > 0 else 0 for item in data_labels.content.split(",")]

        for i_t in range(height * width):
            if data_arr[i_t] == 0:
                if special == 1:
                    zero_special_data_list.append(data_for_now[i_t, :, :])
                    zero_special_label_list.append(data_arr[i_t])
                    # zero_special_function_list.append([region_matrix_dict[str(i)][i_t] for i in region_type_list])
                else:
                    zero_workday_data_list.append(data_for_now[i_t, :, :])
                    zero_workday_label_list.append(data_arr[i_t])
                    # zero_workday_function_list.append([region_matrix_dict[str(i)][i_t] for i in region_type_list])
            else:
                positive_data_list.append(data_for_now[i_t, :, :])
                positive_label_list.append(data_arr[i_t])
                # positive_function_list.append([region_matrix_dict[str(i)][i_t] for i in region_type_list])
        print "finish %s" % dt_str

    cnt_positive = len(positive_label_list)
    cnt_zero_workday = len(zero_workday_label_list)
    cnt_zero_special = len(zero_special_label_list)
    diff = cnt_positive - cnt_zero_special
    if diff > 0:
        print "positive difference"
    total_data_cnt = cnt_zero_special + cnt_zero_workday + cnt_positive
    print "pre total: %d, pos: %d, rate %.3f" % (total_data_cnt, cnt_positive, float(cnt_positive)/float(total_data_cnt))


    train_data_ratio = 0.7
    zero_special_train_len = int(cnt_zero_special * train_data_ratio)
    positive_label_train_len = int(cnt_positive * train_data_ratio)
    zero_workday_train_len = int(cnt_positive * train_data_ratio)



    all_train_data_list = np.array(zero_special_data_list[0:zero_special_train_len] + positive_data_list[0:positive_label_train_len] + zero_workday_data_list[0:zero_workday_train_len])
    all_train_label_list =np.array(zero_special_label_list[0:zero_special_train_len] + positive_label_list[0:positive_label_train_len] + zero_workday_label_list[0:zero_workday_train_len])



    all_val_data_list = np.array(zero_special_data_list[zero_special_train_len:-1] + positive_data_list[positive_label_train_len:-1] + zero_workday_data_list[zero_workday_train_len:cnt_positive])
    all_val_label_list = np.array(zero_special_label_list[zero_special_train_len:-1] + positive_label_list[positive_label_train_len:-1] +zero_workday_label_list[zero_workday_train_len:cnt_positive])

    print "start shuffling 1"
    idx_of_train = np.arange(len(all_train_label_list))
    np.random.shuffle(idx_of_train)
    print "shuffling done 1"

    print "data assigning 1"
    all_train_data_list = all_train_data_list[idx_of_train[:]]
    all_train_label_list = all_train_label_list[idx_of_train[:]]
    print "done data assigning 1"

    print "start shuffling 2"
    idx_of_val = np.arange(len(all_val_label_list))
    np.random.shuffle(idx_of_val)
    print "shuffling done 2"

    print "data assigning 2"
    all_val_data_list = all_val_data_list[idx_of_val[:]]
    all_val_label_list = all_val_label_list[idx_of_val[:]]
    print "done data assigning 1"

    sub_total = cnt_positive * 2 + cnt_zero_special
    print "post total: %d, pos: %d, rate %.3f" % (sub_total, cnt_positive, float(cnt_positive)/float(sub_total))

    out_params = {
                  "out_data_length" : sub_total,
                  "n_time_steps" : n_time_steps,
                  "data_dim" : data_dim,
                  "time_interval" : time_interval,
                  "spatial_interval" : spatial_interval,
                  "n_lng": width,
                  "n_lat": height
              }

    print "start modeling"
    data_dim = out_params["data_dim"]
    timesteps = out_params["n_time_steps"]
    LSTM_dim = 32
    # region_dim = 12
    dense_dim = 64
    # validate_data_ratio = 1.0 - train_data_ratio
    # Input tensor for sequences of 20 timesteps,
    # each containing a 784-dimensional vector
    input_sequences = Input(shape=(timesteps, data_dim))

    lstm1 = LSTM(LSTM_dim, return_sequences=True)(input_sequences)
    lstm2 = LSTM(LSTM_dim, return_sequences=True)(lstm1)
    lstm3 = LSTM(LSTM_dim)(lstm2)

    #region_input = Input(shape=(region_dim, ), name='region_input')
    #concat_layer = keras.layers.concatenate([lstm3, region_input])
    # We stack a deep densely-connected network on top
    x = Dense(dense_dim, activation='relu')(lstm3)
    x = Dense(dense_dim, activation='relu')(x)
    x = Dropout(0.5)(x)
    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=input_sequences, outputs=main_output)
    #model = Model(inputs=[input_sequences, region_input], outputs=[main_output])

    # checkpoint
    checkpointer = ModelCheckpoint(filepath="checkpoint.h5", verbose=1)
    # learning rate adjust dynamic
    # lrate = ReduceLROnPlateau(min_lr=0.00001)
    # early_stoping = EarlyStopping(monitor='val_loss',patience=1)


    model.compile(loss='binary_crossentropy', #loss :rmse?
                  optimizer='rmsprop',# optimizer: adam?
                  metrics=['accuracy'])

    batch_size = 256
    epochs = 1

    print "start fitting"
    model.fit(all_train_data_list, all_train_label_list,
              batch_size=batch_size, epochs= epochs,validation_data=(all_val_data_list,all_val_label_list), callbacks=[checkpointer])

    score, acc = model.evaluate(all_val_data_list, all_val_label_list,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return 0

def pure_lstm(out_pickle_file_path, dt_start, dt_end, time_interval, n, n_d, n_w, **params):

    region_type_list = range(1, 13)
    width = params["n_lng"]
    height = params["n_lat"]
    spatial_interval = params["d_len"]

    # n_time_steps
    n_time_steps = n + (n_d + n_w ) *2 + 2

    # 内层每一个样本点的每个时间点对应的数据维度
    data_dim = 1

    #卷积操作相关
    x_shape = (1, 1, height, width) #n,c,h,w
    out_shape = (1, 1, height * width)
    w_shape = (1, 1, 3, 3) #f,c,hw,ww
    w = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5,]).reshape(w_shape)
    b = np.array([0])
    conv_param = {'stride': 1, 'pad': 1}

    work_day_acc = get_work_day_data_for_train(time_interval, spatial_interval, n, n_d, n_w)
    holiday_dt_start = datetime.datetime.strptime("2016-01-01 00:00:00", second_format)
    tiaoxiu_acc, holiday_3_acc, holiday_7_acc = get_holiday_and_tiaoxiu_data_for_train(holiday_dt_start, dt_end,time_interval, spatial_interval, n, n_d, n_w)

    region_functions = Region_Function.objects.filter(spatial_interval=spatial_interval).order_by("region_type")
    region_matrix_dict = {}

    for r_f in region_functions:
        region_cnt_matrix = [int(item) for item in r_f.region_cnt_matrix.split(",")]
        region_matrix_dict[str(r_f.region_type)] = region_cnt_matrix

    zero_workday_data_list = []
    zero_workday_label_list = []
    zero_workday_function_list = []

    zero_special_data_list = []
    zero_special_label_list = []
    zero_special_function_list = []

    positive_data_list = []
    positive_label_list = []
    positive_function_list = []

    dt_list = []
    dt_now = dt_start
    while dt_now < dt_end:
        dt_list.append(dt_now)
        dt_now += datetime.timedelta(minutes= time_interval)

    dt_cnt = 0
    for dt_now in dt_list:
        dt_str = dt_now.strftime(second_format)
        dt_str_date = dt_now.strftime(date_format)
        if (dt_str_date in holiday_7_list[0]) or (dt_str_date in holiday_3_list[0]) or (dt_str_date == tiaoxiu_list[0]):
            continue
        dt_cnt += 1

        if dt_str_date in holiday_3_list_flatten:
            data_now = holiday_3_acc[dt_str]
            special = 1
        elif dt_str_date in holiday_7_list_flatten:
            data_now = holiday_7_acc[dt_str]
            special = 1
        elif dt_str_date in tiaoxiu_list:
            data_now = tiaoxiu_acc[dt_str]
            special = 1
        else:
            data_now = work_day_acc[dt_str]
            special = 0

        data_last_week = data_now[LAST_WEEK_KEY]
        data_yesterday = data_now[YESTERDAY_KEY]
        data_last_hours = data_now[LAST_N_HOUR_KEY]
        data_labels = data_now[LABEL_KEY]

        data_merge = data_last_week + data_yesterday
        data_merge = data_merge + data_last_hours

        len_data_merge = len(data_merge)
        data_shape = (height * width, len_data_merge, data_dim)
        data_for_now = np.zeros(data_shape)

        for idx, data_i in enumerate(data_merge):
            # extra_data = [float(data_i.weather_severity), int(data_i.pm25), int(data_i.is_holiday), int(data_i.is_weekend), int(data_i.time_segment)]
            data_content = np.array([int(item) for item in data_i.content.split(",")])#.reshape(x_shape)
            # out_conv, _ = conv_forward_naive(data_content, w, b, conv_param)
            # out_conv = out_conv.ravel()
            # data_for_now[:, idx, 0] = [it for it in range(height * width)]
            # data_for_now[:, idx, 1] = out_conv
            data_for_now[:, idx, 0] = data_content
            # data_for_now[:, idx, 1: data_dim] = extra_data
        data_arr = [1 if int(item) > 0 else 0 for item in data_labels.content.split(",")]

        for i_t in range(height * width):
            if data_arr[i_t] == 0:
                if special == 1:
                    zero_special_data_list.append(data_for_now[i_t, :, :])
                    zero_special_label_list.append(data_arr[i_t])
                    zero_special_function_list.append([region_matrix_dict[str(i)][i_t] for i in region_type_list])
                else:
                    zero_workday_data_list.append(data_for_now[i_t, :, :])
                    zero_workday_label_list.append(data_arr[i_t])
                    zero_workday_function_list.append([region_matrix_dict[str(i)][i_t] for i in region_type_list])
            else:
                positive_data_list.append(data_for_now[i_t, :, :])
                positive_label_list.append(data_arr[i_t])
                positive_function_list.append([region_matrix_dict[str(i)][i_t] for i in region_type_list])
        print "finish %s" % dt_str

    cnt_positive = len(positive_label_list)
    cnt_zero_workday = len(zero_workday_label_list)
    cnt_zero_special = len(zero_special_label_list)
    total_data_cnt = cnt_zero_special + cnt_zero_workday + cnt_positive
    print "pre total: %d, pos: %d, rate %.3f" % (total_data_cnt, cnt_positive, float(cnt_positive)/float(total_data_cnt))
    print "cnt_positive: %.3f, cnt_zero_workday: %.3f, cnt_zero_special: %.3f" % (float(cnt_positive)/float(total_data_cnt), float(cnt_zero_workday)/float(total_data_cnt), float(cnt_zero_special)/float(total_data_cnt))


    train_data_ratio = 0.8
    zero_special_train_len = int(cnt_positive * train_data_ratio)
    positive_label_train_len = int(cnt_positive * train_data_ratio)

    print "start shuffling  cnt_zero_workday"
    idx_of_workday = np.arange(cnt_zero_workday)
    np.random.shuffle(idx_of_workday)
    print "shuffling done cnt_zero_workday"

    # sub_zero_workday_data_list = zero_workday_data_list[idx_of_workday[0:cnt_positive]]
    # sub_zero_workday_label_list = zero_workday_label_list[idx_of_workday[0:cnt_positive]]
    # print "finish first ass"
    random_start = random.randint(0,cnt_zero_workday - cnt_positive - 1)
    print "random_start %d" % random_start

    sub_zero_workday_data_list = zero_workday_data_list[random_start: random_start + cnt_positive]

    sub_zero_workday_label_list = zero_workday_label_list[random_start: random_start + cnt_positive]
    sub_zero_function_list = zero_workday_function_list[random_start: random_start + cnt_positive]

    zero_workday_train_len = int(cnt_positive * train_data_ratio)

    all_train_data_list = np.array(zero_special_data_list[0:zero_special_train_len] + positive_data_list[0:positive_label_train_len] + sub_zero_workday_data_list[0:zero_workday_train_len])
    all_train_label_list =np.array(zero_special_label_list[0:zero_special_train_len] + positive_label_list[0:positive_label_train_len] + sub_zero_workday_label_list[0:zero_workday_train_len])
    all_train_function_list = np.array(zero_special_function_list[0:zero_special_train_len] + positive_function_list[0:positive_label_train_len]  + sub_zero_function_list[0:zero_workday_train_len])
    print "finish trainset ass"

    all_val_data_list = np.array(zero_special_data_list[zero_special_train_len: cnt_positive] + positive_data_list[positive_label_train_len:-1] + sub_zero_workday_data_list[zero_workday_train_len:-1])
    all_val_label_list = np.array(zero_special_label_list[zero_special_train_len: cnt_positive] + positive_label_list[positive_label_train_len:-1] +sub_zero_workday_label_list[zero_workday_train_len:-1])
    all_val_function_list = np.array(zero_special_function_list[zero_special_train_len: cnt_positive] + positive_function_list[positive_label_train_len:-1] + sub_zero_function_list[zero_workday_train_len:-1])

    print "finish val ass"

    print "start shuffling 1"
    idx_of_train = np.arange(len(all_train_label_list))
    np.random.shuffle(idx_of_train)
    print "shuffling done 1"

    print "data assigning 1"
    all_train_data_list = all_train_data_list[idx_of_train[:]]
    all_train_label_list = all_train_label_list[idx_of_train[:]]
    all_train_function_list = all_train_function_list[idx_of_train[:]]
    print "done data assigning 1"

    print "start shuffling 2"
    idx_of_val = np.arange(len(all_val_label_list))
    np.random.shuffle(idx_of_val)
    print "shuffling done 2"

    print "data assigning 2"
    all_val_data_list = all_val_data_list[idx_of_val[:]]
    all_val_label_list = all_val_label_list[idx_of_val[:]]
    all_val_function_list = all_val_function_list[idx_of_val[:]]
    print "done data assigning 2"

    sub_total = cnt_positive * 3
    print "post total: %d, pos: %d, rate %.3f" % (sub_total, cnt_positive, float(cnt_positive)/float(sub_total))

    out_params = {
                  "out_data_length" : sub_total,
                  "n_time_steps" : n_time_steps,
                  "data_dim" : data_dim,
                  "time_interval" : time_interval,
                  "spatial_interval" : spatial_interval,
                  "n_lng": width,
                  "n_lat": height
              }

    print "start modeling"
    data_dim = out_params["data_dim"]
    timesteps = out_params["n_time_steps"]

    # LSTM_dim = 64
    region_dim = 12
    # dense_dim = 64
    # validate_data_ratio = 1.0 - train_data_ratio
    # Input tensor for sequences of 20 timesteps,
    # each containing a 784-dimensional vector
    # lstm_layers = [1, 2, 3]
    # lstm_dims = [128, 64, 32, 16]
    # batch_sizes = [64, 128, 256]
    # dropouts = [0.2, 0.4, 0.6]
    # dense_layers = [0,1,2]


    input_sequences = Input(shape=(timesteps, data_dim))

    #2 layer lstm
    # for lstm_dim in lstm_dims:
    lstm_dim = 32

    lstm1 = LSTM(lstm_dim, return_sequences=True)(input_sequences)
    lstm1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(lstm_dim)(lstm1)
    lstm2 = Dropout(0.2)(lstm2)

    # main_output = Dense(1, activation='sigmoid', name='main_output')(lstm2)
    # model = Model(inputs=input_sequences, outputs=main_output)
    #
    # model.compile(loss='binary_crossentropy', #loss :rmse?
    #               optimizer='rmsprop',# optimizer: adam?
    #               metrics=['accuracy'])

    # print "start fitting model 2 layer"

    batch_size = 256
    epochs = 1
    # model.fit(all_train_data_list, all_train_label_list,
    #           batch_size=batch_size, epochs= epochs,validation_data=(all_val_data_list,all_val_label_list), callbacks=[])

    region_input = Input(shape=(region_dim, ), name='region_input')
    concat_layer = keras.layers.concatenate([lstm2, region_input])

    # We stack a deep densely-connected network on top
    x = Dense(128, activation='relu')(concat_layer)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    # model_2_layer = Model(inputs=input_sequences, outputs=main_output)
    model = Model(inputs=[input_sequences, region_input], outputs=[main_output])

    # checkpoint
    checkpointer = ModelCheckpoint(filepath="ckpt_pure_lstm.h5", verbose=1)
    # learning rate adjust dynamic
    lrate = ReduceLROnPlateau(min_lr=0.00001)
    early_stoping = EarlyStopping(monitor='val_loss',patience=1)


    model.compile(loss='binary_crossentropy', #loss :rmse?
                  optimizer='rmsprop',# optimizer: adam?
                  metrics=['accuracy'])

    print "start fitting model with batch_size: %d" % batch_size
    model.fit([all_train_data_list, all_train_function_list], all_train_label_list,
              batch_size=batch_size, epochs= epochs,validation_data=([all_val_data_list, all_val_function_list],all_val_label_list), callbacks=[lrate, early_stoping,checkpointer])


    # model_2_layer.compile(loss='mse', #loss :rmse?
    #               optimizer='adam',# optimizer: adam?
    #               metrics=['accuracy'])
    #
    # print "start fitting model_2_layer with adam"
    # model_2_layer.fit(all_train_data_list, all_train_label_list,
    #           batch_size=batch_size, epochs= epochs,validation_data=(all_val_data_list,all_val_label_list), callbacks=[])
    # x = Dense(128, activation='relu')(lstm)
    # x = Dropout(0.2)(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # main_output_dense = Dense(1, activation='sigmoid', name='main_output')(x)
    # model_2_layer_with_dense = Model(inputs=input_sequences, outputs=main_output_dense)
    # model_2_layer_with_dense.compile(loss='mse', #loss :rmse?
    #               optimizer='rmsprop',# optimizer: adam?
    #               metrics=['accuracy'])
    #
    # print "start fitting model_2_layer with dense"
    # model_2_layer_with_dense.fit(all_train_data_list, all_train_label_list,
    #           batch_size=batch_size, epochs= epochs,validation_data=(all_val_data_list,all_val_label_list), callbacks=[])
    #
    #

    # model = load_model('ckpt_pure_lstm.h5')
    #
    # score, acc = model.evaluate([all_val_data_list, all_val_function_list],all_val_label_list,
    #                             batch_size=batch_size)
    # print('Test score:', score)
    # print('Test accuracy:', acc)

    return 0
#获取调休日和节假日(3天,7天节假日)对应的数据
def get_holiday_and_tiaoxiu_data_for_train(dt_start, dt_end,time_interval, spatial_interval, n, n_d, n_w):

    accidents = get_all_data_in_index(dt_start, dt_end, time_interval,spatial_interval)

    tiaoxiu_accidents_for_train = {}

    tx_dt_list = []
    tx_list_idx = {}
    #先生成调休日的
    for t_i in range(1, len(tiaoxiu_list)):
        date_tx = tiaoxiu_list[t_i]
        dt_tx_st = datetime.datetime.strptime(date_tx + " 00:00:00",second_format)
        dt_tx_ed = datetime.datetime.strptime(date_tx + " 23:59:59",second_format)
        while dt_tx_st < dt_tx_ed:
            tx_dt_list.append(dt_tx_st)
            tx_list_idx[dt_tx_st.strftime(second_format)] = t_i
            dt_tx_st += datetime.timedelta(minutes=time_interval)
    for dt_tiaoxiu in tx_dt_list:
        time_now = dt_tiaoxiu
        time_now_str = time_now.strftime(second_format)
        print "tiaoxiu : %s" % time_now_str
        tiaoxiu_accidents_for_train[time_now_str] = {}

        #当前时刻的事故数据
        data_now = accidents[time_now_str]
        tiaoxiu_accidents_for_train[time_now_str][LABEL_KEY] = data_now

        #上n个time_interval的事故数据
        time_minus_n = time_now - datetime.timedelta(minutes= n * time_interval)
        ts = time_minus_n
        ts_str = ts.strftime(second_format)

        #按照日期-n个time_interval来查找数据
        last_n_arr = []
        while ts < time_now:
            last_n_arr.append(accidents[ts_str])
            ts += datetime.timedelta(minutes= time_interval)
            ts_str = ts.strftime(second_format)
        tiaoxiu_accidents_for_train[time_now_str][LAST_N_HOUR_KEY] = last_n_arr
        #print "tiaoxiu len last_n_arr: %d" % len(last_n_arr)

        # 上一个调休日相同时间前后n_d小时的数据
        last_tiaoxiu_day = datetime.datetime.strptime(tiaoxiu_list[tx_list_idx[time_now_str] - 1] +" "+ time_now_str.split(" ")[1],second_format)

        lt_nd_pre = last_tiaoxiu_day - datetime.timedelta(minutes= n_d * time_interval)
        lt_nd_post = last_tiaoxiu_day + datetime.timedelta(minutes= n_d * time_interval)

        lt_pe_str = lt_nd_pre.strftime(second_format)

        last_tx_nd_arr = []

        while lt_nd_pre <= lt_nd_post:
            last_tx_nd_arr.append(accidents[lt_pe_str])
            lt_nd_pre += datetime.timedelta(minutes= time_interval)
            lt_pe_str = lt_nd_pre.strftime(second_format)
        tiaoxiu_accidents_for_train[time_now_str][YESTERDAY_KEY] = last_tx_nd_arr
        #print "tiaoxiu len last_tx_nd_arr: %d" % len(last_tx_nd_arr)

        last_monday_nw_arr = []
        # 上周一的数据 : 当做上周长周期的数据 , 毕竟调休
        for d_l in range(1 , 14 + 9):
            dt_dl = time_now - datetime.timedelta(hours= 24 * d_l)
            dt_dl_dt_str = dt_dl.strftime(date_format)
            if (dt_dl_dt_str in holiday_str_list) or (dt_dl_dt_str in tiaoxiu_list):
                continue
            elif dt_dl.weekday() != 0:
                #只要周一的
                continue
            else:
                lw_t_pre = dt_dl - datetime.timedelta(minutes= n_w * time_interval)
                lw_t_pre_str = lw_t_pre.strftime(second_format)
                lw_t_post = dt_dl + datetime.timedelta(minutes= n_w * time_interval)

                while lw_t_pre <= lw_t_post:
                    last_monday_nw_arr.append(accidents[lw_t_pre_str])
                    lw_t_pre += datetime.timedelta(minutes= time_interval)
                    lw_t_pre_str = lw_t_pre.strftime(second_format)
                tiaoxiu_accidents_for_train[time_now_str][LAST_WEEK_KEY] = last_monday_nw_arr
                #print "tiaoxiu len last_monday_nw_arr: %d" % len(last_monday_nw_arr)
                break
    holiday_accidents_for_train = [{},{}]
    for idx, h_list in enumerate([holiday_3_list,holiday_7_list]):
        for j_h in range(1,len(h_list)):
            for k_h in range(len(h_list[j_h])):
                date_hl = h_list[j_h][k_h]
                dt_list = []
                dt_hl_st = datetime.datetime.strptime(date_hl + " 00:00:00", second_format)
                dt_hl_ed = datetime.datetime.strptime(date_hl + " 23:59:59", second_format)
                while dt_hl_st < dt_hl_ed:
                    dt_list.append(dt_hl_st)
                    dt_hl_st += datetime.timedelta(minutes=time_interval)
                for dt_now in dt_list:
                    time_now_str = dt_now.strftime(second_format)
                    print "holiday: %s" % time_now_str
                    holiday_accidents_for_train[idx][time_now_str] = {}

                    #当前时刻的事故数据
                    data_now = accidents[time_now_str]
                    holiday_accidents_for_train[idx][time_now_str][LABEL_KEY] = data_now

                    #上n个time_interval的事故数据
                    time_minus_n = dt_now - datetime.timedelta(minutes= n * time_interval)
                    ts = time_minus_n
                    ts_str = ts.strftime(second_format)

                    #按照日期-n个time_interval来查找数据
                    last_n_arr = []
                    while ts < dt_now:
                        last_n_arr.append(accidents[ts_str])
                        ts += datetime.timedelta(minutes= time_interval)
                        ts_str = ts.strftime(second_format)
                    holiday_accidents_for_train[idx][time_now_str][LAST_N_HOUR_KEY] = last_n_arr
                    #print "holiday len last_n_arr: %d" % len(last_n_arr)

                    #昨天的数据对应到长假来说就是上次长假对应编号天的数据
                    lt_nd_pre = lt_nd_post = datetime.datetime.now()
                    lt_pe_str = ""
                    last_holiday = datetime.datetime.strptime(h_list[j_h - 1][k_h] +" "+ time_now_str.split(" ")[1],second_format)

                    while True:
                        lt_nd_pre = last_holiday - datetime.timedelta(minutes= n_d * time_interval)
                        lt_nd_post = last_holiday + datetime.timedelta(minutes= n_d * time_interval)
                        lt_pe_str = lt_nd_pre.strftime(second_format)
                        if lt_pe_str in accidents.keys():
                            break
                        else:
                            last_holiday = datetime.datetime.strptime(h_list[j_h - 1][k_h + 1] +" "+ time_now_str.split(" ")[1],second_format)

                    last_hl_nd_arr = []

                    while lt_nd_pre <= lt_nd_post:
                        last_hl_nd_arr.append(accidents[lt_pe_str])
                        lt_nd_pre += datetime.timedelta(minutes= time_interval)
                        lt_pe_str = lt_nd_pre.strftime(second_format)
                    holiday_accidents_for_train[idx][time_now_str][YESTERDAY_KEY] = last_hl_nd_arr
                    #print "holiday len last holiday: %d" % len(last_hl_nd_arr)

                    last_monday_nw_arr = []
                    # 上周六(如果是长假最后一天则按照周日)的数据
                    demand_weekday = 5
                    if k_h == len(h_list[j_h]) - 1:
                        demand_weekday = 6
                    for d_l in range(1 , 14 + 9):
                        dt_dl = dt_now - datetime.timedelta(hours= 24 * d_l)
                        dt_dl_dt_str = dt_dl.strftime(date_format)
                        if (dt_dl_dt_str in holiday_str_list) or (dt_dl_dt_str in tiaoxiu_list):
                            continue
                        elif dt_dl.weekday() != demand_weekday:
                            continue
                        else:
                            lw_t_pre = dt_dl - datetime.timedelta(minutes= n_w * time_interval)
                            lw_t_pre_str = lw_t_pre.strftime(second_format)
                            lw_t_post = dt_dl + datetime.timedelta(minutes= n_w * time_interval)

                            while lw_t_pre <= lw_t_post:
                                last_monday_nw_arr.append(accidents[lw_t_pre_str])
                                lw_t_pre += datetime.timedelta(minutes= time_interval)
                                lw_t_pre_str = lw_t_pre.strftime(second_format)
                            holiday_accidents_for_train[idx][time_now_str][LAST_WEEK_KEY] = last_monday_nw_arr
                            #print "holiday len last saturday or sunday: %d" % len(last_monday_nw_arr)
                            break
    return tiaoxiu_accidents_for_train, holiday_accidents_for_train[0], holiday_accidents_for_train[1]
def generate_grid_for_beijing(lng_coors, lat_coors,output_file_path):
    output_file = open(output_file_path,"w")
    cnt = 0
    for i_lat in range(len(lat_coors)-1):
        for j_lng in range(len(lng_coors)-1):
            cnt += 1
            min_lng1 = lng_coors[j_lng]
            max_lng1 = lng_coors[j_lng + 1]
            min_lat1 = lat_coors[i_lat]
            max_lat1 = lat_coors[i_lat + 1]

            out_str ='''var rectangle_'''+str(cnt)+''' = new BMap.Polygon([
                            new BMap.Point(''' + str(min_lng1) + ''',''' + str(min_lat1) + '''),
                            new BMap.Point(''' + str(max_lng1) + ''',''' + str(min_lat1) + '''),
                            new BMap.Point(''' + str(max_lng1) + ''',''' + str(max_lat1) + '''),
                            new BMap.Point(''' + str(min_lng1) + ''',''' + str(max_lat1) + ''')
                        ], {strokeColor:"red", strokeWeight:1, strokeOpacity:1,fillColor:''});\n
                        map.addOverlay(rectangle_'''+str(cnt)+''');\n'''
            output_file.write(out_str)
    output_file.close()
def generate_polylines_for_beijing(lng_coors, lat_coors,output_file_path):#,min_lat,max_lat,min_lng,max_lng):
    output_file = open(output_file_path,"w")
    cnt = 0
    for i_lng in range(len(lng_coors)):
        now_lng = lng_coors[i_lng]
        cnt += 1
        out_str ='''var polyline_'''+str(cnt)+''' = new BMap.Polyline([
                        new BMap.Point(''' + str(now_lng) + ''', ''' + str(min_lat) + '''),
                        new BMap.Point(''' + str(now_lng) + ''', ''' + str(max_lat) + ''')
                    ], {strokeColor:"red", strokeWeight:1, strokeOpacity:1,fillColor:''});  \n
                    map.addOverlay(polyline_'''+str(cnt)+''');\n'''
        output_file.write(out_str)
    for i_lat in range(len(lat_coors)):
        now_lat = lat_coors[i_lat]
        cnt += 1
        out_str ='''var polyline_'''+str(cnt)+''' = new BMap.Polyline([
                        new BMap.Point(''' + str(min_lng) + ''', ''' + str(now_lat) + '''),
                        new BMap.Point(''' + str(max_lng) + ''', ''' + str(now_lat) + ''')
                    ], {strokeColor:"red", strokeWeight:1, strokeOpacity:1,fillColor:''});  \n
                    map.addOverlay(polyline_'''+str(cnt)+''');\n'''
        output_file.write(out_str)

    output_file.close()
def partition_geopoints_by_time(input_pickle_path,interval = 60):
    path_pkl_file = open(input_pickle_path,"rb")
    print "start load!"
    accidents = pickle.load(path_pkl_file)
    print "finish load!"
    start_dt = datetime.datetime.strptime("2016-01-01 00:00:00",second_format)
    end_dt = datetime.datetime.strptime("2017-03-01 00:00:00",second_format)
    time_delta = datetime.timedelta(minutes= interval)
    temp_time = start_dt
    temp_time_str = temp_time.strftime(minute_format)
    accidents_of_all = {}
    time_list = []
    while temp_time < end_dt:
        accidents_of_all[temp_time_str] = []
        time_list.append(temp_time_str)
        temp_time = temp_time + time_delta
        temp_time_str = temp_time.strftime(minute_format)
    now_time = start_dt
    now_time_str = now_time.strftime(minute_format)

    for i in range(len(accidents)):
        accidents_time = accidents[i].create_time
        if accidents_time > now_time and accidents_time <= (now_time + time_delta):
            accidents_of_all[now_time_str].append([accidents[i].longitude, accidents[i].latitude])
        elif accidents_time > (now_time + time_delta) :
            now_time = now_time + time_delta
            now_time_str = now_time.strftime(minute_format)
            accidents_of_all[now_time_str].append([accidents[i].longitude, accidents[i].latitude])
    print "finish partition points by time!"
    return time_list,accidents_of_all
#将国测局坐标转成百度坐标
def gcj2bd(point):
    x = point[0]
    y = point[1]
    z = math.sqrt(x * x + y * y) + 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) + 0.000003 * math.cos(x * x_pi)
    return z * math.cos(theta) + 0.0065, z * math.sin(theta) + 0.006
def convert_point_to_point_collection(input_file_path, output_file_path):
    reader = unicode_csv_reader(open(input_file_path,"rb"))
    data = {}
    data["data"] = []
    data["errorno"] = 0
    data["NearestTime"] = datetime.datetime.now().strftime(second_format)
    data["userTime"] = datetime.datetime.now().strftime(second_format)
    for row in reader:
        point = [float(row[2]), float(row[3])]
        point_converted = gcj2bd(point)
        data["data"].append([point_converted[0], point_converted[1], 1])
        #print row[0]

    data["total"] = len(data["data"])
    data["rt_loc_cnt"] = len(data["data"])
    output_file = open(output_file_path, "w")
    pre = "var data = "
    output_file.write(pre)

    wrt_str = json.dumps(data)
    print wrt_str

    output_file.write(wrt_str)
    output_file.close()
def label_all_function_regions(input_region_files,**params):
    d_lat = params['d_lat']
    d_lng = params['d_lng']
    n_lng = params['n_lng']
    n_lat = params['n_lat']
    d_len = params['d_len']
    geo_cnts = []
    for idx, input_region_file in enumerate(input_region_files):
        print "labeling %s" % input_region_file
        geo_cnts.append([0 for i in range(n_lat * n_lng)])

        reader = unicode_csv_reader(open(input_region_file,"rb"))
        for row in reader:
            #先经度, 后纬度
            point = [float(row[2]), float(row[3])]
            point_converted = gcj2bd(point)
            lng = point_converted[0]
            lat = point_converted[1]

            if (not (min_lng <= lng and lng <= max_lng and min_lat <= lat and lat <= max_lat)):
                continue
            else:
                j_lat = int(math.ceil((float(lat) - min_lat)/d_lat)) - 2
                i_lng = int(math.ceil((float(lng) - min_lng)/d_lng)) - 2

                id = i_lng * n_lat + j_lat
                geo_cnts[idx][id] += 1
        # print "idx:%d, len_geo_cnt:%d" % (idx,len(geo_cnts[idx]))
        geo_cnts_str = [str(item) for item in geo_cnts[idx]]
        geo_cnts_concat = ','.join(geo_cnts_str)
        # print "len_str %d" % len(geo_cnts_concat)
        region_function = Region_Function(spatial_interval = d_len, region_type = idx + 1, region_cnt_matrix = geo_cnts_concat)
        region_function.save()
        print "finish labeling %s" % input_region_file
    print "finish all"
def generate_grid_ids(lng_coors, lat_coors):
    #len: n_lat * n_lng - 1

    grids = []
    n_lat = len(lat_coors)-1
    n_lng = len(lng_coors)-1
    for i_lng in range(n_lng):
        for j_lat in range(n_lat):
            id = i_lng * n_lat + j_lat
            l_lat = lat_coors[j_lat]
            r_lat = lat_coors[j_lat + 1]
            d_lng = lng_coors[i_lng]
            u_lng = lng_coors[i_lng + 1]
            geo_points = [[l_lat, d_lng], [r_lat, d_lng], [l_lat, u_lng], [r_lat, u_lng]]
            # print "id: %d" % id
            # print geo_points
            grids.append(geo_points)
    print "len grids: %d" % len(grids)

    return grids
def label_geo_points(geo_points, d_lat, d_lng, n_lng, n_lat):
    geo_cnts = [0 for i in range(n_lat * n_lng)]
    for geo_point in geo_points:
        lng = geo_point[0]
        lat = geo_point[1]

        if (not (min_lng <= lng and lng <= max_lng and min_lat <= lat and lat <= max_lat)):
            continue
        else:
            j_lat = int(math.ceil((float(lat) - min_lat)/d_lat)) - 2
            i_lng = int(math.ceil((float(lng) - min_lng)/d_lng)) - 2

            id = i_lng * n_lat + j_lat
            geo_cnts[id] += 1
    return geo_cnts
def label_all_accidents(input_pickle_file,interval, **params):
    d_lat = params['d_lat']
    d_lng = params['d_lng']
    n_lng = params['n_lng']
    n_lat = params['n_lat']
    d_len = params['d_len']
    time_list, accidents_of_all = partition_geopoints_by_time(input_pickle_file,interval = interval)
    print "start labeling"
    for time_now in time_list:
        print time_now
        geo_points = accidents_of_all[time_now]
        geo_cnts = label_geo_points(geo_points, d_lat, d_lng, n_lng, n_lat)
        geo_cnts_str = [str(item) for item in geo_cnts]
        geo_cnts_concat = ','.join(geo_cnts_str)
        time_now_dt = datetime.datetime.strptime(time_now,minute_format)
        date_str = time_now_dt.strftime(date_format)
        weather= Weather.objects.filter(date_w= time_now_dt.date())[0]
        air_quality = Air_Quality.objects.filter(date_a=time_now_dt.date())[0]
        if date_str in holiday_str_list:
            holiday = True
        else:
            holiday = False
        weekend = False
        t_weekday = time_now_dt.weekday()
        if (t_weekday == 4 and time_now_dt.hour >= 17) or (t_weekday in [5, 6]):
            weekend = True

        t_hour = time_now_dt.hour
        if  t_hour>= 0 and t_hour < 7:
            time_segment = DAWN
        elif t_hour >=7 and t_hour < 9:
            time_segment = MORNING_RUSH
        elif t_hour >=9 and t_hour < 12:
            time_segment = MORNING_WORKING
        elif t_hour >=12 and t_hour < 14:
            time_segment = NOON
        elif t_hour >=14 and t_hour < 17:
            time_segment = AFTERNOON_WORK
        elif t_hour>= 17 and t_hour < 20:
            time_segment = AFTERNOON_RUSH
        else:
            time_segment = NIGHT

        accidents_array = Accidents_Array(time_interval= interval, spatial_interval = d_len, create_time = time_now_dt, content = geo_cnts_concat, highest_temperature= weather.highest_temperature, lowest_temperature=weather.lowest_temperature, wind=weather.wind, weather_severity=weather.weather_severity, aqi=air_quality.aqi, pm25= air_quality.pm25, is_holiday=holiday, is_weekend=weekend, time_segment= time_segment)
        accidents_array.save()
    print "finish labeling"
def get_all_accidents_from_db(output_pickle):
    outfile = open(output_pickle, 'wb')
    print "query start!"
    start_dt = datetime.datetime.strptime("2016-01-01 00:00:00",second_format)
    end_dt = datetime.datetime.strptime("2017-03-01 00:00:00",second_format)
    accidents = Call_Incidence.objects.filter(create_time__range=[start_dt, end_dt]).order_by("create_time")
    print "query finish!"
    print "before: %d" % len(accidents)
    accidents_filtered = []
    accidents_set = set('temp')
    for accident in accidents:
        out_str = "%s,%s,%s" %(accident.create_time.strftime(second_format), accident.latitude, accident.longitude)
        if out_str not in accidents_set:
            accidents_set.add(out_str)
            accidents_filtered.append(accident)
    #accidents_set.remove('temp')
    print "after set filtered len: %d\n start dump!" % len(accidents_filtered)

    pickle.dump(accidents_filtered,outfile,-1)
    outfile.close()

    print "dump success!"
def handle_gaode_ret(ret):
    pois = ret['pois']
    out_strs = []
    for poi in pois:
        out_temp_str = u"%s,%s,%s" % (poi['name'],poi['typecode'],poi['location'])
        out_strs.append(out_temp_str)
    out_str = '\n'.join(out_strs)
    return out_str
def get_pois_from_gaode(type, output_file_path):
    output_file = open(output_file_path, "w")
    url = 'http://restapi.amap.com/v3/place/text?'
    headers = {}
    offset = 20
    params = {
        'types' : type,
        'city': '010',
        'citylimit' : 'true',
        'offset' : offset,
        'page' : 1,
        'output':'JSON',
        'key':'e4b3fa82553f7d984ff168f8c9de115c'
    }
    ret = requests.get(url,params = params,headers=headers).json()

    output_file.write(handle_gaode_ret(ret))

    total =  int(ret['count'])
    total_pages = int(math.ceil(float(total)/offset))
    print "tot: %d, tot_pages: %d" % (total, total_pages)

    for i in range(2,total_pages):
        print "now handle page %d" % i
        params['page'] = i
        ret = requests.get(url,params = params,headers=headers).json()
        output_file.write(handle_gaode_ret(ret))
    output_file.close()
    print "finish write %s" % type
def get_pois_from_baidu(keyword, industry_type, output_file_path):
    url = 'http://api.map.baidu.com/place/v2/search?'
    headers = {}
    params = {
        'query':keyword,
        'page_num':19,
        'page_size':20,
        'output':'json',
        'scope':2,
        'filter':'industry_type:'+ industry_type,
        'region':'北京',
        'ak':'eM0GfCwd27kZRyM49ZOkvkOaidDXz6Wf'
    }

    ret = requests.get(url,params = params,headers=headers).json()
    print ret
    total = int(ret['total'])
    print len(ret['results'])

def get_liuhuan_poi(output_file_path, sep = 500):
    # keyword = '六环'
    # url = 'http://api.map.baidu.com/place/v2/search?'
    # headers = {}
    # params = {
    #     'query':keyword,
    #     'page_num':0,
    #     'page_size':20,
    #     'output':'json',
    #     'scope':1,
    #     'region':'北京',
    #     'ak':'eM0GfCwd27kZRyM49ZOkvkOaidDXz6Wf'
    # }
    # #ret = requests.get(url,params = params,headers=headers).json()
    # results = []
    # min_lat = 9999999.0
    # max_lat = -9999999.0
    # min_lng = 9999999.0
    # max_lng = -9999999.0
    # tot_pages = 5
    # for i in range(tot_pages):
    #     params['page_num'] = i
    #     res = requests.get(url,params = params,headers=headers).json()
    #     res = res['results']
    #     for j in range(len(res)):
    #         res_t = {}
    #         res_t['lat'] = res[j]['location']['lat']
    #         res_t['lng'] = res[j]['location']['lng']
    #         res_t['name'] = res[j]['name']
    #         res_t['address'] = res[j]['address']
    #         if res_t['lat'] < min_lat:
    #             min_lat = res_t['lat']
    #         elif res_t['lat'] > max_lat:
    #             max_lat = res_t['lat']
    #         if res_t['lng'] < min_lng:
    #             min_lng = res_t['lng']
    #         elif res_t['lng'] > max_lng:
    #             max_lng = res_t['lng']
    #         #print res_t
    #         results.append(res_t)
    if sep == 500:
        d_lat = 0.0042
        d_lng = 0.006
    else:
        d_lat = 0.0084
        d_lng = 0.012

    n_lat_delta = int(math.ceil((max_lat - min_lat)/d_lat))
    n_lng_delta = int(math.ceil((max_lng - min_lng)/d_lng))

    lng_coors = [min_lng + i * d_lng for i in range(n_lng_delta)]
    lat_coors = [min_lat + i * d_lat for i in range(n_lat_delta)]
    print "grid size = lng: %d * lat: %d\n" % (len(lng_coors)-1, len(lat_coors)-1)
    generate_grid_ids(lng_coors,lat_coors)
    #generate_grid_for_beijing(lng_coors, lat_coors,output_file_path)
    generate_polylines_for_beijing(lng_coors,lat_coors,output_file_path)#,min_lat,max_lat,min_lng,max_lng)
    print 'min_lat: %f, max_lat: %f, min_lng: %f , max_lng: %f\n' % (min_lat, max_lat, min_lng, max_lng)
    return min_lat,max_lat,min_lng,max_lng
