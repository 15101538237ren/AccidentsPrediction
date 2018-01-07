# -*- coding: utf-8 -*-
import sys,os,requests,urllib,json,math,pickle,datetime,simplejson,decimal
from  models import *
from import_data import unicode_csv_reader
import numpy as np #导入Numpy
from AccidentsPrediction.settings import BASE_DIR
import pickle
from class_for_shape import Vector2, Rect, CheckRectLine

reload(sys)
sys.setdefaultencoding('utf8')
## 六环
# min_lat = 39.696203
# max_lat = 40.181729
# min_lng = 116.099649
# max_lng = 116.718542

## 五环
min_lat = 39.764427
max_lat = 40.033227#40.028983
min_lng = 116.214834
max_lng = 116.562834#116.554975
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
date_new_format = "%Y%m%d"
second_new_format = "%Y%m%d %H:%M:%S"
minute_format = "%Y-%m-%d %H:%M"
minute_format2 = "%Y%m%d_%H_%M"
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

error_mapping = {
    "LOGIN_NEEDED": (1, "login needed"),
    "PERMISSION_DENIED": (2, "permission denied"),
    "DATABASE_ERROR": (3, "operate database error"),
    "ONLY_FOR_AJAX": (4, "the url is only for ajax request")
}

class ApiError(Exception):
    def __init__(self, key, **kwargs):
        Exception.__init__(self)
        self.key = key if key in error_mapping else "UNKNOWN"
        self.kwargs = kwargs

def ajax_required(func):
    def __decorator(request, *args, **kwargs):
        if request.is_ajax:
            return func(request, *args, **kwargs)
        else:
            raise ApiError("ONLY_FOR_AJAX")
    return __decorator

def safe_new_datetime(d):
    kw = [d.year, d.month, d.day]
    if isinstance(d, datetime.datetime):
        kw.extend([d.hour, d.minute, d.second, d.microsecond, d.tzinfo])
    return datetime.datetime(*kw)

def safe_new_date(d):
    return datetime.date(d.year, d.month, d.day)

class DatetimeJSONEncoder(simplejson.JSONEncoder):
    """可以序列化时间的JSON"""

    DATE_FORMAT = "%Y-%m-%d"
    TIME_FORMAT = "%H:%M:%S"

    def default(self, o):
        if isinstance(o, datetime.datetime):
            d = safe_new_datetime(o)
            return d.strftime("%s %s" % (self.DATE_FORMAT, self.TIME_FORMAT))
        elif isinstance(o, datetime.date):
            d = safe_new_date(o)
            return d.strftime(self.DATE_FORMAT)
        elif isinstance(o, datetime.time):
            return o.strftime(self.TIME_FORMAT)
        elif isinstance(o, decimal.Decimal):
            return str(o)
        else:
            return super(DatetimeJSONEncoder, self).default(o)

def get_conv_kernal_crespond_data(x, w, b, conv_param):
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
  out = np.zeros((N,F,Ho,Wo,HH*WW))

  for f in xrange(F):
    for i in xrange(Ho):
      for j in xrange(Wo):
        # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
        out[:,f,i,j,:] = x_pad[:, :, i*S : i*S+HH, j*S : j*S+WW].flatten()
  return out

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
#获得所有网格对应的中心经纬度
def get_lng_lat_centers_of_all_grids(spatial_interval):
    if spatial_interval == 500:
        d_lat = 0.0042
        d_lng = 0.006
    else:
        d_lat = 0.0084
        d_lng = 0.012
    n_lat_delta_origin = (max_lat - min_lat)/d_lat
    n_lat_delta = int(math.ceil(n_lat_delta_origin)) + 1
    n_lng_delta_origin = (max_lng - min_lng)/d_lng + 1
    n_lng_delta = int(math.ceil(n_lng_delta_origin))

    lng_coors = [min_lng + i * d_lng for i in range(n_lng_delta)]
    lat_coors = [min_lat + i * d_lat for i in range(n_lat_delta)]
    n_lng = len(lng_coors)-1
    n_lat = len(lat_coors)-1
    print "grid size = lng: %d * lat: %d\n" % (n_lng, n_lat)
    n_lat = len(lat_coors)-1
    n_lng = len(lng_coors)-1
    lngs = []
    lats = []
    mean_lngs = np.zeros(n_lng * n_lat)
    mean_lats = np.zeros(n_lng * n_lat)

    for i_lng in range(n_lng):
        for j_lat in range(n_lat):
            id = i_lng * n_lat + j_lat
            min_lng1 = lng_coors[i_lng]
            max_lng1 = lng_coors[i_lng + 1]
            min_lat1 = lat_coors[j_lat]
            max_lat1 = lat_coors[j_lat + 1]
            center_lng = (min_lng1 + max_lng1)/2.0
            center_lat = (min_lat1 + max_lat1)/2.0
            mean_lngs[id] += center_lng
            mean_lats[id] += center_lat

            lngs.append(center_lng)
            lats.append(center_lat)
    mean_lng = mean_lngs.mean()
    mean_lat = mean_lats.mean()
    lngs = np.array(lngs) - mean_lng
    lats = np.array(lats) - mean_lat

    return [lngs, lats]


#获取数据库中工作日时段的所有事故相关的数据
def get_work_day_data(dt_start,dt_end,data_bounds,time_interval, spatial_interval):
    work_day_accidents = {}
    work_day_accidents_arr = []
    for day_start, day_end in data_bounds:
        dt_start = datetime.datetime.strptime(day_start + hour_0, second_format)
        dt_end = datetime.datetime.strptime(day_end + end_of_day, second_format)
        accidents = Accidents_Array.objects.filter(time_interval= time_interval, spatial_interval= spatial_interval, create_time__range=[ dt_start, dt_end]).order_by("create_time")
        for accident in accidents:
            if dt_start <= accident.create_time < dt_end:
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

#获取工作日训练数据(n:过去的几个小时,n_d:昨天当前时刻前后的n_d个小时,n_w:上周对应星期几的对应时间前后的n_w个小时
def get_work_day_data_for_train(dt_start,dt_end,time_interval, spatial_interval, n, n_d, n_w):
    work_day_accidents, work_day_accidents_arr = get_work_day_data(dt_start,dt_end,work_day_bounds,time_interval, spatial_interval)

    len_arr = len(work_day_accidents)
    print "len arr %d" % len_arr
    hour_delta = datetime.timedelta(hours=1)
    work_day_accidents_for_train = {}
    t_time_interval = int(60 / time_interval)
    print "start get workday data for train"
    #从1月12日开始生成训练数据
    for i in range(len(work_day_accidents_arr)):
        time_now = work_day_accidents_arr[i].create_time
        if time_now < dt_start:
            continue
        if time_now >= dt_end:
            break
        time_now_str = time_now.strftime(second_format)
        now_week_day = time_now.weekday()
        # print "get workday: %s" % time_now_str
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
def generate_arrays_of_train(data_list, label_list, batch_size):
    len_arr = len(data_list)
    len_lbl = len(label_list)
    if len_arr != len_lbl:
        print "len_arr != len_lbl of train generator"
    idx_of_test = np.arange(len_arr)
    np.random.shuffle(idx_of_test)
    while 1:
        X = []
        Y = []
        for idx in range(len(idx_of_test)):
            try:
                X.append(data_list[idx_of_test[idx]])
                Y.append(label_list[idx_of_test[idx]])
                if (idx + 1) % batch_size == 0:
                    X_arr = np.array(X)
                    Y_arr = np.array(Y)

                    yield (X_arr, Y_arr)
                    X = []
                    Y = []
            except IndexError, e:
                print "idx %d, r_idx: %d, len_data_list: %d" %(idx, idx_of_test[idx], len(data_list))

def generate_arrays_of_train_aux(data_list, aux_data_list, label_list, batch_size):
    len_arr = len(data_list)
    len_lbl = len(label_list)
    if len_arr != len_lbl:
        print "len_arr != len_lbl of train generator"
    idx_of_test = np.arange(len_arr)
    np.random.shuffle(idx_of_test)
    while 1:
        X = []
        Y = []
        aux = []
        for idx in range(len(idx_of_test)):
            try:
                X.append(data_list[idx_of_test[idx]])
                aux.append(aux_data_list[idx_of_test[idx]])
                Y.append(label_list[idx_of_test[idx]])
                if (idx + 1) % batch_size == 0:
                    yield ([np.array(X), np.array(aux)], np.array(Y))
                    X = []
                    aux = []
                    Y = []
            except IndexError, e:
                print "idx %d, r_idx: %d, len_data_list: %d" %(idx, idx_of_test[idx], len(data_list))
#生成训练用的数据generator
def generate_arrays_of_test(data_list, label_list, batch_size):
    len_arr = len(data_list)
    len_lbl = len(label_list)
    if len_arr != len_lbl:
        print "len_arr != len_lbl of test generator"
    idx_of_test = np.arange(len_arr)
    np.random.shuffle(idx_of_test)
    while 1:
        X = []
        Y = []
        for idx in range(len(idx_of_test)):
            try:
                X.append(data_list[idx_of_test[idx]])
                Y.append(label_list[idx_of_test[idx]])
                if (idx + 1) % batch_size == 0:
                    X_arr = np.array(X)
                    Y_arr = np.array(Y)

                    yield (X_arr, Y_arr)
                    X = []
                    Y = []
            except IndexError, e:
                print "idx %d, r_idx: %d, len_data_list: %d" %(idx, idx_of_test[idx], len(data_list))
def generate_function_arrays_of_train(data_list, label_list, function_list, batch_size):
    len_arr = len(data_list)
    idx_of_train = np.arange(len_arr)
    np.random.shuffle(idx_of_train)
    while 1:
        X1 = []
        X2 = []
        Y = []
        for idx in range(len(idx_of_train)):
            try:
                X1.append(data_list[idx_of_train[idx]])
                X2.append(function_list[idx_of_train[idx]])
                Y.append(label_list[idx_of_train[idx]])
                if (idx + 1) % batch_size == 0:
                    yield (np.array([X1, X2]), np.array(Y))
                    X1 = []
                    Y = []
            except IndexError, e:
                print "idx %d, r_idx: %d, len_data_list: %d" %(idx, idx_of_train[idx], len(data_list))
def generate_arrays_of_validation(data_list, label_list, batch_size):
    len_arr = len(data_list)
    len_lbl = len(label_list)
    if len_arr != len_lbl:
        print "len_arr != len_lbl of validator generator"
    idx_of_val = np.arange(len_arr)
    np.random.shuffle(idx_of_val)
    while 1:
        X = []
        Y = []
        for idx in range(len_arr):
            try:
                X.append(data_list[idx_of_val[idx]])
                Y.append(label_list[idx_of_val[idx]])
                if (idx + 1) % batch_size == 0:
                    X_arr = np.array(X)
                    Y_arr = np.array(Y)
                    # print "Val X shape:",
                    yield (X_arr, Y_arr)
                    X = []
                    Y = []
            except IndexError, e:
                print "idx %d, r_idx: %d, len_data_list: %d" %(idx, idx_of_val[idx], len(data_list))
def generate_arrays_of_validation_aux(data_list,aux_data, label_list, batch_size):
    len_arr = len(data_list)
    len_lbl = len(label_list)
    if len_arr != len_lbl:
        print "len_arr != len_lbl of validator generator"
    idx_of_val = np.arange(len_arr)
    np.random.shuffle(idx_of_val)
    while 1:
        X = []
        Y = []
        aux = []
        for idx in range(len_arr):
            try:
                X.append(data_list[idx_of_val[idx]])
                aux.append(aux_data[idx_of_val[idx]])
                Y.append(label_list[idx_of_val[idx]])
                if (idx + 1) % batch_size == 0:
                    # print "Val X shape:",
                    yield ([np.array(X), np.array(aux)], np.array(Y))
                    X = []
                    Y = []
                    aux = []
            except IndexError, e:
                print "idx %d, r_idx: %d, len_data_list: %d" %(idx, idx_of_val[idx], len(data_list))

def generate_function_arrays_of_validation(data_list, label_list,function_list, batch_size):
    len_arr = len(data_list)
    idx_of_val = np.arange(len_arr)
    np.random.shuffle(idx_of_val)
    while 1:
        X1 = []
        X2 = []
        Y = []
        for idx in range(len_arr):
            try:
                X1.append(data_list[idx_of_val[idx]])
                X2.append(function_list[idx_of_val[idx]])
                Y.append(label_list[idx_of_val[idx]])
                if (idx + 1) % batch_size == 0:
                    yield (np.array([X1,X2]), np.array(Y))
                    X1 = []
                    X2 = []
                    Y = []
            except IndexError, e:
                print "idx %d, r_idx: %d, len_data_list: %d" %(idx, idx_of_val[idx], len(data_list))
def get_array_of_seq(zero_special_list, positive_list, zero_workday_list, zero_special_lable_list, positive_label_list, zero_workday_label_list):
    cnt_zero_special = len(zero_special_list)
    cnt_positive = len(positive_list)
    cnt_zero_workday = len(zero_workday_list)

    cnt_zero_workday_label = len(zero_workday_label_list)
    cnt_positive_label = len(positive_label_list)
    cnt_zero_special_label = len(zero_special_lable_list)

    rtn_arr = []
    rtn_lbl_arr = []

    if (cnt_zero_workday != cnt_zero_workday_label) or (cnt_positive != cnt_positive_label) or (cnt_zero_special != cnt_zero_special_label):
        print "size not match!"
        return np.array(rtn_arr), np.array(rtn_lbl_arr)

    max_len = max(max(cnt_zero_workday, cnt_zero_special), cnt_positive)

    print "get arr of max_len: %d " % max_len
    cnt = 0
    while cnt < max_len:
        if cnt < cnt_zero_special:
            rtn_arr.append(zero_special_list[cnt])
            rtn_lbl_arr.append(zero_special_lable_list[cnt])
        if cnt < cnt_positive:
            rtn_arr.append(positive_list[cnt])
            rtn_lbl_arr.append(positive_label_list[cnt])
        if cnt < cnt_zero_workday:
            rtn_arr.append(zero_workday_list[cnt])
            rtn_lbl_arr.append(zero_workday_label_list[cnt])
        cnt += 1
        if cnt % 10000==0:
            print "cnt %d, percent:%.3f" % (cnt, float(cnt)/float(max_len))

    return np.array(rtn_arr), np.array(rtn_lbl_arr)
def get_array_of_seq_of_function(zero_special_list, positive_list, zero_workday_list, zero_special_lable_list, positive_label_list, zero_workday_label_list, zero_special_function_list,temp_positive_function_list, zero_workday_function_list):
    cnt_zero_special = len(zero_special_list)
    cnt_positive = len(positive_list)
    cnt_zero_workday = len(zero_workday_list)
    cnt_zero_workday_function = len(zero_workday_function_list)
    rtn_arr = []
    rtn_lbl_arr = []
    rtn_function_arr = []

    cnt_zero_workday_label = len(zero_workday_label_list)
    if (cnt_zero_workday != cnt_zero_workday_label) or (cnt_zero_workday!= cnt_zero_workday_function):
        print "size not match!"
        return np.array(rtn_arr), np.array(rtn_lbl_arr), np.array(rtn_function_arr)

    max_len = max(max(cnt_zero_workday, cnt_zero_special), cnt_positive)

    print "get arr of max_len: %d " % max_len
    cnt = 0
    while cnt < max_len:
        if cnt < cnt_zero_special:
            rtn_arr.append(zero_special_list[cnt])
            rtn_lbl_arr.append(zero_special_lable_list[cnt])
            rtn_function_arr.append(zero_special_function_list[cnt])
        if cnt < cnt_positive:
            rtn_arr.append(positive_list[cnt])
            rtn_lbl_arr.append(positive_label_list[cnt])
            rtn_function_arr.append(temp_positive_function_list[cnt])
        if cnt < cnt_zero_workday:
            rtn_arr.append(zero_workday_list[cnt])
            rtn_lbl_arr.append(zero_workday_label_list[cnt])
            rtn_function_arr.append(zero_workday_function_list[cnt])
        cnt += 1
        if cnt % 10000==0:
            print "cnt %d, percent:%.3f" % (cnt, float(cnt)/float(max_len))

    return np.array(rtn_arr), np.array(rtn_lbl_arr), np.array(rtn_function_arr)

def generate_data_for_train_and_test(flatten,load_traffic_data,out_pickle_file_path, dt_start, train_dt_end, validation_dt_end, test_dt_end, time_interval, n, n_d, n_w, **params):
    train_data = []
    train_label = []

    test_data = []
    test_label = []

    validation_data = []
    validation_label = []

    train_and_validation_data = []
    train_and_validation_label = []

    cnt_pos_train = cnt_neg_train = 0
    cnt_pos_val = cnt_neg_val = 0
    cnt_pos_tv = cnt_neg_tv = 0
    cnt_pos_test = cnt_neg_test = 0

    #经度网格数量
    width = params["n_lng"]
    #纬度网格数量
    height = params["n_lat"]
    #空间间隔长度
    spatial_interval = params["d_len"]

    # 内层每一个样本点的每个时间点对应的数据维度
    # conv_dim = 9
    # data_dim = 4 + conv_dim

    if load_traffic_data:
        added = 1
    else:
        added = 0
    data_dim = added + 1 + 4#conv_dim * 2 + 4
    #卷积操作相关

    x_shape = (1, 1, height, width) #n,c,h,w
    out_shape = (1, 1, height * width)
    w_shape = (1, 1, 3, 3) #f,c,hw,ww
    w = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5,]).reshape(w_shape)
    b = np.array([0])
    conv_param = {'stride': 1, 'pad': 1}

    work_day_dt_start = datetime.datetime.strptime("2016-01-12 00:00:00", second_format)
    work_day_acc = get_work_day_data_for_train(work_day_dt_start,test_dt_end,time_interval, spatial_interval, n, n_d, n_w)
    holiday_dt_start = datetime.datetime.strptime("2016-01-01 00:00:00", second_format)
    tiaoxiu_acc, holiday_3_acc, holiday_7_acc = get_holiday_and_tiaoxiu_data_for_train(holiday_dt_start, test_dt_end,time_interval, spatial_interval, n, n_d, n_w)

    dt_list = []
    dt_now = dt_start
    while dt_now < test_dt_end:
        dt_list.append(dt_now)
        dt_now += datetime.timedelta(minutes= time_interval)

    dt_cnt = 0
    count_limit = -1
    [lngs, lats] = get_lng_lat_centers_of_all_grids(spatial_interval)

    [top_indexs,region_normed_accidents_prob] = region_difference_calc(dt_start, test_dt_end, time_interval,spatial_interval, BASE_DIR+'/static/js/region.js', 5)
    for dt_now in dt_list:
        dt_str = dt_now.strftime(second_format)
        dt_str_date = dt_now.strftime(date_format)
        if (dt_str_date in holiday_7_list[0]) or (dt_str_date in holiday_3_list[0]) or (dt_str_date == tiaoxiu_list[0]):
            continue
        dt_cnt += 1

        if dt_cnt < count_limit or count_limit < 0:
            if dt_str_date in holiday_3_list_flatten:
                data_now = holiday_3_acc[dt_str]
            elif dt_str_date in holiday_7_list_flatten:
                data_now = holiday_7_acc[dt_str]
            elif dt_str_date in tiaoxiu_list:
                data_now = tiaoxiu_acc[dt_str]
            else:
                data_now = work_day_acc[dt_str]

            data_last_week = data_now[LAST_WEEK_KEY]
            data_yesterday = data_now[YESTERDAY_KEY]
            data_last_hours = data_now[LAST_N_HOUR_KEY]
            data_labels = data_now[LABEL_KEY]

            data_merge = data_last_week + data_yesterday + data_last_hours
            len_data_merge = len(data_merge)
            data_shape = (height * width, 2)
            # data_shape = (height * width, len_data_merge, data_dim)
            data_for_now = np.zeros(data_shape)

            for idx, data_i in enumerate(data_merge):
                # data_content = np.array([int(item) for item in data_i.content.split(",")])
                # data_for_now[:, idx, 0] = data_content

                # extra_data = [float(data_i.weather_severity)/5.0,  float(data_i.pm25)/430.0,float(data_i.time_segment)/6.0,int(data_i.is_holiday)]
                # data_for_now[:, idx,1: data_dim-added] = extra_data

                if load_traffic_data:
                    grid_speed_nows = Grid_Speed.objects.filter(time_interval=time_interval, spatial_interval=spatial_interval, create_time=data_i.create_time)
                    if len(grid_speed_nows):
                        grid_speed_now = np.array([float(item) for item in grid_speed_nows[0].content.split(",")])
                        # grid_speed_now = grid_speed_now.reshape(x_shape)
                        # out_conv_of_speed= get_conv_kernal_crespond_data(grid_speed_now, w, b, conv_param)
                        data_for_now[:,idx, -1] = grid_speed_now

                        # for w_i in range(width):
                        #     for h_j in range(height):
                        #         wh_id = w_i * height + h_j
                        #         data_for_now[wh_id,idx, conv_dim: 2
            # data_arr = [1 if int(item) > 0 else 0 for item in data_labels.content.split(",")]

            # print "data_for_now shape: ",
            # print data_for_now.shape

            data_arr =[]
            for item in data_labels.content.split(","):
                if int(item) > 0:
                    content_to_append = float(item)
                else:
                    content_to_append = 0
                data_arr.append(content_to_append)
            # if load_traffic_data:
            #     grid_speed_nows = Grid_Speed.objects.filter(time_interval=time_interval, spatial_interval=spatial_interval, create_time=data_labels.create_time)
            #     grid_speed_now = np.array([float(item) for item in grid_speed_nows[0].content.split(",")])
            #     data_for_now[:, -1] = grid_speed_now
            for i_t in range(height * width):
                # data_for_now[i_t, 1 : ] = [float(data_labels.weather_severity)/5.0,  float(data_labels.pm25)/430.0,float(data_labels.time_segment)/6.0,int(data_labels.is_holiday),lngs[i_t], lats[i_t]]#[lngs[i_t], lats[i_t]]#
                # if flatten:
                #     arr_flatten = list(data_for_now[i_t, :].flatten())
                #     arr_flatten.extend([lngs[i_t], lats[i_t]])#lngs[i_t], lats[i_t]
                # else:
                #     arr_flatten = data_for_now[i_t, :]
                data_for_now[i_t, 0:] = [lngs[i_t], lats[i_t]]#
                arr_flatten = data_for_now[i_t, :]
                dt_label_now = data_arr[i_t]
                if dt_start < dt_now < validation_dt_end:
                    # tv_here = False
                    # if dt_label_now == 1:
                    #     cnt_pos_tv += 1
                    #     tv_here = True
                    # elif cnt_neg_tv < cnt_pos_tv:
                    #     cnt_neg_tv += 1
                    #     tv_here = True
                    # if tv_here:
                    train_and_validation_data.append(arr_flatten)
                    train_and_validation_label.append(dt_label_now)
                    if dt_start < dt_now < train_dt_end:
                        # train_here = False
                        # if dt_label_now == 1:
                        #     cnt_pos_train += 1
                        #     train_here = True
                        # elif cnt_neg_train < cnt_pos_train:
                        #     cnt_neg_train += 1
                        #     train_here = True
                        # if train_here:
                        train_data.append(arr_flatten)
                        train_label.append(dt_label_now)
                    elif train_dt_end < dt_now < validation_dt_end:
                        # validate_here = False
                        # if dt_label_now == 1:
                        #     cnt_pos_val += 1
                        #     validate_here = True
                        # elif cnt_neg_val < cnt_pos_val:
                        #     cnt_neg_val += 1
                        #     validate_here = True
                        # if validate_here:
                        validation_data.append(arr_flatten)
                        validation_label.append(dt_label_now)
                else:
                    # test_here = False
                    # if dt_label_now == 1:
                    #     cnt_pos_test += 1
                    #     test_here = True
                    # elif cnt_neg_test < cnt_pos_test:
                    #     cnt_neg_test += 1
                    #     test_here = True
                    # if test_here:
                    test_data.append(arr_flatten)
                    test_label.append(dt_label_now)

            print "finish %s" % dt_str

    # print "pos_train %d, neg_train %d" %(cnt_pos_train, cnt_neg_train)
    # print "pos_val %d, neg_val %d" %(cnt_pos_val, cnt_neg_val)
    # print "pos_test %d, neg_test %d" %(cnt_pos_test, cnt_neg_test)
    # print "pos_tv %d, neg_tv %d" %(cnt_pos_tv, cnt_neg_tv)
    return [train_data, validation_data, test_data, train_and_validation_data , train_label, validation_label, test_label, train_and_validation_label]

def generate_data_for_train_and_test_bk(flatten,is_tsne,load_traffic_data,out_pickle_file_path, dt_start, train_dt_end, validation_dt_end, test_dt_end, time_interval, n, n_d, n_w, **params):
    train_data = []
    train_label = []

    test_data = []
    test_label = []

    validation_data = []
    validation_label = []

    train_and_validation_data = []
    train_and_validation_label = []

    all_data = []
    all_label = []

    train_aux_data = []
    val_aux_data = []
    test_aux_data = []

    cnt_pos_train = cnt_neg_train = 0
    cnt_pos_val = cnt_neg_val = 0
    cnt_pos_tv = cnt_neg_tv = 0
    cnt_pos_test = cnt_neg_test = 0

    #经度网格数量
    width = params["n_lng"]
    #纬度网格数量
    height = params["n_lat"]
    #空间间隔长度
    spatial_interval = params["d_len"]

    # 内层每一个样本点的每个时间点对应的数据维度
    # conv_dim = 9
    # data_dim = 4 + conv_dim

    if load_traffic_data:
        added = 1
    else:
        added = 0
    data_dim = 1 + added # + 4#+ 1# + 4#conv_dim * 2 + 4
    #卷积操作相关

    x_shape = (1, 1, height, width) #n,c,h,w
    out_shape = (1, 1, height * width)
    w_shape = (1, 1, 3, 3) #f,c,hw,ww
    w = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5,]).reshape(w_shape)
    b = np.array([0])
    conv_param = {'stride': 1, 'pad': 1}

    work_day_dt_start = datetime.datetime.strptime("2016-01-12 00:00:00", second_format)
    work_day_acc = get_work_day_data_for_train(work_day_dt_start,test_dt_end,time_interval, spatial_interval, n, n_d, n_w)
    holiday_dt_start = datetime.datetime.strptime("2016-01-01 00:00:00", second_format)
    tiaoxiu_acc, holiday_3_acc, holiday_7_acc = get_holiday_and_tiaoxiu_data_for_train(holiday_dt_start, test_dt_end,time_interval, spatial_interval, n, n_d, n_w)

    dt_list = []
    dt_now = dt_start
    while dt_now < test_dt_end:
        dt_list.append(dt_now)
        dt_now += datetime.timedelta(minutes= time_interval)

    dt_cnt = 0
    count_limit = -1
    [lngs, lats] = get_lng_lat_centers_of_all_grids(spatial_interval)

    [top_indexs,region_normed_accidents_prob] = region_difference_calc(dt_start, test_dt_end, time_interval,spatial_interval, BASE_DIR+'/static/js/region.js', 5)
    for dt_now in dt_list:
        dt_str = dt_now.strftime(second_format)
        dt_str_date = dt_now.strftime(date_format)
        if (dt_str_date in holiday_7_list[0]) or (dt_str_date in holiday_3_list[0]) or (dt_str_date == tiaoxiu_list[0]):
            continue
        dt_cnt += 1

        if dt_cnt < count_limit or count_limit < 0:
            if dt_str_date in holiday_3_list_flatten:
                data_now = holiday_3_acc[dt_str]
            elif dt_str_date in holiday_7_list_flatten:
                data_now = holiday_7_acc[dt_str]
            elif dt_str_date in tiaoxiu_list:
                data_now = tiaoxiu_acc[dt_str]
            else:
                data_now = work_day_acc[dt_str]

            data_last_week = data_now[LAST_WEEK_KEY]
            data_yesterday = data_now[YESTERDAY_KEY]
            data_last_hours = data_now[LAST_N_HOUR_KEY]
            data_labels = data_now[LABEL_KEY]

            data_merge = data_last_week + data_yesterday + data_last_hours
            len_data_merge = len(data_merge)
            # data_shape = (height * width, len_data_merge)
            data_shape = (height * width, len_data_merge, data_dim)
            data_for_now = np.zeros(data_shape)

            for idx, data_i in enumerate(data_merge):
                # data_content = np.array([int(item) for item in data_i.content.split(",")])
                # data_for_now[:, idx, 0] = data_content

                # data_content = data_content.reshape(x_shape)
                # out_conv= get_conv_kernal_crespond_data(data_content, w, b, conv_param)
                # data_for_now[:, idx] = data_content

                # for w_i in range(width):
                #     for h_j in range(height):
                #         wh_id = w_i * height + h_j
                #         data_for_now[wh_id,idx, 0: conv_dim] = out_conv[0,0,h_j, w_i,:]
                # ,  int(data_i.is_weekend)* conv_dim] = out_conv_of_speed[0,0,h_j, w_i,:]

                # extra_data = [float(data_i.weather_severity)/5.0,  float(data_i.pm25)/430.0,float(data_i.time_segment)/6.0,int(data_i.is_holiday)]
                # data_for_now[:, idx,1: data_dim-added] = extra_data

                if load_traffic_data:
                    grid_speed_nows = Grid_Speed.objects.filter(time_interval=time_interval, spatial_interval=spatial_interval, create_time=data_i.create_time)
                    if len(grid_speed_nows):
                        grid_speed_now = np.array([float(item) for item in grid_speed_nows[0].content.split(",")])
                        # grid_speed_now = grid_speed_now.reshape(x_shape)
                        # out_conv_of_speed= get_conv_kernal_crespond_data(grid_speed_now, w, b, conv_param)
                        data_for_now[:,idx, -1] = grid_speed_now

                        # for w_i in range(width):
                        #     for h_j in range(height):
                        #         wh_id = w_i * height + h_j
                        #         data_for_now[wh_id,idx, conv_dim: 2
            # data_arr = [1 if int(item) > 0 else 0 for item in data_labels.content.split(",")]

            # print "data_for_now shape: ",
            # print data_for_now.shape

            data_arr =[]
            for item in data_labels.content.split(","):
                if int(item) > 0:
                    content_to_append = float(item)
                else:
                    content_to_append = 0
                data_arr.append(content_to_append)
            # for i_t in range(height * width):
            #     if flatten:
            #         arr_flatten = list(data_for_now[i_t, :].flatten())
            #         arr_flatten.extend([region_normed_accidents_prob[i_t],lngs[i_t], lats[i_t]])
            #     else:
            #         arr_flatten = data_for_now[i_t, :]
            #     aux_data = [region_normed_accidents_prob[i_t],lngs[i_t], lats[i_t]]
            #     if dt_start < dt_now < validation_dt_end:
            #             train_and_validation_data.append(arr_flatten)
            #             train_and_validation_label.append(data_arr[i_t])
            #             if dt_start < dt_now < train_dt_end:
            #                 train_data.append(arr_flatten)
            #                 train_aux_data.append(aux_data)
            #                 train_label.append(data_arr[i_t])
            #             elif train_dt_end < dt_now < validation_dt_end:
            #                 validation_data.append(arr_flatten)
            #                 val_aux_data.append(aux_data)
            #                 validation_label.append(data_arr[i_t])
            #     else:
            #         test_data.append(arr_flatten)
            #         test_aux_data.append(aux_data)
            #         test_label.append(data_arr[i_t])
            #
            #     if is_tsne:
            #         if i_t in top_indexs:
            #             all_data.append(arr_flatten)
            #             all_label.append(data_arr[i_t])

            for i_t in range(height * width):
                if flatten:
                    arr_flatten = [[0]]#list(data_for_now[i_t, :].flatten())
                    # arr_flatten.extend([lngs[i_t], lats[i_t]])#lngs[i_t], lats[i_t]
                else:
                    arr_flatten = [[0]]#data_for_now[i_t, :]
                dt_label_now = data_arr[i_t]
                aux_data = [lngs[i_t], lats[i_t]]#
                if dt_start < dt_now < validation_dt_end:
                        # tv_here = False
                        # if dt_label_now == 1:
                        #     cnt_pos_tv += 1
                        #     tv_here = True
                        # elif cnt_neg_tv < cnt_pos_tv:
                        #     cnt_neg_tv += 1
                        #     tv_here = True
                        # if tv_here:
                        # train_and_validation_data.append(arr_flatten)
                        # train_and_validation_label.append(dt_label_now)
                        if dt_start < dt_now < train_dt_end:
                            # train_here = False
                            # if dt_label_now == 1:
                            #     cnt_pos_train += 1
                            #     train_here = True
                            # elif cnt_neg_train < cnt_pos_train:
                            #     cnt_neg_train += 1
                            #     train_here = True
                            # if train_here:
                            train_data.append(arr_flatten)
                            train_aux_data.append(aux_data)
                            train_label.append(dt_label_now)
                        elif train_dt_end < dt_now < validation_dt_end:
                            # validate_here = False
                            # if dt_label_now == 1:
                            #     cnt_pos_val += 1
                            #     validate_here = True
                            # elif cnt_neg_val < cnt_pos_val:
                            #     cnt_neg_val += 1
                            #     validate_here = True
                            # if validate_here:
                            validation_data.append(arr_flatten)
                            val_aux_data.append(aux_data)
                            validation_label.append(dt_label_now)
                else:
                    # test_here = False
                    # if dt_label_now == 1:
                    #     cnt_pos_test += 1
                    #     test_here = True
                    # elif cnt_neg_test < cnt_pos_test:
                    #     cnt_neg_test += 1
                    #     test_here = True
                    # if test_here:
                    test_data.append(arr_flatten)
                    test_aux_data.append(aux_data)
                    test_label.append(dt_label_now)

                # if is_tsne:
                #     if i_t in top_indexs:
                #         all_data.append(arr_flatten)
                #         all_label.append(dt_label_now)

            #balancing

            # if dt_start < dt_now < validation_dt_end:
            #     for i_t in range(height * width):
            #         dt_label_now = data_arr[i_t]
            #         if dt_start < dt_now < train_dt_end:
            #             if dt_label_now == 1:
            #                 cnt_pos_train += 1
            #                 train_data.append(data_for_now[i_t, :])
            #                 train_label.append(data_arr[i_t])
            #             elif cnt_neg_train < cnt_pos_train:
            #                 cnt_neg_train += 1
            #                 train_data.append(data_for_now[i_t, :])
            #                 train_label.append(data_arr[i_t])
            #
            #         elif train_dt_end < dt_now < validation_dt_end:
            #             if dt_label_now == 1:
            #                 cnt_pos_val += 1
            #                 validation_data.append(data_for_now[i_t, :])
            #                 validation_label.append(data_arr[i_t])
            #             elif cnt_neg_val < cnt_pos_val:
            #                 cnt_neg_val += 1
            #                 validation_data.append(data_for_now[i_t, :])
            #                 validation_label.append(data_arr[i_t])
            #         if dt_label_now == 1:
            #             cnt_pos_tv += 1
            #             train_and_validation_data.append(data_for_now[i_t, :])
            #             train_and_validation_label.append(data_arr[i_t])
            #         elif cnt_neg_tv < cnt_pos_tv:
            #             cnt_neg_tv += 1
            #             train_and_validation_data.append(data_for_now[i_t, :])
            #             train_and_validation_label.append(data_arr[i_t])
            # # else:
            #     for i_t in range(height * width):
            #         dt_label_now = data_arr[i_t]
            #         if dt_label_now == 1:
            #             cnt_pos_test += 1
            #             test_data.append(data_for_now[i_t, :])
            #             test_label.append(data_arr[i_t])
            #         elif cnt_neg_test < cnt_pos_test:
            #             cnt_neg_test += 1
            #             test_data.append(data_for_now[i_t, :])
            #             test_label.append(data_arr[i_t])
            # else:
            #     for i_t in range(height * width):
            #         test_data.append(data_for_now[i_t, :])
            #         test_label.append(data_arr[i_t])
            print "finish %s" % dt_str

    print "pos_train %d, neg_train %d" %(cnt_pos_train, cnt_neg_train)
    print "pos_val %d, neg_val %d" %(cnt_pos_val, cnt_neg_val)
    print "pos_test %d, neg_test %d" %(cnt_pos_test, cnt_neg_test)
    print "pos_tv %d, neg_tv %d" %(cnt_pos_tv, cnt_neg_tv)
    if not flatten:
        return [train_data, validation_data, test_data, train_aux_data, val_aux_data, test_aux_data, train_label, validation_label, test_label]
    else:
        return [train_data, validation_data, test_data, train_and_validation_data , train_label, validation_label, test_label, train_and_validation_label]



#获取调休日和节假日(3天,7天节假日)对应的数据
def get_holiday_and_tiaoxiu_data_for_train(dt_start, dt_end,time_interval, spatial_interval, n, n_d, n_w):
    print "start get holiday and tiaoxiu data for train"
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
            if dt_start <= dt_tx_st < dt_end:
                tx_dt_list.append(dt_tx_st)
                tx_list_idx[dt_tx_st.strftime(second_format)] = t_i
                dt_tx_st += datetime.timedelta(minutes=time_interval)
            elif dt_tx_st >= dt_end:
                break
    for dt_tiaoxiu in tx_dt_list:
        time_now = dt_tiaoxiu
        time_now_str = time_now.strftime(second_format)
        # print "tiaoxiu : %s" % time_now_str
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
                if dt_hl_st >= dt_end:
                    break
                dt_hl_ed = datetime.datetime.strptime(date_hl + " 23:59:59", second_format)
                while dt_hl_st < dt_hl_ed:
                    if dt_start <= dt_hl_st < dt_end:
                        dt_list.append(dt_hl_st)
                        dt_hl_st += datetime.timedelta(minutes=time_interval)
                    elif dt_hl_st >= dt_end:
                        break
                for dt_now in dt_list:
                    time_now_str = dt_now.strftime(second_format)
                    # print "holiday: %s" % time_now_str
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
def color_all_rects_with_segments(points, outfile_path, spatial_interval,d_lat,d_lng,n_lat, n_lng):
    if spatial_interval == 500:
        d_lat = 0.0042
        d_lng = 0.006
    else:
        d_lat = 0.0084
        d_lng = 0.012

    n_lat_delta_origin = (max_lat - min_lat)/d_lat
    n_lat_delta = int(math.ceil(n_lat_delta_origin)) + 1

    n_lng_delta_origin = (max_lng - min_lng)/d_lng + 1
    n_lng_delta = int(math.ceil(n_lng_delta_origin))

    lng_coors = [min_lng + i * d_lng for i in range(n_lng_delta)]
    lat_coors = [min_lat + i * d_lat for i in range(n_lat_delta)]
    n_lng = len(lng_coors)-1
    n_lat = len(lat_coors)-1

    grid_colors = [0 for it in range(n_lat * n_lng)]

    for item in points:
        [start_point, end_point] = item
        segment_overlap_with_ids = query_rect_segment_in(start_point, end_point, spatial_interval,d_lat,d_lng,n_lat,n_lng)
        for id_tmp in segment_overlap_with_ids:
            grid_colors[id_tmp] = 1

    out_str_tmp = ""
    for i_lat in range(len(lat_coors)-1):
        for j_lng in range(len(lng_coors)-1):

            id_tmp = j_lng * (len(lat_coors)-1) + i_lat
            color = "red" if grid_colors[id_tmp] else ""

            min_lng1 = lng_coors[j_lng]
            max_lng1 = lng_coors[j_lng + 1]
            min_lat1 = lat_coors[i_lat]
            max_lat1 = lat_coors[i_lat + 1]

            # center_lng = (min_lng1 + max_lng1)/2.0
            # center_lat = (min_lat1 + max_lat1)/2.0

            out_str ='''var rectangle_'''+str(id_tmp)+''' = new BMap.Polygon([
                            new BMap.Point(''' + str(min_lng1) + ''',''' + str(min_lat1) + '''),
                            new BMap.Point(''' + str(max_lng1) + ''',''' + str(min_lat1) + '''),
                            new BMap.Point(''' + str(max_lng1) + ''',''' + str(max_lat1) + '''),
                            new BMap.Point(''' + str(min_lng1) + ''',''' + str(max_lat1) + ''')
                        ], {strokeColor:"red", strokeWeight:1, strokeOpacity:1,fillColor:"'''+str(color)+'''"});\n
                        map.addOverlay(rectangle_'''+str(id_tmp)+''');\n'''
                        # var point_'''+str(id_tmp)+''' = new BMap.Point(''' + str(center_lng) + ''',''' + str(center_lat) + ''');\n
                        # var marker_'''+str(id_tmp)+''' = new BMap.Marker(point_'''+str(id_tmp)+''');\n
                        # var label_'''+str(id_tmp)+''' = new BMap.Label("'''+str(id_tmp)+'''", {position: point_'''+str(id_tmp)+''',offset: new BMap.Size(20, -10)});\n
                        # label_'''+str(id_tmp)+'''.setStyle({color: "black",fontSize: "12px",border: "0",backgroundColor: "0.0"});\n
                        # marker_'''+str(id_tmp)+'''.setLabel(label_'''+str(id_tmp)+''');\n
                        # map.addOverlay(marker_'''+str(id_tmp)+''');'''
            out_str_tmp += out_str

    output_file = open(outfile_path,"w")
    output_file.write(out_str_tmp)
    output_file.close()



#输入起始点,终止点,返回一个候选的rect id列表
def query_rect_segment_in(start_point, end_point, spatial_interval,d_lat,d_lng,n_lat,n_lng):

    segment_overlap_with_ids = []
    #获取所有矩形
    output_file_path = ""
    rect_dict = get_all_rects_of_beijing(spatial_interval,output_file_path)
    size_of_dict = len(rect_dict)

    min_lng1 = min(start_point.X, end_point.X)
    max_lng1 = max(start_point.X, end_point.X)

    min_lng2 = int(math.floor((min_lng1 - min_lng)/d_lng)) * d_lng + min_lng
    max_lng2 = int(math.ceil((max_lng1 - min_lng)/d_lng)) * d_lng + min_lng

    min_lat1 = min(start_point.Y, end_point.Y)
    max_lat1 = max(start_point.Y, end_point.Y)

    min_lat2 = int(math.floor((min_lat1 - min_lat)/d_lat)) * d_lat + min_lat
    max_lat2 = int(math.ceil((max_lat1 - min_lat)/d_lat)) * d_lat + min_lat

    lng_lat_list = [[start_point.X, start_point.Y],[end_point.X, end_point.Y]]

    keys_to_detect = []

    for key in range(size_of_dict):
        rect = rect_dict[key]

        l_lng = rect.LeftTop.X
        r_lng = rect.RightBottom.X

        up_lat = rect.LeftTop.Y
        down_lat = rect.RightBottom.Y

        #夹在两个点中间,不相交
        if (((min_lng1 < l_lng < max_lng1) and ( min_lng1 < r_lng < max_lng1 )) or ( (min_lat1 < up_lat < max_lat1) and (min_lat1 < down_lat < max_lat1))) and (l_lng >= min_lng2 and r_lng <= max_lng2 and up_lat <= max_lat2 and down_lat >= min_lat2):
            keys_to_detect.append(key)

    for key in keys_to_detect:
        rect = rect_dict[key]
        if CheckRectLine(start_point, end_point, rect):
            segment_overlap_with_ids.append(key)

    for item in lng_lat_list:
        [lng, lat] = item
        if (not (min_lng <= lng and lng <= max_lng and min_lat <= lat and lat <= max_lat)):
            continue
        else:

            j_lat_origin = (float(lat) - min_lat)/d_lat
            j_lat = int(math.ceil(j_lat_origin)) - 1
            i_lng_origin = (float(lng) - min_lng)/d_lng
            i_lng = int(math.ceil(i_lng_origin)) - 1
            id = i_lng * n_lat + j_lat
            segment_overlap_with_ids.append(id)

    return segment_overlap_with_ids

#输入空间间隔,返回北京市所有网格字典
def get_all_rects_of_beijing(spatial_interval, output_file_path):
    if spatial_interval == 500:
        d_lat = 0.0042
        d_lng = 0.006
    else:
        d_lat = 0.0084
        d_lng = 0.012
    n_lat_delta_origin = (max_lat - min_lat)/d_lat
    n_lat_delta = int(math.ceil(n_lat_delta_origin)) + 1

    n_lng_delta_origin = (max_lng - min_lng)/d_lng + 1
    n_lng_delta = int(math.ceil(n_lng_delta_origin))

    lng_coors = [min_lng + i * d_lng for i in range(n_lng_delta)]
    lat_coors = [min_lat + i * d_lat for i in range(n_lat_delta)]
    n_lng = len(lng_coors)-1
    n_lat = len(lat_coors)-1
    print "grid size = lng: %d * lat: %d\n" % (n_lng, n_lat)
    rect_dict = generate_grid_for_beijing(lng_coors, lat_coors, output_file_path)

    return rect_dict

def generate_grid_for_beijing(lng_coors, lat_coors,output_file_path):
    rects = {}
    out_str_tmp = ""
    for i_lat in range(len(lat_coors)-1):
        for j_lng in range(len(lng_coors)-1):

            id = j_lng * (len(lat_coors)-1) + i_lat

            min_lng1 = lng_coors[j_lng]
            max_lng1 = lng_coors[j_lng + 1]
            min_lat1 = lat_coors[i_lat]
            max_lat1 = lat_coors[i_lat + 1]

            left_top = Vector2(min_lng1, max_lat1)
            right_bottom = Vector2(max_lng1, min_lat1)
            rect = Rect(left_top,right_bottom)
            rects[id] = rect

            out_str ='''var rectangle_'''+str(id)+''' = new BMap.Polygon([
                            new BMap.Point(''' + str(min_lng1) + ''',''' + str(min_lat1) + '''),
                            new BMap.Point(''' + str(max_lng1) + ''',''' + str(min_lat1) + '''),
                            new BMap.Point(''' + str(max_lng1) + ''',''' + str(max_lat1) + '''),
                            new BMap.Point(''' + str(min_lng1) + ''',''' + str(max_lat1) + ''')
                        ], {strokeColor:"red", strokeWeight:1, strokeOpacity:1,fillColor:''});\n
                        map.addOverlay(rectangle_'''+str(id)+''');\n'''
            out_str_tmp += out_str


    if output_file_path!="":
        output_file = open(output_file_path,"w")
        output_file.write(out_str_tmp)
        output_file.close()
    return rects

def generate_polylines_for_beijing(lng_coors, lat_coors,output_file_path,min_lat1,max_lat1,min_lng1,max_lng1):
    output_file = open(output_file_path,"w")
    cnt = 0
    for i_lng in range(len(lng_coors)):
        now_lng = lng_coors[i_lng]
        cnt += 1
        out_str ='''var polyline_'''+str(cnt)+''' = new BMap.Polyline([
                        new BMap.Point(''' + str(now_lng) + ''', ''' + str(min_lat1) + '''),
                        new BMap.Point(''' + str(now_lng) + ''', ''' + str(max_lat1) + ''')
                    ], {strokeColor:"red", strokeWeight:1, strokeOpacity:1,fillColor:''});  \n
                    map.addOverlay(polyline_'''+str(cnt)+''');\n'''
        output_file.write(out_str)
    for i_lat in range(len(lat_coors)):
        now_lat = lat_coors[i_lat]
        cnt += 1
        out_str ='''var polyline_'''+str(cnt)+''' = new BMap.Polyline([
                        new BMap.Point(''' + str(min_lng1) + ''', ''' + str(now_lat) + '''),
                        new BMap.Point(''' + str(max_lng1) + ''', ''' + str(now_lat) + ''')
                    ], {strokeColor:"red", strokeWeight:1, strokeOpacity:1,fillColor:''});  \n
                    map.addOverlay(polyline_'''+str(cnt)+''');\n'''
        output_file.write(out_str)

    output_file.close()
def generate_grid_timelines_for_beijing(datetime_query,lng_coors,lat_coors,out_data_file,sep,time_interval):
    query_of_accidents = Accidents_Array.objects.filter(spatial_interval=sep,time_interval=time_interval,create_time=datetime_query)
    print "generate_grid_timelines_for_beijing %s" % str(datetime_query)
    print "out_data_file %s" % out_data_file
    if len(query_of_accidents):
        print "time_interval %d" % time_interval
        print "sep %d" % sep
        print "len of query_of_accidents %d" % len(query_of_accidents)
        print "query_of_accidents[0].content\n %s" % query_of_accidents[0].content

        accidents_arr = [int(item) for item in query_of_accidents[0].content.split(",")]

        output_file = open(out_data_file,"w")
        n_lat = len(lat_coors)-1
        n_lng = len(lng_coors)-1

        for it in range(n_lng * n_lat):
            out_str = "delete rectangle_" + str(it) + ";\n"
            output_file.write(out_str)

        for i_lng in range(n_lng):
            for j_lat in range(n_lat):
                id = i_lng * n_lat + j_lat
                min_lng1 = lng_coors[i_lng]
                max_lng1 = lng_coors[i_lng + 1]
                min_lat1 = lat_coors[j_lat]
                max_lat1 = lat_coors[j_lat + 1]
                # center_lng = (min_lng1 + max_lng1)/2.0
                # center_lat = (min_lat1 + max_lat1)/2.0
                accident_cnt_of_id = accidents_arr[id]

                if accident_cnt_of_id == 0:
                    color = 'white'
                elif accident_cnt_of_id == 1:
                    color = 'orange'
                else:
                    color = 'red'
                out_str ='''var rectangle_'''+str(id)+''' = new BMap.Polygon([
                                new BMap.Point(''' + str(min_lng1) + ''',''' + str(min_lat1) + '''),
                                new BMap.Point(''' + str(max_lng1) + ''',''' + str(min_lat1) + '''),
                                new BMap.Point(''' + str(max_lng1) + ''',''' + str(max_lat1) + '''),
                                new BMap.Point(''' + str(min_lng1) + ''',''' + str(max_lat1) + ''')
                            ], {strokeColor:"red", strokeWeight:1, strokeOpacity:1,fillColor:"'''+color+'''",fillOpacity:0.5});\n
                            map.addOverlay(rectangle_'''+str(id)+''');\n'''
                            # var point_'''+str(id)+''' = new BMap.Point(''' + str(center_lng) + ''',''' + str(center_lat) + ''');\n
                            # var marker_'''+str(id)+''' = new BMap.Marker(point_'''+str(id)+''');\n
                            # var label_'''+str(id)+''' = new BMap.Label("'''+str(accident_cnt_of_id)+'''", {position: point_'''+str(id)+''',offset: new BMap.Size(20, -10)});\n
                            # label_'''+str(id)+'''.setStyle({color: "black",fontSize: "12px",border: "0",backgroundColor: "0.0"});\n
                            # marker_'''+str(id)+'''.setLabel(label_'''+str(id)+''');\n
                            # map.addOverlay(marker_'''+str(id)+''');'''
                output_file.write(out_str)
        output_file.close()
        return 0
    else:
        return -1
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

                j_lat_origin = (float(lat) - min_lat)/d_lat
                j_lat = int(math.ceil(j_lat_origin)) - 1
                i_lng_origin = (float(lng) - min_lng)/d_lng
                i_lng = int(math.ceil(i_lng_origin)) - 1

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
    #n_lat和n_lng为纬度和经度网格的数量
    geo_cnts = [0 for i in range(n_lat * n_lng)]
    for geo_point in geo_points:
        lng = geo_point[0]
        lat = geo_point[1]

        if (not (min_lng <= lng and lng <= max_lng and min_lat <= lat and lat <= max_lat)):
            continue
        else:
            j_lat_origin = (float(lat) - min_lat)/d_lat
            j_lat = int(math.ceil(j_lat_origin)) - 1
            i_lng_origin = (float(lng) - min_lng)/d_lng
            i_lng = int(math.ceil(i_lng_origin)) - 1
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


def get_grid_timeline(datetime_query,out_data_file, sep= 1000,time_interval = 60):
    if sep == 500:
        d_lat = 0.0042
        d_lng = 0.006
    else:
        d_lat = 0.0084
        d_lng = 0.012
    n_lat_delta_origin = (max_lat - min_lat)/d_lat
    n_lat_delta = int(math.ceil(n_lat_delta_origin)) + 1
    n_lng_delta_origin = (max_lng - min_lng)/d_lng + 1
    n_lng_delta = int(math.ceil(n_lng_delta_origin))

    lng_coors = [min_lng + i * d_lng for i in range(n_lng_delta)]
    lat_coors = [min_lat + i * d_lat for i in range(n_lat_delta)]
    n_lng = len(lng_coors)-1
    n_lat = len(lat_coors)-1
    print "grid size = lng: %d * lat: %d\n" % (n_lng, n_lat)

    generate_grid_timelines_for_beijing(datetime_query,lng_coors,lat_coors,out_data_file,sep,time_interval)#,min_lat,max_lat,min_lng,max_lng)
    print 'min_lat: %f, max_lat: %f, min_lng: %f , max_lng: %f\n' % (min_lat, max_lat, min_lng, max_lng)
    return min_lat,max_lat,min_lng,max_lng
def region_difference_calc(start_time, end_time, time_interval,spatial_interval, out_file_path, slice = -1):
    if spatial_interval == 500:
        d_lat = 0.0042
        d_lng = 0.006
    else:
        d_lat = 0.0084
        d_lng = 0.012
    n_lat_delta_origin = (max_lat - min_lat)/d_lat
    n_lat_delta = int(math.ceil(n_lat_delta_origin)) + 1
    n_lng_delta_origin = (max_lng - min_lng)/d_lng + 1
    n_lng_delta = int(math.ceil(n_lng_delta_origin))

    lng_coors = [min_lng + i * d_lng for i in range(n_lng_delta)]
    lat_coors = [min_lat + i * d_lat for i in range(n_lat_delta)]
    n_lng = len(lng_coors)-1
    n_lat = len(lat_coors)-1
    print "grid size = lng: %d * lat: %d\n" % (n_lng, n_lat)

    query_of_accidents = Accidents_Array.objects.filter(spatial_interval=spatial_interval,time_interval=time_interval,create_time__range=[start_time,end_time])

    if len(query_of_accidents):
        print "time_interval %d" % time_interval
        print "spatial_interval %d" % spatial_interval
        print "len of query_of_accidents %d" % len(query_of_accidents)

        numpy_sum_accidents = np.zeros(n_lng * n_lat)
        output_file = open(out_file_path,"w")
        for accident in query_of_accidents:
            # zero_c = datetime.datetime.strptime("2016-08-12 19:00:00",second_format)
            # one_c = datetime.datetime.strptime("2016-08-12 21:30:00",second_format)
            # if zero_c<= accident.create_time<= one_c:
            #     print accident.create_time.strftime(second_format)
            # print "now adding %s" % accident.create_time.strftime(second_format)
            accidents_arr = np.array([int(item) for item in accident.content.split(",")])
            numpy_sum_accidents = numpy_sum_accidents + accidents_arr

        max_accident_cnt = numpy_sum_accidents.max()
        color_array = ((numpy_sum_accidents / max_accident_cnt) * 255.0).astype(int)
        # print color_array

        sort_indexs = np.argsort(numpy_sum_accidents)

        top_n_index = list(sort_indexs[0:slice])
        top_n_index.extend(list(sort_indexs[-slice-1:-1]))
        region_normed_accidents_prob = numpy_sum_accidents / max_accident_cnt
        if slice > 0:
            return [top_n_index,region_normed_accidents_prob]

        n_lat = len(lat_coors)-1
        n_lng = len(lng_coors)-1

        # for it in range(n_lng * n_lat):
        #     out_str = "delete rectangle_" + str(it) + ";\n"
        #     output_file.write(out_str)

        for i_lng in range(n_lng):
            for j_lat in range(n_lat):
                id = i_lng * n_lat + j_lat
                min_lng1 = lng_coors[i_lng]
                max_lng1 = lng_coors[i_lng + 1]
                min_lat1 = lat_coors[j_lat]
                max_lat1 = lat_coors[j_lat + 1]
                center_lng = (min_lng1 + max_lng1)/2.0
                center_lat = (min_lat1 + max_lat1)/2.0

                # color_i = "rgb("+str(color_array[id])+", 0, 0)"
                # if color_array[id] < 100:
                #     color_value = 0
                # else:
                color_value = color_array[id]
                color_i = "gradient["+str(color_value)+"]"
                fill_capacity = 1.0
                # if color_value > 0:
                #     fill_capacity = 1.0
                # else:
                #     fill_capacity = 0.2

                out_str ='''var rectangle_'''+str(id)+''' = new BMap.Polygon([
                                new BMap.Point(''' + str(min_lng1) + ''',''' + str(min_lat1) + '''),
                                new BMap.Point(''' + str(max_lng1) + ''',''' + str(min_lat1) + '''),
                                new BMap.Point(''' + str(max_lng1) + ''',''' + str(max_lat1) + '''),
                                new BMap.Point(''' + str(min_lng1) + ''',''' + str(max_lat1) + ''')
                            ], {strokeColor:"black", strokeWeight:0.5, strokeOpacity:0.2,fillColor:'''+color_i+''',fillOpacity:'''+str(fill_capacity)+'''});\n
                            map.addOverlay(rectangle_'''+str(id)+''');\n'''
                            # var point_'''+str(id)+''' = new BMap.Point(''' + str(center_lng) + ''',''' + str(center_lat) + ''');\n
                            # var marker_'''+str(id)+''' = new BMap.Marker(point_'''+str(id)+''');\n
                            # var label_'''+str(id)+''' = new BMap.Label("'''+str(numpy_sum_accidents[id])+'''", {position: point_'''+str(id)+''',offset: new BMap.Size(20, -10)});\n
                            # label_'''+str(id)+'''.setStyle({color: "black",fontSize: "12px",border: "0",backgroundColor: "0.0"});\n
                            # marker_'''+str(id)+'''.setLabel(label_'''+str(id)+''');\n
                            # map.addOverlay(marker_'''+str(id)+''');'''
                output_file.write(out_str)
        output_file.close()
        print "finish!"
    return 0
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
    n_lat_delta_origin = (max_lat - min_lat)/d_lat
    n_lat_delta = int(math.ceil(n_lat_delta_origin)) + 1
    n_lng_delta_origin = (max_lng - min_lng)/d_lng + 1
    n_lng_delta = int(math.ceil(n_lng_delta_origin))

    lng_coors = [min_lng + i * d_lng for i in range(n_lng_delta)]
    lat_coors = [min_lat + i * d_lat for i in range(n_lat_delta)]
    n_lng = len(lng_coors)-1
    n_lat = len(lat_coors)-1
    print "grid size = lng: %d * lat: %d\n" % (n_lng, n_lat)
    generate_grid_ids(lng_coors,lat_coors)
    generate_grid_for_beijing(lng_coors, lat_coors, output_file_path)
    # generate_polylines_for_beijing(lng_coors,lat_coors,output_file_path,min_lat,lat_coors[n_lat],min_lng,lng_coors[n_lng])
    print 'min_lat: %f, max_lat: %f, min_lng: %f , max_lng: %f\n' % (min_lat, lat_coors[n_lat], min_lng, lng_coors[n_lng])
    return min_lat,max_lat,min_lng,max_lng

def get_all_dt_in_call_incidences_db(start_time,end_time,time_interval=60):
    tmp_dt = start_time
    ret_list = []

    while tmp_dt < end_time:
        ret_list.append(tmp_dt.strftime(second_format))
        tmp_dt += datetime.timedelta(minutes=time_interval)
    return ret_list

#获得指定时间段的事故,并写道文件中
def get_call_incidences(dt_start, dt_end):
    call_incidences = Call_Incidence.objects.filter(create_time__range=[dt_start,dt_end])
    file_to_wrt_path = BASE_DIR + os.sep + "static" + os.sep + "points.json"
    file_to_wrt = open(file_to_wrt_path,"w")

    call_incidences_to_dump = []
    for call_incidence in call_incidences:
        call_incidence_tmp = {}
        call_incidence_tmp["lng"] = call_incidence.longitude
        call_incidence_tmp["lat"] = call_incidence.latitude
        call_incidence_tmp["place"] = call_incidence.place
        call_incidence_tmp["create_time"] = call_incidence.create_time
        call_incidences_to_dump.append(call_incidence_tmp)

    js_str = simplejson.dumps(call_incidences_to_dump, use_decimal=True,cls=DatetimeJSONEncoder)
    file_to_wrt.write(js_str)

    file_to_wrt.close()
def get_call_accidents_count_for_each_hour(dt_start, dt_end, delta_t = 60):
    acc_arr = []
    dt_now = dt_start
    while (dt_now <= dt_end):
        call_incidences = Call_Incidence.objects.filter(create_time__range=[dt_now, dt_now + datetime.timedelta(minutes=delta_t)])
        acc_arr.append(dt_now.strftime(minute_format) +"\t" + str(len(call_incidences)))
        print "%s\t %d" %(dt_now.strftime(minute_format), len(call_incidences))
        dt_now = dt_now + datetime.timedelta(minutes=delta_t)

    file_to_wrt_path = BASE_DIR + os.sep + "static" + os.sep + "accident_count.tsv"
    with open(file_to_wrt_path,"w") as file_to_wrt:
        file_to_wrt.write("\t".join(acc_arr))
        print "write %s successful" % file_to_wrt_path