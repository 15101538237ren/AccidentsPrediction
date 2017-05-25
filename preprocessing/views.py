# coding: utf-8
from django.shortcuts import render_to_response
from django.template import RequestContext
from django.http import JsonResponse
from import_data import *
from util import *
from AccidentsPrediction.settings import BASE_DIR
from correlation_analysis import get_all_data_for_analysis
from classifier import *
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
# Create your views here.
def import_data_to_db():
    i = 11
    input_call_incidence_file = "/Users/Ren/PycharmProjects/PoliceIndex/beijing_data/2016_accidents/"+str(i)+".xls"
    import_call_incidence_data_of_2016(input_call_incidence_file=input_call_incidence_file)
    i_list=[2]
    for i in i_list:
        input_call_incidence_file = "/Users/Ren/PycharmProjects/PoliceIndex/beijing_data/2017/shuju/122_17_0"+str(i)+"_cleaned.xls"
        import_call_incidence_data_of_2016(input_call_incidence_file=input_call_incidence_file)

    input_violation_file = '/Users/Ren/Desktop/violation.csv'
    import_violation_data(input_violation_file)
    weather_file = BASE_DIR+'/preprocessing/data/weather.csv'
    import_weather_to_db(weather_file)
    air_file = BASE_DIR+'/preprocessing/data/air_houbao.csv'
    import_air_quality_to_db(air_file)
def index(request):
    outpkl_file_path = BASE_DIR+'/preprocessing/data/accidents.pkl'
    #partition_geopoints_by_time(outpkl_file_path)
    # get_all_accidents_from_db(outpkl_file_path)

    keywords = [u"交通设施", u"住宿", u"医院", u"商务住宅_公司", u"商场超市", u"娱乐场所", u"学校", u"旅游景点", u"生活服务", u"自住住宅", u"银行金融", u"餐饮"]
    input_file_list = []
    #keyword = u"医院"
    #code = '0901|0902'
    #industry_type = u"cater"
    for keyword  in keywords:
        input_file_list.append(BASE_DIR+u"/preprocessing/data/poi/" + keyword + u".csv")
    #get_pois_from_baidu(keyword, industry_type, output_file)
    #get_pois_from_gaode(code, output_file)
    out_js = BASE_DIR+'/static/js/point_collection.js'
    #convert_point_to_point_collection(output_file, out_js)

    param_500 = {}
    param_500['d_len'] = 500
    param_500['d_lat'] = 0.0042
    param_500['d_lng'] = 0.006
    param_500['n_lng'] = 58
    param_500['n_lat'] = 64

    param_1000 = {}
    param_1000['d_len'] = 1000
    param_1000['d_lat'] = 0.0084
    param_1000['d_lng'] = 0.012
    param_1000['n_lng'] = 29
    param_1000['n_lat'] = 32

    # label_all_accidents(outpkl_file_path, 60, **param_1000)
    # label_all_function_regions(input_file_list,**param_1000)
    #return render_to_response('prep/index.html', locals(), context_instance=RequestContext(request))

    #get_work_day_data(work_day_bounds,time_interval=60, spatial_interval=1000)
    # get_holiday_and_tiaoxiu_data_for_train(time_interval=30, spatial_interval=1000, n = 5, n_d = 3, n_w = 4)
    dt_start = datetime.datetime.strptime("2016-01-13 00:00:00", second_format)
    # dt_end = datetime.datetime.strptime("2016-01-31 00:00:00",second_format)
    # dt_end = datetime.datetime.strptime("2016-04-01 00:00:00",second_format)
    dt_end = datetime.datetime.strptime("2017-02-28 23:59:59",second_format)
    time_interval = 60
    spatial_interval = 1000
    print "base_dir: %s" % BASE_DIR

    outpkl_file_path = BASE_DIR + '/preprocessing/data/lstm_data_'+dt_start.strftime(date_format)+'_'+dt_end.strftime(date_format)+'_'+str(time_interval)+'_'+str(spatial_interval)+'.pkl'
    n = 12
    n_d = n_w = 5

    n_time_steps = n + (n_d + n_w ) * 2 + 2
    split_ratio = 0.7
    data_dim = 1
    class_weight = {0: 1, 1: 19}

    # get_all_data_for_analysis(dt_start,dt_end,time_interval=time_interval,n=n,n_d=n_d,n_w=n_w,**param_1000)

    # [all_data_list, all_label_list] = generate_data_for_train_and_test(outpkl_file_path,dt_start, dt_end, time_interval= time_interval, n = n, n_d = n_d, n_w = n_w, **param_1000)
    #
    # print('Original dataset shape {}'.format(Counter(all_label_list)))
    #
    # # train_and_test_model_with_gru(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio,class_weight=class_weight)
    # train_and_test_model_with_lstm(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio,class_weight=class_weight)
    # train_and_test_model_with_keras_logistic_regression(data_dim, n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio,class_weight=class_weight)
    # rus = RandomUnderSampler(random_state=42)
    #
    # X_res, y_res = rus.fit_sample(all_data_list, all_label_list)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    # train_and_test_model_with_gru(data_dim,n_time_steps, X_res,y_res, split_ratio=split_ratio,class_weight=class_weight)
    # train_and_test_model_with_lstm(data_dim,n_time_steps,X_res,y_res, split_ratio=split_ratio,class_weight=class_weight)


    # all_data_list_flatten = [item.flatten() for item in all_data_list]
    # all_data_list = all_data_list_flatten

    # no_of_pca = 1
    # pca = PCA(n_components=n_time_steps)
    # all_data_list = pca.fit_transform(all_data_list_flatten)
    #
    # train_and_test_model_with_lda(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio)
    # train_and_test_model_with_qda(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio)
    # train_and_test_model_with_logistic_regression(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio,class_weight=class_weight)

    # train_and_test_model_with_ada_boost(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio)
    # # train_and_test_model_with_bagging(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio)
    # train_and_test_model_with_random_forest(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio,class_weight=class_weight)
    # train_and_test_model_with_gradient_boosting(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio)
    # # train_and_test_model_with_extra_tree(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio,class_weight=class_weight)
    # train_and_test_model_with_decision_tree(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio,class_weight=class_weight)
    # train_and_test_model_with_svm(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio,class_weight=class_weight)
    #
    # train_and_test_model_with_dense_network(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio,class_weight=class_weight)
    # train_and_test_model_with_2_layer_dense_network(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio,class_weight=class_weight)
    #
    # # print'Resampled of Traditional methods'
    # X_res, y_res = rus.fit_sample(all_data_list, all_label_list)
    #
    # train_and_test_model_with_keras_logistic_regression(data_dim, n_time_steps, X_res,y_res, split_ratio=split_ratio,class_weight=class_weight)
    # train_and_test_model_with_ada_boost(data_dim,n_time_steps,  X_res,y_res,  split_ratio=split_ratio)
    # # train_and_test_model_with_bagging(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio)
    # train_and_test_model_with_random_forest(data_dim,n_time_steps,  X_res,y_res,  split_ratio=split_ratio,class_weight=class_weight)
    # train_and_test_model_with_gradient_boosting(data_dim,n_time_steps, X_res,y_res,  split_ratio=split_ratio)
    # # train_and_test_model_with_extra_tree(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio,class_weight=class_weight)
    # train_and_test_model_with_decision_tree(data_dim,n_time_steps, X_res,y_res,  split_ratio=split_ratio,class_weight=class_weight)
    # train_and_test_model_with_svm(data_dim,n_time_steps, X_res,y_res, split_ratio=split_ratio,class_weight=class_weight)
    #
    # train_and_test_model_with_dense_network(data_dim,n_time_steps, X_res,y_res, split_ratio=split_ratio,class_weight=class_weight)
    # train_and_test_model_with_2_layer_dense_network(data_dim,n_time_steps, X_res,y_res,  split_ratio=split_ratio,class_weight=class_weight)



    # train_and_test_model_with_rbf_nu_svm(data_dim,n_time_steps, all_data_list,all_label_list, split_ratio=split_ratio)



    return render_to_response('prep/index.html', locals(), context_instance=RequestContext(request))
def grid(request):
    out_data_file = BASE_DIR+'/static/js/grid_polyline.js'
    sep = 1000
    min_lat,max_lat,min_lng,max_lng = get_liuhuan_poi(out_data_file, sep= sep)
    return render_to_response('prep/grid.html', locals(), context_instance=RequestContext(request))
#按照网格,用不同颜色显示每个网格的事故量,并用label标出事故量
def grid_timeline(request):
    time_interval = 60
    sep = 1000
    if request.method == 'GET':
        start_time = datetime.datetime.strptime("2016-01-01 00:00:00",second_format)
        end_time = datetime.datetime.strptime("2017-03-01 00:00:00",second_format)
        dt_list = get_all_dt_in_call_incidences_db(start_time,end_time,time_interval=time_interval)
        dt_start = start_time.strftime(second_format)
        slider_cnts = len(dt_list)
        return render_to_response('prep/grid_timeline.html', locals(), context_instance=RequestContext(request))
    else:
        datetime_query = request.POST.get("query_dt","2016-01-01 00:00:00")
        from_dt = datetime.datetime.strptime(datetime_query,second_format)
        end_dt = from_dt + datetime.timedelta(minutes=time_interval)

        out_data_file = BASE_DIR+'/static/js/grid_timeline.js'

        min_lat,max_lat,min_lng,max_lng = get_grid_timeline(datetime_query,out_data_file, sep= sep,time_interval = 60)
        addr = '/static/js/grid_timeline.js'
        response_dict = {}
        response_dict["code"] = 0
        response_dict["addr"] = addr
        return JsonResponse(response_dict)
def timeline(request):
    start_time = datetime.datetime.strptime("2016-01-01 00:00:00",second_format)
    end_time = datetime.datetime.strptime("2017-03-01 00:00:00",second_format)
    time_interval = 60
    dt_list = get_all_dt_in_call_incidences_db(start_time,end_time,time_interval=time_interval)
    dt_start = start_time.strftime(second_format)
    slider_cnts = len(dt_list)
    print slider_cnts
    return render_to_response('prep/timeline.html', locals(), context_instance=RequestContext(request))
#获取指定时间的事故情况
@ajax_required
def query_status(request):
    time_interval = 60
    datetime_query = request.POST.get("query_dt","2016-01-01 00:00:00")
    from_dt = datetime.datetime.strptime(datetime_query,second_format)
    end_dt = from_dt + datetime.timedelta(minutes=time_interval)
    get_call_incidences(from_dt,end_dt)
    addr = '/static/points.json'
    response_dict = {}
    response_dict["code"] = 0
    response_dict["addr"] = addr
    return JsonResponse(response_dict)
