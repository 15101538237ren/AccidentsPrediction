# coding: utf-8
from django.shortcuts import render_to_response
from django.template import RequestContext
from import_data import *
from util import *
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
    weather_file = '/Users/Ren/PycharmProjects/AccidentsPrediction/preprocessing/data/weather.csv'
    import_weather_to_db(weather_file)
    air_file = '/Users/Ren/PycharmProjects/AccidentsPrediction/preprocessing/data/air_houbao.csv'
    import_air_quality_to_db(air_file)
def index(request):
    return render_to_response('prep/index.html', locals(), context_instance=RequestContext(request))
def grid(request):
    out_data_file = '/Users/Ren/PycharmProjects/AccidentsPrediction/static/js/grid_polyline.js'
    sep = 500
    min_lat,max_lat,min_lng,max_lng = get_liuhuan_poi(out_data_file, sep= sep)
    return render_to_response('prep/grid.html', locals(), context_instance=RequestContext(request))
def timeline(request):
    # outpkl_file_path = '/Users/Ren/PycharmProjects/AccidentsPrediction/preprocessing/data/accidents.pkl'
    #partition_geopoints_by_time(outpkl_file_path)
    #get_all_accidents_from_db(outpkl_file_path)
    keywords = [u"交通设施", u"住宿", u"医院", u"商务住宅_公司", u"商场超市", u"娱乐场所", u"学校", u"旅游景点", u"生活服务", u"自住住宅", u"银行金融", u"餐饮"]
    input_file_list = []
    #keyword = u"医院"
    #code = '0901|0902'
    #industry_type = u"cater"
    for keyword  in keywords:
        input_file_list.append(u"/Users/Ren/PycharmProjects/AccidentsPrediction/preprocessing/data/poi/" + keyword + u".csv")
    #get_pois_from_baidu(keyword, industry_type, output_file)
    #get_pois_from_gaode(code, output_file)
    out_js = '/Users/Ren/PycharmProjects/AccidentsPrediction/static/js/point_collection.js'
    #convert_point_to_point_collection(output_file, out_js)

    param_500 = {}
    param_500['d_len'] = 500
    param_500['d_lat'] = 0.0042
    param_500['d_lng'] = 0.006
    param_500['n_lng'] = 56
    param_500['n_lat'] = 62

    param_1000 = {}
    param_1000['d_len'] = 1000
    param_1000['d_lat'] = 0.0084
    param_1000['d_lng'] = 0.012
    param_1000['n_lng'] = 28
    param_1000['n_lat'] = 31
    # label_all_function_regions(input_file_list,**param_1000)
    #label_all_accidents(outpkl_file_path, 20, **param_1000)
    #get_work_day_data(work_day_bounds,time_interval=60, spatial_interval=1000)
    # get_holiday_and_tiaoxiu_data_for_train(time_interval=30, spatial_interval=1000, n = 5, n_d = 3, n_w = 4)
    dt_start = datetime.datetime.strptime("2016-01-13 00:00:00", second_format)
    dt_end = datetime.datetime.strptime("2017-02-10 23:59:59",second_format)
    time_interval = 60
    spatial_interval = 1000
    outpkl_file_path = '/Users/Ren/PycharmProjects/AccidentsPrediction/preprocessing/data/lstm_data_'+dt_start.strftime(date_format)+'_'+dt_end.strftime(date_format)+'_'+str(time_interval)+'_'+str(spatial_interval)+'.pkl'
    prepare_lstm_data(outpkl_file_path,dt_start, dt_end, time_interval= time_interval, n=5, n_d= 3, n_w=3, **param_1000)
    return render_to_response('prep/timeline.html', locals(), context_instance=RequestContext(request))