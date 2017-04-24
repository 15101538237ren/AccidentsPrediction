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
    air_file = '/Users/Ren/PycharmProjects/AccidentsPrediction/preprocessing/data/air.csv'
    import_air_quality_to_db(air_file)
def index(request):
    return render_to_response('prep/index.html', locals(), context_instance=RequestContext(request))
def grid(request):
    out_data_file = '/Users/Ren/PycharmProjects/AccidentsPrediction/static/js/grid_polyline.js'
    sep = 1000
    min_lat,max_lat,min_lng,max_lng = get_liuhuan_poi(out_data_file, sep= sep)
    return render_to_response('prep/grid.html', locals(), context_instance=RequestContext(request))
def timeline(request):
    outpkl_file_path = '/Users/Ren/PycharmProjects/AccidentsPrediction/preprocessing/data/accidents_1.pkl'
    #partition_geopoints_by_time(outpkl_file_path)
    #get_all_accidents_from_db(outpkl_file_path)
    #label_all_accidents(outpkl_file_path, 0.0084, 0.012, 28, 31, interval= 60, dlen=1000)
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
    label_all_function_regions(input_file_list, 0.0084, 0.012, 28, 31)
    return render_to_response('prep/timeline.html', locals(), context_instance=RequestContext(request))