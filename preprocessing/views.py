# coding: utf-8
from django.shortcuts import render_to_response
from django.template import RequestContext
from import_data import *
from util import get_liuhuan_poi,generate_grid_for_beijing
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
    out_data_file = '/Users/Ren/PycharmProjects/AccidentsPrediction/static/js/grid_polyline.js'
    min_lat,max_lat,min_lng,max_lng = get_liuhuan_poi(out_data_file)
    return render_to_response('prep/index.html', locals(), context_instance=RequestContext(request))