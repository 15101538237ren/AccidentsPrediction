# -*- coding: utf-8 -*-
import sys,os,math,pickle,datetime,xlrd,csv,json
import numpy as np #导入Numpy
from convert import gps2gcj,gcj2bd
from class_for_shape import Vector2
from util import color_all_rects_with_segments, date_new_format,second_new_format
from models import Route_Info, Route_Speed

def unicode_csv_reader(gbk_data, dialect=csv.excel, **kwargs):
    csv_reader = csv.reader(gbk_data, dialect=dialect, **kwargs)
    next(csv_reader, None)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]

#导入所有的出租车速度数据
def import_all_route_speed_to_db(input_file_dirpath, dt_start, dt_end):
    route_infos = Route_Info.objects.all().order_by("route_id")
    route_ids = [route_info.route_id for route_info in route_infos]

    dt_list = []
    dt_now = dt_start
    while dt_now < dt_end:
        dt_list.append(dt_now.strftime(date_new_format))
        dt_now += datetime.timedelta(hours=24)

    for route_id in route_ids:
        if route_id >= 1869:
            for dt in dt_list:
                file_path =  input_file_dirpath +os.sep + "R"+str(route_id)+os.sep+"R"+str(route_id)+"_"+dt+".csv"
                if os.path.exists(file_path):
                    route_file = open(file_path,"r")
                    reader = csv.reader(route_file)

                    #跳过第一行的header
                    next(reader, None)

                    for row in reader:
                        period = int(row[0])
                        avg_speed = float(row[1])
                        create_time = datetime.datetime.strptime(dt + " 00:00:00",second_new_format) + datetime.timedelta(minutes=5 * period)
                        route_speed= Route_Speed(route_id=route_id, create_time=create_time, avg_speed=avg_speed)
                        route_speed.save()

        print "finish route: %d" % route_id

#导入道路信息数据到数据库
def import_all_route_info_to_db(input_file_path):
    reader = unicode_csv_reader(open(input_file_path))
    for route_id, route_name, route_direction, start_name, end_name, route_length, start_lon, start_lat, end_lon, end_lat, route_subid, route_preid, route_corid in reader:
        point_start = [float(start_lon),float(start_lat)]
        converted_p_start = gcj2bd(point_start)

        point_end = [float(end_lon),float(end_lat)]
        converted_p_end = gcj2bd(point_end)

        route_info = Route_Info(route_id = int(route_id), route_name=route_name, start_lon = converted_p_start[0], start_lat=converted_p_start[1],
                                start_name=start_name, end_lon=converted_p_end[0], end_lat=converted_p_end[1], end_name=end_name,
                                route_direction=route_direction, route_length=int(route_length), route_subid=route_subid, route_preid=route_preid)
        route_info.save()
    print "save route_info successful!"

def get_all_routes(outjson_file_path, out_grid_file_path, **params):
    spatial_interval = params['d_len']
    d_lat = params['d_lat']
    d_lng = params['d_lng']
    n_lat = params['n_lat']
    n_lng = params['n_lng']

    points = []
    routes_dict = {}
    cnt = 0

    route_infos = Route_Info.objects.all().order_by("route_id")

    for route_info in route_infos:
        route_id = route_info.route_id
        # if int(route_id) < 76:
        #     continue
        # elif int(route_id) > 76:
        #     break
        # if cnt > 100:
        #     break
        routes_dict[route_id] = {}
        routes_dict[route_id]["route_name"] = route_info.route_name

        routes_dict[route_id]["start_lon"] = float(route_info.start_lon)
        routes_dict[route_id]["start_lat"] = float(route_info.start_lat)
        routes_dict[route_id]["start_name"] = route_info.start_name

        routes_dict[route_id]["end_lon"] = float(route_info.end_lon)
        routes_dict[route_id]["end_lat"] = float(route_info.end_lat)
        routes_dict[route_id]["end_name"] = route_info.end_name
        cnt += 1
        
        start_point = Vector2(float(route_info.start_lon), float(route_info.start_lat))
        end_point = Vector2(float(route_info.end_lon), float(route_info.end_lat))
        points.append([start_point, end_point])

    json_str = json.dumps(routes_dict)
    with open(outjson_file_path,"w") as json_file:
        json_file.write(json_str)
    print "json write success!"
    
    color_all_rects_with_segments(points, out_grid_file_path, spatial_interval,d_lat,d_lng,n_lat, n_lng)

