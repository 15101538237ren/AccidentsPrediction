# -*- coding: utf-8 -*-
import sys,os,math,pickle,datetime,xlrd,csv,json
import numpy as np #导入Numpy
from convert import gps2gcj,gcj2bd
from class_for_shape import Vector2
from util import color_all_rects_with_segments
from models import Route_Info

def unicode_csv_reader(gbk_data, dialect=csv.excel, **kwargs):
    csv_reader = csv.reader(gbk_data, dialect=dialect, **kwargs)
    next(csv_reader, None)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]


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
        if cnt > 100:
            break
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

