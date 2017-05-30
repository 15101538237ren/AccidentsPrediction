# -*- coding: utf-8 -*-
import sys,os,math,pickle,datetime,xlrd,csv,json
import numpy as np #导入Numpy
from convert import gps2gcj,gcj2bd
from class_for_shape import Vector2
from util import color_all_rects_with_segments

def unicode_csv_reader(gbk_data, dialect=csv.excel, **kwargs):
    csv_reader = csv.reader(gbk_data, dialect=dialect, **kwargs)
    next(csv_reader, None)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]

def get_all_routes(input_file_path, outjson_file_path, out_grid_file_path, **params):
    spatial_interval = params['d_len']
    d_lat = params['d_lat']
    d_lng = params['d_lng']
    n_lat = params['n_lat']
    n_lng = params['n_lng']
    reader = unicode_csv_reader(open(input_file_path))
    points = []
    routes_dict = {}
    cnt = 0
    for route_id, route_name, route_direction, start_name, end_name, route_length, start_lon, start_lat, end_lon, end_lat, route_subid, route_preid, route_corid in reader:
        # if int(route_id) < 76:
        #     continue
        # elif int(route_id) > 76:
        #     break
        if cnt > 100:
            break
        routes_dict[route_id] = {}
        routes_dict[route_id]["route_name"] = route_name

        point_start = [float(start_lon),float(start_lat)]
        converted_p_start = gcj2bd(point_start)

        routes_dict[route_id]["start_lon"] = converted_p_start[0]
        routes_dict[route_id]["start_lat"] = converted_p_start[1]
        routes_dict[route_id]["start_name"] = start_name

        point_end = [float(end_lon),float(end_lat)]
        converted_p_end = gcj2bd(point_end)

        routes_dict[route_id]["end_lon"] = converted_p_end[0]
        routes_dict[route_id]["end_lat"] = converted_p_end[1]
        routes_dict[route_id]["end_name"] = end_name
        cnt += 1
        
        start_point = Vector2(converted_p_start[0],converted_p_start[1])
        end_point = Vector2(converted_p_end[0],converted_p_end[1])
        points.append([start_point, end_point])

    json_str = json.dumps(routes_dict)
    with open(outjson_file_path,"w") as json_file:
        json_file.write(json_str)
    print "json write success!"
    
    color_all_rects_with_segments(points, out_grid_file_path, spatial_interval,d_lat,d_lng,n_lat, n_lng)

