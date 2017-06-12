# -*- coding: utf-8 -*-
import sys,os,math,pickle,datetime,xlrd,csv,json
import numpy as np #导入Numpy
from convert import gps2gcj,gcj2bd
from class_for_shape import Vector2
from util import color_all_rects_with_segments, date_new_format,second_new_format,query_rect_segment_in,second_format,get_conv_kernal_crespond_data
from models import Route_Info, Route_Speed, Route_Related_Grid, Grid_Speed

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
        routes_dict[route_id]["valid"] = int(route_info.valid)
        routes_dict[route_id]["end_name"] = route_info.end_name
        cnt += 1
        
        start_point = Vector2(float(route_info.start_lon), float(route_info.start_lat))
        end_point = Vector2(float(route_info.end_lon), float(route_info.end_lat))
        points.append([start_point, end_point])
        # segment_overlap_with_ids = query_rect_segment_in(start_point, end_point, spatial_interval,d_lat,d_lng,n_lat,n_lng)
        # segment_overlap_with_ids_str = [str(sid) for sid in segment_overlap_with_ids]
        # str_of_ids = "" if len(segment_overlap_with_ids_str) == 0 else ",".join(segment_overlap_with_ids_str)
        # route_related_grid = Route_Related_Grid(route_id=route_id, grid_ids= str_of_ids)
        # route_related_grid.save()
        print "finish route %d" % route_id

    json_str = json.dumps(routes_dict)
    with open(outjson_file_path,"w") as json_file:
        json_file.write(json_str)
    print "json write success!"
    
    #color_all_rects_with_segments(points, out_grid_file_path, spatial_interval,d_lat,d_lng,n_lat, n_lng)
#验证并归一化相对速度
def validate_and_normalize_route():
    max_route_id = 9856

    for routeid in range(1983, max_route_id + 1):
        print "start handling route: %d" % routeid
        records = Route_Speed.objects.filter(route_id=routeid, create_time__hour=0, create_time__minute=0, create_time__second=0)
        if len(records) == 0:
            route_info = Route_Info.objects.filter(route_id=routeid)
            #标记该道路为速度非法
            route_of_invalid = route_info[0]
            route_of_invalid.valid = 0
            route_of_invalid.save()
            print "route: %d is invalid" % routeid
        else:
            avg_speed = np.array([item.avg_speed for item in records]).mean()
            records_for_modify = Route_Speed.objects.filter(route_id=routeid)
            for record in records_for_modify:
                record.relative_speed = record.avg_speed / avg_speed
                record.save()
        print "finished handling route: %d" % routeid

#创建城市速度网格
def create_grid_speed(outpkl_path,start_time, end_time, time_interval, spatial_interval, n_lat, n_lng):
    w_shape = (1, 1, 3, 3) #f,c,hw,ww
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,]).reshape(w_shape)
    b = np.array([0])
    conv_param = {'stride': 1, 'pad': 1}
    x_shape = (1, 1, n_lat, n_lng)

    #生成date_time_list
    now_time = start_time
    dt_str_list = []
    dt_list = []
    while now_time < end_time:
        tmp_now = now_time
        dt_str = now_time.strftime(second_format)
        dt_str_list.append(dt_str)
        now_time += datetime.timedelta(minutes= time_interval)
        dt_list.append([tmp_now, now_time])
    #筛选合法的route_id
    valid_route_ids = []
    out_pickle_file = open(outpkl_path,"wb")
    
    print "start filter route_id"
    first_filtered_ids = [route_info.route_id for route_info in Route_Info.objects.filter(valid=1)]

    for rid in first_filtered_ids:
        r_grid_ids = Route_Related_Grid.objects.filter(route_id=rid)[0]
        if r_grid_ids.grid_ids != "":
            valid_route_ids.append(rid)

    pickle.dump(valid_route_ids, out_pickle_file,-1)

    len_dt_str_list = len(dt_str_list)


    for it in range(len_dt_str_list):
        dt_str = dt_str_list[it]
        print "start handling %s" % dt_str

        dt_range = dt_list[it]
        print dt_range

        tmp_grid_speed_arr= [[] for i in range(n_lat * n_lng)]

        route_speeds = Route_Speed.objects.filter(create_time__range=dt_range, route_id__in=valid_route_ids)        
        
        starttime = datetime.datetime.now()
        for idx,route_speed in enumerate(route_speeds):
            if idx % 1000 == 999:
                endtime = datetime.datetime.now()
                second_of_query = (endtime - starttime).seconds
                print "route speed of %d/%d in %d seconds of %s" % (idx,len(route_speeds),second_of_query, dt_str)
                starttime = endtime
            relative_speed = route_speed.relative_speed
            route_id = route_speed.route_id
            related_grid = Route_Related_Grid.objects.filter(route_id=route_id)[0]
            related_grid_ids = [int(item) for item in related_grid.grid_ids.split(",")]
            for related_grid_id in related_grid_ids:
                tmp_grid_speed_arr[related_grid_id].append(relative_speed)
        print "finish route_speeds of %s" % dt_str
        tmp_grid_speed_final= [0.0 for i in range(n_lat * n_lng)]
        for it in range(n_lat * n_lng):
            if len(tmp_grid_speed_arr[it]):
                mean_of_speed = np.array(tmp_grid_speed_arr[it]).mean()
                tmp_grid_speed_final[it] = mean_of_speed

        data_content = np.array(tmp_grid_speed_final).reshape(x_shape)
        out_conv= get_conv_kernal_crespond_data(data_content, w, b, conv_param)

        for w_i in range(n_lng):
            for h_j in range(n_lat):
                wh_id = w_i * n_lat + h_j
                if math.fabs(tmp_grid_speed_final[wh_id]) < 1e-6:
                    item_list = []
                    for item in out_conv[0,0,h_j, w_i,:]:
                        if math.fabs(item) > 1e-6:
                            item_list.append(item)
                    if len(item_list):
                        tmp_grid_speed_final[wh_id] = np.array(item_list).mean()

        content = ",".join([str(round(float(item), 3)) for item in tmp_grid_speed_final])
        dt_now = datetime.datetime.strptime(dt_str,second_format)
        grid_speed = Grid_Speed(time_interval=time_interval, spatial_interval=spatial_interval,content=content,
                                create_time=dt_now)
        grid_speed.save()
        print "finish grid speed of %s" % dt_str
    out_pickle_file.close()

#在dt_start和dt_end间的是有问题的数据
def fix_zero_value_or_data_error_of(dt_starts, dt_ends, time_interval, spatial_interval):
    print "start get all grid speed!"
    all_grid_speeds = Grid_Speed.objects.filter(time_interval=time_interval, spatial_interval=spatial_interval).order_by("create_time")
    length_of_data = len(all_grid_speeds)

    print "finish get all grid speed!"
    len_of_dt = len(dt_starts)

    if length_of_data:
        #是否修改过对应的速度数组
        visited_arr = [0 for it in range(length_of_data)]
        label_dt_arr = [0 for it in range(length_of_data)]
        hour_of_data = [0 for it in range(length_of_data)]
        length_of_grid = len(all_grid_speeds[0].content.split(","))
        len_hour_of_day = 24
        avg_speed_arr = []

        for it in range(length_of_grid):
            avg_speed_arr.append([])
            for t_of_day in range(len_hour_of_day):
                avg_speed_arr[it].append([])
        speed_arr = []

        for idx, grid_speed in enumerate(all_grid_speeds):
            speeds_now = [float(item) for item in grid_speed.content.split(",")]
            speed_arr.append(speeds_now)
            hour_now = int(grid_speed.create_time.hour)
            hour_of_data[idx] = hour_now
            #不在日期范围内并且数不是0的可以加入到平均速度数组中
            in_dt_range = False
            for it in range(len_of_dt):
                if dt_starts[it] <= grid_speed.create_time <= dt_ends[it]:
                    in_dt_range = True
                    label_dt_arr[idx] = 1
                    break

            if not in_dt_range:
                for it in range(length_of_grid):
                    data_of_speed = speeds_now[it]
                    if math.fabs(data_of_speed) > 1e-6:
                        avg_speed_arr[it][hour_now].append(data_of_speed)
            print "finish 1st step of %s" % grid_speed.create_time.strftime(second_format)


        mean_speed_arr = []

        for it in range(length_of_grid):
            mean_speed_arr.append([])
            for time_of_day in range(len_hour_of_day):
                if len(avg_speed_arr[it][time_of_day]):
                    mean_speed = np.array(avg_speed_arr[it][time_of_day]).mean()
                else:
                    mean_speed = 0.0
                mean_speed_arr[it].append(mean_speed)

        for idx in range(length_of_data):
            for it in range(length_of_grid):
                data_tmp = speed_arr[idx][it]
                if label_dt_arr[idx] or math.fabs(data_tmp) < 1e-6:
                    speed_arr[idx][it] = mean_speed_arr[it][hour_of_data[idx]]
                    visited_arr[idx] = 1

            if visited_arr[idx]:
                grid_speed = all_grid_speeds[idx]
                grid_speed.content = ",".join([str(round(item,3)) for item in speed_arr[idx]])
                print "%s has been edited" % grid_speed.create_time.strftime(second_format)
                grid_speed.save()
