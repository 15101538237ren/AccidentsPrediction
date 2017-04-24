# -*- coding: utf-8 -*-
import sys,os,requests,urllib,json,math,pickle,datetime
from  models import Call_Incidence
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
    start_dt = datetime.datetime.strptime("2016-01-01 00:00:00","%Y-%m-%d %H:%M:%S")
    end_dt = datetime.datetime.strptime("2016-01-03 00:00:00","%Y-%m-%d %H:%M:%S")
    time_delta = datetime.timedelta(minutes= interval)
    temp_time = start_dt
    temp_time_str = temp_time.strftime("%Y-%m-%d %H:%M")
    accidents_of_all = {}
    time_list = []
    while temp_time < end_dt:
        accidents_of_all[temp_time_str] = []
        time_list.append(temp_time_str)
        temp_time = temp_time + time_delta
        temp_time_str = temp_time.strftime("%Y-%m-%d %H:%M")
    now_time = start_dt
    now_time_str = now_time.strftime("%Y-%m-%d %H:%M")

    for i in range(len(accidents)):
        accidents_time = accidents[i].create_time
        if accidents_time > now_time and accidents_time <= (now_time + time_delta):
            accidents_of_all[now_time_str].append([accidents[i].longitude, accidents[i].latitude])
        elif accidents_time > (now_time + time_delta) :
            now_time = now_time + time_delta
            now_time_str = now_time.strftime("%Y-%m-%d %H:%M")
            accidents_of_all[now_time_str].append([accidents[i].longitude, accidents[i].latitude])
    print "finish partition points by time!"
    return time_list,accidents_of_all

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
def label_all_accidents(input_pickle_file, d_lat, d_lng, n_lng, n_lat, interval = 60):
    time_list, accidents_of_all = partition_geopoints_by_time(input_pickle_file,interval = interval)
    accidents_arr = {}
    print "start labeling"
    for time_now in time_list:
        geo_points = accidents_of_all[time_now]
        geo_cnts = label_geo_points(geo_points, d_lat, d_lng, n_lng, n_lat)
        accidents_arr[time_now] = geo_cnts
    print "finish labeling"
    print len(accidents_arr["2016-02-28 08:00"])
def get_all_accidents_from_db(output_pickle):
    outfile = open(output_pickle, 'wb')
    print "query start!"
    start_dt = datetime.datetime.strptime("2016-01-01 00:00:00","%Y-%m-%d %H:%M:%S")
    end_dt = datetime.datetime.strptime("2016-01-03 00:00:00","%Y-%m-%d %H:%M:%S")
    accidents = Call_Incidence.objects.filter(create_time__range=[start_dt, end_dt]).order_by("create_time")
    print "query finish!"
    print "before: %d" % len(accidents)
    accidents_filtered = []
    accidents_set = set('temp')
    for accident in accidents:
        out_str = "%s,%s,%s" %(accident.create_time.strftime("%Y-%m-%d %H:%M:%S"), accident.latitude, accident.longitude)
        if out_str not in accidents_set:
            accidents_set.add(out_str)
            accidents_filtered.append(accident)
    #accidents_set.remove('temp')
    print "after set filtered len: %d\n start dump!" % len(accidents_filtered)

    pickle.dump(accidents_filtered,outfile,-1)
    outfile.close()

    print "dump success!"
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
