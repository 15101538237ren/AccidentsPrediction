# -*- coding: utf-8 -*-
import sys,os,requests,urllib,json,math
reload(sys)
sys.setdefaultencoding('utf8')
min_lat = 39.696203
max_lat = 40.181729
min_lng = 116.099649
max_lng = 116.718542
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
def generate_polylines_for_beijing(lng_coors, lat_coors,output_file_path):
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

def get_liuhuan_poi(output_file_path):
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
    # ret = requests.get(url,params = params,headers=headers).json()
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

    d_lat = 0.0042
    d_lng = 0.006

    n_lat_delta = int(math.ceil((max_lat - min_lat)/d_lat))
    n_lng_delta = int(math.ceil((max_lng - min_lng)/d_lng))

    lng_coors = [min_lng + i * d_lng for i in range(n_lng_delta)]
    lat_coors = [min_lat + i * d_lat for i in range(n_lat_delta)]
    #generate_grid_for_beijing(lng_coors, lat_coors,output_file_path)
    generate_polylines_for_beijing(lng_coors,lat_coors,output_file_path)
    print 'min_lat: %f, max_lat: %f, min_lng: %f , max_lng: %f\n' % (min_lat, max_lat, min_lng, max_lng)
    return min_lat,max_lat,min_lng,max_lng
