# -*- coding: utf-8 -*-
import sys,os,requests,urllib,json,math
reload(sys)
sys.setdefaultencoding('utf8')

def generate_grid_for_beijing(lng_coors, lat_coors):
    pass

def get_liuhuan_poi():
    keyword = '六环'
    url = 'http://api.map.baidu.com/place/v2/search?'
    headers = {}
    params = {
        'query':keyword,
        'page_num':0,
        'page_size':20,
        'output':'json',
        'scope':1,
        'region':'北京',
        'ak':'eM0GfCwd27kZRyM49ZOkvkOaidDXz6Wf'
    }
    ret = requests.get(url,params = params,headers=headers).json()
    results = []
    min_lat = 9999999
    max_lat = -9999999
    min_lng = 9999999
    max_lng = -9999999
    tot_pages = 5
    for i in range(tot_pages):
        params['page_num'] = i
        res = requests.get(url,params = params,headers=headers).json()
        res = res['results']
        for j in range(len(res)):
            res_t = {}
            res_t['lat'] = res[j]['location']['lat']
            res_t['lng'] = res[j]['location']['lng']
            res_t['name'] = res[j]['name']
            res_t['address'] = res[j]['address']
            if res_t['lat'] < min_lat:
                min_lat = res_t['lat']
            elif res_t['lat'] > max_lat:
                max_lat = res_t['lat']
            if res_t['lng'] < min_lng:
                min_lng = res_t['lng']
            elif res_t['lng'] > max_lng:
                max_lng = res_t['lng']
            print res_t
            results.append(res_t)
    print 'get poi finished'

    d_lat = 0.004
    d_lng = 0.005

    n_lat_delta = int(math.ceil((max_lat - min_lat)/d_lat))
    n_lng_delta = int(math.ceil((max_lng - min_lng)/d_lng))

    lng_coors = [min_lng + i * d_lng for i in range(n_lng_delta)]
    lat_coors = [min_lat + i * d_lat for i in range(n_lat_delta)]

    return min_lat,max_lat,min_lng,max_lng
