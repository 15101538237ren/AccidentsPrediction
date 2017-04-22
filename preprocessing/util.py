# -*- coding: utf-8 -*-
import sys,os,requests,urllib,json
reload(sys)
sys.setdefaultencoding('utf8')

def get_liuhuan_poi():
    keyword = '六环'
    url = 'http://api.map.baidu.com/place/v2/search?'
    headers = {}
    tot_pages = 0
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
    if ret['status'] == 0:
        total_cnt = ret['total']
        tot_pages = total_cnt / 20
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
                print res_t
                results.append(res_t)
        print 'total pages: %d finished' % tot_pages

    else:
        print unicode(ret['message'])
