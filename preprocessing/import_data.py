# -*- coding: utf-8 -*-
import xlrd,sys,os,datetime,math,csv,urllib2,json
from models import *
from preprocessing.baidumap import BaiduMap
reload(sys)
sys.setdefaultencoding('utf8')

weather_severity = {u"晴":0, u"浮尘":0, u"阴":0, u"多云":0, u"霾":1, u"雾":1, u"小雨":2, u"阵雨":2, u"雷阵雨":2, u"中雨":3, u"小雪":3, u"大雨":4, u"雨夹雪":4, u"暴雨":4, u"中雪":5, u"大雪":5}

def unicode_csv_reader(gbk_data, dialect=csv.excel, **kwargs):
    csv_reader = csv.reader(gbk_data, dialect=dialect, **kwargs)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]
# 获取excel的列对应的数字编号
def get_excel_index():
    excel_index_list = [chr(i) for i in range(65,91)]
    global excel_dict
    excel_dict = {ch:i for i,ch in enumerate(excel_index_list)}
    #print(excel_dict)

def query_baidu_address(lng,lat):

    query_url = 'http://api.map.baidu.com/geocoder/v2/?location='+str(lat) + ',' + str(lng)+'&output=json&pois=0&ak=rBHgzWXGwp7M0w0E8MSUUzrr'
    req = urllib2.Request(query_url)
    res = urllib2.urlopen(req).read()
    result = json.loads(res.decode("utf-8"))
    if result["status"] == 0:
        address = result["result"]["sematic_description"]
    else:
        address = u"未知"
    print address
    return address
def clear_122_lng_lat_invalid_data():
    call_incidences = Call_Incidence.objects.filter(id__range=[975861, 1019380])#Call_Incidence.objects.all()
    print "got all call_incidences"
    cnt = 0
    for call_incidence in call_incidences:
        if math.fabs(call_incidence.latitude) < 1e-6 or math.fabs(call_incidence.longitude) < 1e-6:
            cnt += 1
            call_incidence.delete()
            call_incidence.save()
    print "deleted %d call_incidences" % cnt
def import_app_incidences_from_json(input_file_path):
    input_file = open(input_file_path,"r")
    input_str = input_file.read().decode("utf-8")
    json_obj = json.loads(input_str)
    for item in json_obj:
        lng = item[0]
        lat = item[1]
        create_time = item[3]
        try:
            if not isinstance(lng, unicode) or not isinstance(lat,unicode) \
                    or not isinstance(create_time,unicode):
                continue
            lng = float(lng)
            lat = float(lat)
            create_time = datetime.datetime.strptime(create_time,"%Y-%m-%d %H:%M:%S")
            address = query_baidu_address(lng,lat)
            call_incidences = Call_Incidence(create_time=create_time, latitude=lat, longitude=lng, place=address)
            call_incidences.save()
        except Exception,e:
            print "decode failed"
            continue

    print len(json_obj)
    input_file.close()
#导入122电话事故举报数据
def import_call_incidence_data_of_2016(input_call_incidence_file):
    get_excel_index()
    TIME_INDEX = 'C'
    LON_INDEX = 'E'  #数据中存在经纬度都是0的情况，表示经纬度不存在
    LAT_INDEX = 'F'
    PLACE_INDEX = 'D'
    file = xlrd.open_workbook(input_call_incidence_file)
    file_sheets = file.sheets()
    num = 0
    bdmap = BaiduMap("北京市")
    for idx,table in enumerate(file_sheets):
        if True:#idx==1:
            nrows = table.nrows
            print '%s nrows: %d' % (input_call_incidence_file,nrows)
            num += 1
            if num % 10000 ==0:
                print "processed %d 122 data" % num

            places_time = []

            for i in range(1, nrows - 1):
                create_time = xlrd.xldate.xldate_as_datetime(table.cell(i, excel_dict[TIME_INDEX]).value, 0)  ##将float类型的时间转换成datetime类型
                create_time_str = create_time.strftime("%Y-%m-%d %H:%M:%S")
                longitude = table.cell(i, excel_dict[LON_INDEX]).value  ##float类型
                latitude = table.cell(i, excel_dict[LAT_INDEX]).value
                place = table.cell(i, excel_dict[PLACE_INDEX]).value
                # print "%f, %f,%s" %(longitude,latitude,place)
                if(longitude == 0 or latitude == 0):
                    places_time.append([create_time,place])
                    continue
                num += 1
                # output_str = str(region) + "," + create_time_str + "," + str(longitude) + "," + str(latitude) + "," + event_content + "," + place
                # call_incidence = Call_Incidence(create_time=create_time,longitude=longitude,latitude=latitude,place=place)
                # call_incidence.save()
            print "finish 1st stage"
            len_placetime=len(places_time)
            print
            cnt = 0
            dt_str = '2017-02-27 19:17:31'
            dt_tmp = datetime.datetime.strptime(dt_str,"%Y-%m-%d %H:%M:%S")
            for create_time,place in places_time:
                cnt+=1
                if True:#cnt>4060 and create_time > dt_tmp:
                    # print "cnt: %d, ct:%s\t %s" %(cnt,str(create_time),place)
                    if cnt % 20==0:
                        print "%d of %d: %f" %(cnt,len_placetime,cnt/float(len_placetime))
                    try:
                        bd_point = bdmap.getLocation(place)
                        if bd_point is None:
                            continue
                        # 地点名称,经度,纬度,可信度
                        # rtd_gps_info = (place, bd_point[0], bd_point[1], bd_point[2])
                        # print "%s\t%s\t%s\t%s" % (place, bd_point[0], bd_point[1], bd_point[2])
                        #可信度小于50%丢弃
                        if bd_point[2] <= 50:
                            continue
                        longitude = bd_point[0]
                        latitude = bd_point[1]
                        # final_preserve.append([create_time,place,longitude,latitude])
                        call_incidence = Call_Incidence(create_time=create_time,longitude=longitude,latitude=latitude,place=place)
                        call_incidence.save()
                    except Exception as e:
                        # longitude, latitude, region = -1, -1, -1
                        print("---bdmap error!! info:" + str(e))
                        continue

            print "finish 2nd stage"
    print("import call 122 finished!")

#导入App事故数据
def import_app_incidences_data_from_json(input_json_filepath):
    input_file = open(input_json_filepath,"r")
    input_str = input_file.read().decode("utf-8")
    json_obj = json.loads(input_str)
    for item in json_obj:
        longitude, latitude, latlng_address, create_time = item[0], item[1], item[2], item[3]
        try:
            if not isinstance(longitude, unicode) or not isinstance(latitude,unicode) or not isinstance(latlng_address,unicode) or not isinstance(create_time,unicode):
                continue
            create_time = datetime.datetime.strptime(create_time,"%Y-%m-%d %H:%M:%S")
            app_incidence = App_Incidence(longitude=longitude, latitude=latitude, place=latlng_address,
                                      create_time=create_time)
            app_incidence.save()
        except Exception,e:
            print "decode failed: %s" % str(e)
            continue
def import_violation_data(input_file_path):
    csvfile = open(input_file_path,"rb")
    reader = csv.reader(csvfile)
    for row in reader:
        lng = float(row[1]) if row[1]!='' else 0.0
        lat = float(row[2]) if row[2]!='' else 0.0
        if lng < 120 and lng > 100 and lat > 30 and lat < 45:
            dt = datetime.datetime.strptime(row[0], "%Y/%m/%d %H:%M")
            #print "lng: %f\tlat: %f\n" % (lng, lat)
            violation = Violation(create_time = dt,latitude = lat, longitude = lng)
            violation.save()
    print "import violation data sucess!"
def handle_wind(wind_desc):
    wind_desc = wind_desc.replace(u"级", u"").replace(u"小于", u"")
    wind_arr = wind_desc.split("-")
    if len(wind_arr) == 2:
        wind1 = int(wind_arr[0])
        wind2 = int(wind_arr[1])
        wind = (wind1 + wind2) / 2.0
    else:
        wind = float(wind_arr[0]) if wind_arr[0]!=u"微风" else 1.0
    return wind
def convert_to_weather_severity(weather_desc):
    weather_desc = weather_desc.replace(u"转", u"~")
    weather_arr = weather_desc.split(u"到")
    if len(weather_arr) == 1:
        weather_arr = weather_arr[0].split("~")
        if len(weather_arr) == 1:
            ws = float(weather_severity[weather_arr[0]])
        else:
            ws = 0.5 * (weather_severity[weather_arr[0]]+ weather_severity[weather_arr[1]])
    else:
        weather_0 = weather_arr[0] + weather_arr[1][-1]
        weather_1 = weather_arr[1]
        ws = 0.5 * (weather_severity[weather_0]+ weather_severity[weather_1])
    return ws
def import_weather_to_db(input_file_path):
    reader = unicode_csv_reader(open(input_file_path,"rb"))
    for row in reader:
        dt = datetime.datetime.strptime(row[0], "%Y-%m-%d")
        dt2 = dt.date()
        ht = int(row[1])
        lt = int(row[2])
        weather_desc = row[3]
        ws = convert_to_weather_severity(weather_desc)
        wind = handle_wind(row[4])
        #print "dt: %s, ht: %d, lt:%d, wd: %s, ws: %.2f, wind:%.2f\n" %(row[0],ht,lt,weather_desc,ws,wind)
        weather = Weather(date_w = dt2, highest_temperature= ht, lowest_temperature= lt, wind= wind, weather_desc=weather_desc, weather_severity = ws)
        weather.save()
    print "import weather success!"
def import_air_quality_to_db(input_file_path):
    csvfile = open(input_file_path,"rb")
    reader = csv.reader(csvfile)
    for row in reader:
        dt = datetime.datetime.strptime(row[0], "%Y-%m-%d")
        dt2 = dt.date()
        aqi = int(row[1])
        pm25 = int(row[2])
        air_quality = Air_Quality(date_a= dt2, aqi= aqi, pm25= pm25)
        air_quality.save()
    print "import air quality success!"
if __name__ == "__main__":
    for i in range(10,11):
        input_call_incidence_file = "/Users/Ren/PycharmProjects/PoliceIndex/beijing_data/2016_accidents/"+str(i)+".xls"
        import_call_incidence_data_of_2016(input_call_incidence_file=input_call_incidence_file)