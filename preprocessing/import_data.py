# -*- coding: utf-8 -*-
import xlrd,sys,os,datetime,pytz,pickle,csv,calendar
from models import *
from preprocessing.baidumap import BaiduMap
reload(sys)
sys.setdefaultencoding('utf8')
# 获取excel的列对应的数字编号
def get_excel_index():
    excel_index_list = [chr(i) for i in range(65,91)]
    global excel_dict
    excel_dict = {ch:i for i,ch in enumerate(excel_index_list)}
    #print(excel_dict)

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
def import_weather_to_db(input_file_path):
    pass
if __name__ == "__main__":
    for i in range(10,11):
        input_call_incidence_file = "/Users/Ren/PycharmProjects/PoliceIndex/beijing_data/2016_accidents/"+str(i)+".xls"
        import_call_incidence_data_of_2016(input_call_incidence_file=input_call_incidence_file)