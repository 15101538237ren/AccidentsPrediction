# -*- coding: utf-8 -*-

from django.db import models
from django_permanent.models import PermanentModel
# Create your models here.


#122电话事故举报对应的数据库表
class Call_Incidence(PermanentModel):
    create_time = models.DateTimeField('122报警时间')
    longitude = models.DecimalField('经度',max_digits=10,decimal_places=7)
    latitude = models.DecimalField('纬度',max_digits=10,decimal_places=7)
    place = models.TextField('地点')
    def __unicode__(self):
        return  u'地点:' + str(self.place) +u', 举报时间:' + str(self.create_time) +u', 经度:'+ str(self.longitude) +u', 纬度:'+ str(self.latitude) +u'\n'

#App事故举报对应的数据库表
class App_Incidence(PermanentModel):
    longitude = models.DecimalField('经度',max_digits=10,decimal_places=7)
    latitude = models.DecimalField('纬度',max_digits=10,decimal_places=7)
    place = models.TextField('地点')
    create_time = models.DateTimeField('举报时间')
    def __unicode__(self):
        return u'经度:'+str(self.longitude) + u', 纬度:' + str(self.latitude) + u', 举报时间:' + str(self.create_time) + u', 地点:' + str(self.place) +u'\n'

#App违法举报对应的数据库表
class Violation(PermanentModel):
    longitude = models.DecimalField('经度',max_digits=10, decimal_places=7)
    latitude = models.DecimalField('纬度',max_digits=10, decimal_places=7)
    create_time = models.DateTimeField('举报时间')
    def __unicode__(self):
        return u'经度:'+str(self.longitude) + u', 纬度:' + str(self.latitude) + u', 举报时间:' + str(self.create_time) +u'\n'
class Weather(models.Model):
    date_w = models.DateField('日期')
    highest_temperature = models.IntegerField('最高气温')
    lowest_temperature = models.IntegerField('最低气温')
    wind = models.DecimalField('风力',max_digits=5, decimal_places=2)
    weather_severity = models.DecimalField('天气严重性', max_digits = 5, decimal_places = 3)
    weather_desc = models.TextField('天气描述')
    def __unicode__(self):
        return u'日期:'+str(self.date_w) + u',高温:'+str(self.highest_temperature) + u', 低温:' + str(self.lowest_temperature) + u', 天气严重性:' + str(self.weather_severity) + u', 天气:' + str(self.weather_desc)  + u', 风力:' + str(self.wind) +u'\n'

class Air_Quality(models.Model):
    date_a = models.DateField('日期')
    aqi = models.IntegerField('空气质量指数AQI')
    pm25 = models.IntegerField('PM2.5')
    def __unicode__(self):
        return u'日期:'+str(self.date_a) + u',AQI:'+str(self.aqi) + u', PM2.5:' + str(self.pm25) +u'\n'
class Accidents_Array(models.Model):
    DAWN = 0
    MORNING_RUSH = 1
    MORNING_WORKING = 2
    NOON = 3
    AFTERNOON_WORK = 4
    AFTERNOON_RUSH = 5
    NIGHT = 6

    TIME_SEG_TYPE = (
        (DAWN, "午夜"),
        (MORNING_RUSH,"早高峰"),
        (MORNING_WORKING,"早工作"),
        (NOON,"中午休息"),
        (AFTERNOON_WORK,"下午工作"),
        (AFTERNOON_RUSH,"下午高峰"),
        (NIGHT,"晚间"),

    )

    time_interval = models.IntegerField('时间间隔/min')
    spatial_interval = models.IntegerField('空间间隔/m')
    create_time = models.DateTimeField('时间区间')
    content = models.TextField('事故内容')

    highest_temperature = models.IntegerField('最高气温')
    lowest_temperature = models.IntegerField('最低气温')
    wind = models.DecimalField('风力',max_digits=5, decimal_places=2)
    weather_severity = models.DecimalField('天气严重性', max_digits = 5, decimal_places = 3)

    aqi = models.IntegerField('空气质量指数AQI')
    pm25 = models.IntegerField('PM2.5')

    is_holiday = models.BooleanField('是否节假日')
    is_weekend = models.BooleanField('是否周末')
    time_segment = models.SmallIntegerField('时间段', choices= TIME_SEG_TYPE)

class Region_Function(models.Model):
    REGION_TYPE = (
        (1,"交通设施"),
        (2,"住宿"),
        (3,"医院"),
        (4,"商务住宅_公司"),
        (5,"商场超市"),
        (6,"娱乐场所"),
        (7,"学校"),
        (8,"旅游景点"),
        (9,"生活服务"),
        (10,"自住住宅"),
        (11,"银行金融"),
        (12,"餐饮"),

    )
    spatial_interval = models.IntegerField('空间间隔/m')
    region_type = models.SmallIntegerField('区域类型',choices= REGION_TYPE)
    region_cnt_matrix = models.TextField('区域内容')
class Route_Info(models.Model):
    route_id = models.IntegerField('道路ID')
    route_name = models.TextField('道路名称')
    route_direction = models.TextField('道路方向')
    start_name = models.TextField('起点路名称')
    end_name = models.TextField('终点路名称')
    route_length = models.IntegerField('道路长度')
    start_lon = models.DecimalField('起点经度',max_digits=10,decimal_places=7)
    start_lat = models.DecimalField('起点纬度',max_digits=10,decimal_places=7)

    end_lon = models.DecimalField('终点经度',max_digits=10,decimal_places=7)
    end_lat = models.DecimalField('终点纬度',max_digits=10,decimal_places=7)

    route_subid = models.TextField('后继routeid')
    route_preid = models.TextField('前驱routeid')

class Route_Speed(models.Model):
    route_id = models.IntegerField('道路ID')
    create_time = models.DateTimeField('时间')
    avg_speed = models.DecimalField('平均速度',max_digits=5, decimal_places=2)
    valid = models.SmallIntegerField('零点是否有数据',default=1)
    relative_speed = models.DecimalField('相对零点的速度',max_digits=5, decimal_places=2,default=-1.0)
class Route_Related_Grid(models.Model):
    route_id = models.IntegerField('道路ID')
    grid_ids = models.TextField('道路经过的网格ID列表')

