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
