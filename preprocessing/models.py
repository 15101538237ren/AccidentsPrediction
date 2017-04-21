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
    longitude = models.DecimalField('经度',max_digits=10,decimal_places=7)
    latitude = models.DecimalField('纬度',max_digits=10,decimal_places=7)
    create_time = models.DateTimeField('举报时间')
    def __unicode__(self):
        return u'经度:'+str(self.longitude) + u', 纬度:' + str(self.latitude) + u', 举报时间:' + str(self.create_time) +u'\n'
