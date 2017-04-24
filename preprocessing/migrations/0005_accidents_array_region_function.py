# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('preprocessing', '0004_auto_20170422_1045'),
    ]

    operations = [
        migrations.CreateModel(
            name='Accidents_Array',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('time_interval', models.IntegerField(verbose_name=b'\xe6\x97\xb6\xe9\x97\xb4\xe9\x97\xb4\xe9\x9a\x94/min')),
                ('spatial_interval', models.IntegerField(verbose_name=b'\xe7\xa9\xba\xe9\x97\xb4\xe9\x97\xb4\xe9\x9a\x94/m')),
                ('create_time', models.DateTimeField(verbose_name=b'\xe6\x97\xb6\xe9\x97\xb4\xe5\x8c\xba\xe9\x97\xb4')),
                ('content', models.TextField(verbose_name=b'\xe4\xba\x8b\xe6\x95\x85\xe5\x86\x85\xe5\xae\xb9')),
                ('highest_temperature', models.IntegerField(verbose_name=b'\xe6\x9c\x80\xe9\xab\x98\xe6\xb0\x94\xe6\xb8\xa9')),
                ('lowest_temperature', models.IntegerField(verbose_name=b'\xe6\x9c\x80\xe4\xbd\x8e\xe6\xb0\x94\xe6\xb8\xa9')),
                ('wind', models.DecimalField(verbose_name=b'\xe9\xa3\x8e\xe5\x8a\x9b', max_digits=5, decimal_places=2)),
                ('weather_severity', models.DecimalField(verbose_name=b'\xe5\xa4\xa9\xe6\xb0\x94\xe4\xb8\xa5\xe9\x87\x8d\xe6\x80\xa7', max_digits=5, decimal_places=3)),
                ('aqi', models.IntegerField(verbose_name=b'\xe7\xa9\xba\xe6\xb0\x94\xe8\xb4\xa8\xe9\x87\x8f\xe6\x8c\x87\xe6\x95\xb0AQI')),
                ('pm25', models.IntegerField(verbose_name=b'PM2.5')),
                ('is_holiday', models.BooleanField(verbose_name=b'\xe6\x98\xaf\xe5\x90\xa6\xe8\x8a\x82\xe5\x81\x87\xe6\x97\xa5')),
                ('is_weekend', models.BooleanField(verbose_name=b'\xe6\x98\xaf\xe5\x90\xa6\xe5\x91\xa8\xe6\x9c\xab')),
                ('time_segment', models.SmallIntegerField(verbose_name=b'\xe6\x97\xb6\xe9\x97\xb4\xe6\xae\xb5', choices=[(0, b'\xe5\x8d\x88\xe5\xa4\x9c'), (1, b'\xe6\x97\xa9\xe9\xab\x98\xe5\xb3\xb0'), (2, b'\xe6\x97\xa9\xe5\xb7\xa5\xe4\xbd\x9c'), (3, b'\xe4\xb8\xad\xe5\x8d\x88\xe4\xbc\x91\xe6\x81\xaf'), (4, b'\xe4\xb8\x8b\xe5\x8d\x88\xe5\xb7\xa5\xe4\xbd\x9c'), (5, b'\xe4\xb8\x8b\xe5\x8d\x88\xe9\xab\x98\xe5\xb3\xb0'), (6, b'\xe6\x99\x9a\xe9\x97\xb4')])),
            ],
        ),
        migrations.CreateModel(
            name='Region_Function',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('spatial_interval', models.IntegerField(verbose_name=b'\xe7\xa9\xba\xe9\x97\xb4\xe9\x97\xb4\xe9\x9a\x94/m')),
                ('region_type', models.SmallIntegerField(verbose_name=b'\xe5\x8c\xba\xe5\x9f\x9f\xe7\xb1\xbb\xe5\x9e\x8b', choices=[(1, b'\xe4\xba\xa4\xe9\x80\x9a\xe8\xae\xbe\xe6\x96\xbd'), (2, b'\xe4\xbd\x8f\xe5\xae\xbf'), (3, b'\xe5\x8c\xbb\xe9\x99\xa2'), (4, b'\xe5\x95\x86\xe5\x8a\xa1\xe4\xbd\x8f\xe5\xae\x85_\xe5\x85\xac\xe5\x8f\xb8'), (5, b'\xe5\x95\x86\xe5\x9c\xba\xe8\xb6\x85\xe5\xb8\x82'), (6, b'\xe5\xa8\xb1\xe4\xb9\x90\xe5\x9c\xba\xe6\x89\x80'), (7, b'\xe5\xad\xa6\xe6\xa0\xa1'), (8, b'\xe6\x97\x85\xe6\xb8\xb8\xe6\x99\xaf\xe7\x82\xb9'), (9, b'\xe7\x94\x9f\xe6\xb4\xbb\xe6\x9c\x8d\xe5\x8a\xa1'), (10, b'\xe8\x87\xaa\xe4\xbd\x8f\xe4\xbd\x8f\xe5\xae\x85'), (11, b'\xe9\x93\xb6\xe8\xa1\x8c\xe9\x87\x91\xe8\x9e\x8d'), (12, b'\xe9\xa4\x90\xe9\xa5\xae')])),
                ('region_cnt_matrix', models.TextField(verbose_name=b'\xe5\x8c\xba\xe5\x9f\x9f\xe5\x86\x85\xe5\xae\xb9')),
            ],
        ),
    ]