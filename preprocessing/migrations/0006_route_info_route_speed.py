# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('preprocessing', '0005_accidents_array_region_function'),
    ]

    operations = [
        migrations.CreateModel(
            name='Route_Info',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('route_id', models.IntegerField(verbose_name=b'\xe9\x81\x93\xe8\xb7\xafID')),
                ('route_name', models.TextField(verbose_name=b'\xe9\x81\x93\xe8\xb7\xaf\xe5\x90\x8d\xe7\xa7\xb0')),
                ('route_direction', models.TextField(verbose_name=b'\xe9\x81\x93\xe8\xb7\xaf\xe6\x96\xb9\xe5\x90\x91')),
                ('start_name', models.TextField(verbose_name=b'\xe8\xb5\xb7\xe7\x82\xb9\xe8\xb7\xaf\xe5\x90\x8d\xe7\xa7\xb0')),
                ('end_name', models.TextField(verbose_name=b'\xe7\xbb\x88\xe7\x82\xb9\xe8\xb7\xaf\xe5\x90\x8d\xe7\xa7\xb0')),
                ('route_length', models.IntegerField(verbose_name=b'\xe9\x81\x93\xe8\xb7\xaf\xe9\x95\xbf\xe5\xba\xa6')),
                ('start_lon', models.DecimalField(verbose_name=b'\xe8\xb5\xb7\xe7\x82\xb9\xe7\xbb\x8f\xe5\xba\xa6', max_digits=10, decimal_places=7)),
                ('start_lat', models.DecimalField(verbose_name=b'\xe8\xb5\xb7\xe7\x82\xb9\xe7\xba\xac\xe5\xba\xa6', max_digits=10, decimal_places=7)),
                ('end_lon', models.DecimalField(verbose_name=b'\xe7\xbb\x88\xe7\x82\xb9\xe7\xbb\x8f\xe5\xba\xa6', max_digits=10, decimal_places=7)),
                ('end_lat', models.DecimalField(verbose_name=b'\xe7\xbb\x88\xe7\x82\xb9\xe7\xba\xac\xe5\xba\xa6', max_digits=10, decimal_places=7)),
                ('route_subid', models.IntegerField(default=-1, verbose_name=b'\xe5\x90\x8e\xe7\xbb\xa7routeid')),
                ('route_preid', models.IntegerField(default=-1, verbose_name=b'\xe5\x89\x8d\xe9\xa9\xb1routeid')),
            ],
        ),
        migrations.CreateModel(
            name='Route_Speed',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('route_id', models.IntegerField(verbose_name=b'\xe9\x81\x93\xe8\xb7\xafID')),
                ('create_time', models.DateTimeField(verbose_name=b'\xe6\x97\xb6\xe9\x97\xb4')),
                ('avg_speed', models.DecimalField(verbose_name=b'\xe5\xb9\xb3\xe5\x9d\x87\xe9\x80\x9f\xe5\xba\xa6', max_digits=5, decimal_places=2)),
            ],
        ),
    ]
