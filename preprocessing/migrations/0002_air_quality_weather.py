# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('preprocessing', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Air_Quality',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('date_a', models.DateField(verbose_name=b'\xe6\x97\xa5\xe6\x9c\x9f')),
                ('aqi', models.IntegerField(max_length=6, verbose_name=b'\xe7\xa9\xba\xe6\xb0\x94\xe8\xb4\xa8\xe9\x87\x8f\xe6\x8c\x87\xe6\x95\xb0AQI')),
                ('pm25', models.IntegerField(max_length=6, verbose_name=b'PM2.5')),
            ],
        ),
        migrations.CreateModel(
            name='Weather',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('date_w', models.DateField(verbose_name=b'\xe6\x97\xa5\xe6\x9c\x9f')),
                ('highest_temperature', models.IntegerField(max_length=3, verbose_name=b'\xe6\x9c\x80\xe9\xab\x98\xe6\xb0\x94\xe6\xb8\xa9')),
                ('lowest_temperature', models.IntegerField(max_length=3, verbose_name=b'\xe6\x9c\x80\xe4\xbd\x8e\xe6\xb0\x94\xe6\xb8\xa9')),
                ('wind', models.IntegerField(max_length=2, verbose_name=b'\xe9\xa3\x8e\xe5\x8a\x9b')),
                ('weather_severity', models.DecimalField(verbose_name=b'\xe5\xa4\xa9\xe6\xb0\x94\xe4\xb8\xa5\xe9\x87\x8d\xe6\x80\xa7', max_digits=5, decimal_places=3)),
                ('weather_desc', models.TextField(verbose_name=b'\xe5\xa4\xa9\xe6\xb0\x94\xe6\x8f\x8f\xe8\xbf\xb0')),
            ],
        ),
    ]
