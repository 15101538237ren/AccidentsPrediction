# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('preprocessing', '0002_air_quality_weather'),
    ]

    operations = [
        migrations.AlterField(
            model_name='air_quality',
            name='aqi',
            field=models.IntegerField(verbose_name=b'\xe7\xa9\xba\xe6\xb0\x94\xe8\xb4\xa8\xe9\x87\x8f\xe6\x8c\x87\xe6\x95\xb0AQI'),
        ),
        migrations.AlterField(
            model_name='air_quality',
            name='pm25',
            field=models.IntegerField(verbose_name=b'PM2.5'),
        ),
        migrations.AlterField(
            model_name='weather',
            name='highest_temperature',
            field=models.IntegerField(verbose_name=b'\xe6\x9c\x80\xe9\xab\x98\xe6\xb0\x94\xe6\xb8\xa9'),
        ),
        migrations.AlterField(
            model_name='weather',
            name='lowest_temperature',
            field=models.IntegerField(verbose_name=b'\xe6\x9c\x80\xe4\xbd\x8e\xe6\xb0\x94\xe6\xb8\xa9'),
        ),
        migrations.AlterField(
            model_name='weather',
            name='wind',
            field=models.IntegerField(verbose_name=b'\xe9\xa3\x8e\xe5\x8a\x9b'),
        ),
    ]
