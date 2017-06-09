# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('preprocessing', '0010_auto_20170604_1433'),
    ]

    operations = [
        migrations.CreateModel(
            name='Grid_Speed',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('time_interval', models.IntegerField(verbose_name=b'\xe6\x97\xb6\xe9\x97\xb4\xe9\x97\xb4\xe9\x9a\x94/min')),
                ('spatial_interval', models.IntegerField(verbose_name=b'\xe7\xa9\xba\xe9\x97\xb4\xe9\x97\xb4\xe9\x9a\x94/m')),
                ('create_time', models.DateTimeField(verbose_name=b'\xe6\x97\xb6\xe9\x97\xb4\xe5\x8c\xba\xe9\x97\xb4')),
                ('content', models.TextField(verbose_name=b'\xe9\x80\x9f\xe5\xba\xa6\xe5\x86\x85\xe5\xae\xb9')),
            ],
        ),
    ]
