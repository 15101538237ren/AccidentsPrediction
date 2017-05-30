# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('preprocessing', '0007_auto_20170530_1125'),
    ]

    operations = [
        migrations.CreateModel(
            name='Route_Related_Grid',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('route_id', models.IntegerField(verbose_name=b'\xe9\x81\x93\xe8\xb7\xafID')),
                ('grid_ids', models.TextField(verbose_name=b'\xe9\x81\x93\xe8\xb7\xaf\xe7\xbb\x8f\xe8\xbf\x87\xe7\x9a\x84\xe7\xbd\x91\xe6\xa0\xbcID\xe5\x88\x97\xe8\xa1\xa8')),
            ],
        ),
    ]
