# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('preprocessing', '0009_auto_20170604_0818'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='route_speed',
            name='valid',
        ),
        migrations.AddField(
            model_name='route_info',
            name='valid',
            field=models.SmallIntegerField(default=1, verbose_name=b'\xe9\x9b\xb6\xe7\x82\xb9\xe6\x98\xaf\xe5\x90\xa6\xe6\x9c\x89\xe6\x95\xb0\xe6\x8d\xae'),
        ),
    ]
