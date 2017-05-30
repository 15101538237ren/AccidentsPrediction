# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('preprocessing', '0006_route_info_route_speed'),
    ]

    operations = [
        migrations.AlterField(
            model_name='route_info',
            name='route_preid',
            field=models.TextField(verbose_name=b'\xe5\x89\x8d\xe9\xa9\xb1routeid'),
        ),
        migrations.AlterField(
            model_name='route_info',
            name='route_subid',
            field=models.TextField(verbose_name=b'\xe5\x90\x8e\xe7\xbb\xa7routeid'),
        ),
    ]
