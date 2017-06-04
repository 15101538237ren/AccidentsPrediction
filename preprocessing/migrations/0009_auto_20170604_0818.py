# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('preprocessing', '0008_route_related_grid'),
    ]

    operations = [
        migrations.AddField(
            model_name='route_speed',
            name='relative_speed',
            field=models.DecimalField(default=-1.0, verbose_name=b'\xe7\x9b\xb8\xe5\xaf\xb9\xe9\x9b\xb6\xe7\x82\xb9\xe7\x9a\x84\xe9\x80\x9f\xe5\xba\xa6', max_digits=5, decimal_places=2),
        ),
        migrations.AddField(
            model_name='route_speed',
            name='valid',
            field=models.SmallIntegerField(default=1, verbose_name=b'\xe9\x9b\xb6\xe7\x82\xb9\xe6\x98\xaf\xe5\x90\xa6\xe6\x9c\x89\xe6\x95\xb0\xe6\x8d\xae'),
        ),
    ]
