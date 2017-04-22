# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('preprocessing', '0003_auto_20170422_0906'),
    ]

    operations = [
        migrations.AlterField(
            model_name='weather',
            name='wind',
            field=models.DecimalField(verbose_name=b'\xe9\xa3\x8e\xe5\x8a\x9b', max_digits=5, decimal_places=2),
        ),
    ]
