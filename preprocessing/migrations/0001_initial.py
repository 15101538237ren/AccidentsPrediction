# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='App_Incidence',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('removed', models.DateTimeField(default=None, null=True, editable=False, blank=True)),
                ('longitude', models.DecimalField(verbose_name=b'\xe7\xbb\x8f\xe5\xba\xa6', max_digits=10, decimal_places=7)),
                ('latitude', models.DecimalField(verbose_name=b'\xe7\xba\xac\xe5\xba\xa6', max_digits=10, decimal_places=7)),
                ('place', models.TextField(verbose_name=b'\xe5\x9c\xb0\xe7\x82\xb9')),
                ('create_time', models.DateTimeField(verbose_name=b'\xe4\xb8\xbe\xe6\x8a\xa5\xe6\x97\xb6\xe9\x97\xb4')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Call_Incidence',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('removed', models.DateTimeField(default=None, null=True, editable=False, blank=True)),
                ('create_time', models.DateTimeField(verbose_name=b'122\xe6\x8a\xa5\xe8\xad\xa6\xe6\x97\xb6\xe9\x97\xb4')),
                ('longitude', models.DecimalField(verbose_name=b'\xe7\xbb\x8f\xe5\xba\xa6', max_digits=10, decimal_places=7)),
                ('latitude', models.DecimalField(verbose_name=b'\xe7\xba\xac\xe5\xba\xa6', max_digits=10, decimal_places=7)),
                ('place', models.TextField(verbose_name=b'\xe5\x9c\xb0\xe7\x82\xb9')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Violation',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('removed', models.DateTimeField(default=None, null=True, editable=False, blank=True)),
                ('longitude', models.DecimalField(verbose_name=b'\xe7\xbb\x8f\xe5\xba\xa6', max_digits=10, decimal_places=7)),
                ('latitude', models.DecimalField(verbose_name=b'\xe7\xba\xac\xe5\xba\xa6', max_digits=10, decimal_places=7)),
                ('create_time', models.DateTimeField(verbose_name=b'\xe4\xb8\xbe\xe6\x8a\xa5\xe6\x97\xb6\xe9\x97\xb4')),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
