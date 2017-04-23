from django.conf.urls import patterns, include, url
from django.contrib.auth import views as auth_views
from .views import *

urlpatterns = patterns('preprocessing.views',
 url(r'^index$','index',name='index'),
 url(r'^grid','grid',name='grid'),
)