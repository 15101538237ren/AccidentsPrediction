from django.conf.urls import patterns, include, url
from django.contrib.auth import views as auth_views

urlpatterns = patterns('preprocessing.views',
 url(r'^index$','index',name='index'),
 url(r'^grid','grid',name='grid'),
 url(r'^timeline','timeline',name='timeline'),
 url(r'^gtl','grid_timeline',name='grid_timeline'),
 url(r'^roads','visualize_roads',name='visualize_roads'),
 url(r'^query_status','query_status',name='query_status'),
)