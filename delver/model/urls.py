from django.urls import path
from . import views
  
urlpatterns = [
    path('',views.search,name="search"),
    path('login/', views.login, name ='login'),
    path('author-page/', views.authorPage, name ='authorPage'),
]