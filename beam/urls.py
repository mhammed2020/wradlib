from django.urls import path
from . import views

urlpatterns = [
    # path('', views.my_view_helper,name ="helper-path"),
    path('', views.my_view,name ="my_view-path"),
    path('all-senarios/', views.list_location,name ="all-senarios"),
    #  path('simulation/', views.home,name ="my_view-path"),

]
