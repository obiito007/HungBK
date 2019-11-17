from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
urlpatterns = [
    path('', views.index),
    path('kianh/',views.teach),
    path('tao/',views.index1),
    path('list/',views.list),
    #path('VCS/<int:id>/',views.Poss),
    path('up/',views.upload_file, name='upload_file'),
    path('bang/',views.bang),
    path('xacthuc/',views.xac),
    path('login/',auth_views.LoginView.as_view(template_name="VCS/login.html"), name="login"),
    path('logout/',auth_views.LogoutView.as_view(next_page='/'),name='logout'),
    path('register/',views.register,name='register'),
    path('danh/',views.danh),
    path('export/xls/$', views.export_users_xls, name='export_users_xls'),

]