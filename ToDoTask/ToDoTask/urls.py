from django.contrib import admin
from django.urls import path, include
from django.contrib.auth.views import login, logout
from . import views

urlpatterns = [
    path('', include('Todos.urls')),
    path('register/', include('Todos.urls')),
    path('todo/', include('Todos.urls')),
    path('add/', include('Todos.urls')),
    path('edit/<str:username>', include('Todos.urls')),
    path('delete/<str:username>', include('Todos.urls')),
    path('search/<str:status>', include('Todos.urls')),
    path('details/<id>', include('Todos.urls')),
    path('', views.login_redirect, name='login_redirect'),
    path('login/', login, {'template_name': 'login.html'}),
    path('logout/', logout, {'template_name': 'logout.html'}),
    path('admin/', admin.site.urls)
]
