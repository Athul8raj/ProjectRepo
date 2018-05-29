from django.urls import path
from django.contrib.auth.views import login, logout
from . import views

urlpatterns = [
    path('', views.tasks, name='tasks'),
    path('todo/', views.tasks, name='tasks'),
    path('add/', views.add, name='add'),
    path('edit/<str:username>', views.edit, name='edit'),
    path('delete/<str:username>', views.delete, name='delete'),
    path('search/<str:status>', views.search, name='search'),
    path('details/<id>', views.details, name='details'),
    path('login/', login, {'template_name': 'login.html'}),
    path('logout/', logout, {'template_name': 'logout.html'}),
    path('register/', views.register, name='register')
]
