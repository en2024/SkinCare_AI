from django.urls import path
from . import views

urlpatterns = [
    path('',          views.home,           name='home'),
    path('analyzer/', views.analyzer_page,  name='analyzer'),
    path('analyze/',  views.analyze_skin,   name='analyze'),
    path('catalog/',  views.catalog,        name='catalog'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin-dashboard/add/', views.add_product, name='add_product'),
    path('admin-dashboard/edit/', views.edit_product, name='edit_product'),
    path('admin-dashboard/delete/', views.delete_product, name='delete_product'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('register/',views.register_user , name='register'),
    path('analyze-product/', views.analyze_product, name='analyze_product'),


]