from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views
urlpatterns = [
    path('', views.home, name='home'),
    path('analyzer/', views.analyzer_page, name='analyzer'),
    path('catalog/', views.catalog, name='catalog'),
    path('analyze/', views.analyze_skin, name='analyze'),
    path('analyze-hybrid/', views.analyze_skin_hybrid, name='analyze_hybrid'),
    path('analyze-product/', views.analyze_product, name='analyze_product'),
    path('toggle-favorite/', views.toggle_favorite, name='toggle_favorite'),
    # login & sign up
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('register/', views.register_user, name='register'),
    path('profile/', views.profile_view, name='profile'),
    # admin
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin-dashboard/add/', views.add_product, name='add_product'),
    path('admin-dashboard/edit/', views.edit_product, name='edit_product'),
    path('admin-dashboard/delete/', views.delete_product, name='delete_product'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)