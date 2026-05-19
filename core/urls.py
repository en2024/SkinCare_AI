"""
Root URL configuration.
All analyzer routes (pages, API, auth, admin) are handled by the analyzer app.
The built-in Django admin panel is available at /admin/.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('analyzer.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
