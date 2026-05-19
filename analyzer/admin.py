"""Register models with the Django admin panel so they can be managed through /admin/."""
from django.contrib import admin
from .models import Product, UserProfile

admin.site.register(Product)
admin.site.register(UserProfile)
