from django.shortcuts import render
from django.conf import settings
from django.views.decorators.csrf import ensure_csrf_cookie

from ..models import Product


def home(request):
    return render(request, 'index.html')


@ensure_csrf_cookie
def analyzer_page(request):
    return render(request, 'analyzer.html')


def catalog(request):
    products = Product.objects.all()
    favorite_ids = set()
    if request.user.is_authenticated:
        from ..models import UserProfile
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        favorite_ids = set(profile.favorite_products.values_list('id', flat=True))
    return render(request, 'catalog.html', {
        'products': products,
        'MEDIA_URL': settings.MEDIA_URL,
        'favorite_ids': favorite_ids,
    })
