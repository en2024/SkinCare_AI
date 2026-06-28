"""
Page views — simple views that render HTML templates.
No heavy logic here; just fetch data and pass it to the template.
"""
from django.shortcuts import render
from django.conf import settings
from django.views.decorators.csrf import ensure_csrf_cookie

from ..models import Product


@ensure_csrf_cookie
def clinical_scan_view(request):
    """Clinical skin scan page."""
    return render(request, 'clinical_scan.html')


def home(request):
    """Landing page."""
    return render(request, 'index.html')


@ensure_csrf_cookie
def analyzer_page(request):
    """AI skin analyzer page (needs CSRF cookie for the JS fetch calls)."""
    return render(request, 'analyzer.html')


def catalog(request):
    """Product catalog — shows all products and highlights the user's favorites."""
    products = Product.objects.all()

    # Build a set of product IDs the logged-in user has favorited
    # so the template can show filled hearts on those cards
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
