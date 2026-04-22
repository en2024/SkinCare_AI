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
    return render(request, 'catalog.html', {'products': products, 'MEDIA_URL': settings.MEDIA_URL})
