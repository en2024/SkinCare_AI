from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect

from ..models import Product


@login_required(login_url='login')
def admin_dashboard(request):
    if not request.user.is_staff:
        return redirect('home')
    return render(request, 'admin.html', {'products': Product.objects.all()})


@login_required(login_url='login')
def add_product(request):
    if request.method == 'POST' and request.user.is_staff:
        Product.objects.create(
            name=request.POST.get('name'),
            category=request.POST.get('category'),
            skin_type=request.POST.get('skin_type'),
            description=request.POST.get('description', ''),
            how_to_use=request.POST.get('how_to_use', ''),
            image=request.FILES.get('image')
        )
    return redirect('/admin-dashboard/')


@login_required(login_url='login')
def edit_product(request):
    if request.method == 'POST' and request.user.is_staff:
        p = Product.objects.get(id=request.POST.get('product_id'))
        p.name = request.POST.get('name')
        p.category = request.POST.get('category')
        p.skin_type = request.POST.get('skin_type')
        p.description = request.POST.get('description')
        p.how_to_use = request.POST.get('how_to_use', '')
        if request.FILES.get('image'):
            p.image = request.FILES.get('image')
        p.save()
    return redirect('/admin-dashboard/')


@login_required(login_url='login')
def delete_product(request):
    if request.method == 'POST' and request.user.is_staff:
        Product.objects.filter(id=request.POST.get('product_id')).delete()
    return redirect('/admin-dashboard/')
