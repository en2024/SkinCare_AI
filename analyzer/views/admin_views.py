# Admin views — these handle the product management dashboard.
# Only staff users (admins) can access these pages.
# They provide CRUD operations (Create, Read, Update, Delete) for products.
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404

from ..models import Product


# Dashboard view — shows all products to the admin.
# login_required ensures only logged-in users can access it.
# We also check is_staff to make sure it's an admin, not a regular user.
@login_required(login_url='login')
def admin_dashboard(request):
    if not request.user.is_staff:
        return redirect('home')
    return render(request, 'admin.html', {'products': Product.objects.all()})


# Add product — creates a new product from the form data.
# The image is handled separately via request.FILES.
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


# Edit product — updates an existing product's details.
# Only replaces the image if a new one was uploaded.
@login_required(login_url='login')
def edit_product(request):
    if request.method == 'POST' and request.user.is_staff:
        p = get_object_or_404(Product, id=request.POST.get('product_id'))
        p.name = request.POST.get('name')
        p.category = request.POST.get('category')
        p.skin_type = request.POST.get('skin_type')
        p.description = request.POST.get('description')
        p.how_to_use = request.POST.get('how_to_use', '')
        if request.FILES.get('image'):
            p.image = request.FILES.get('image')
        p.save()
    return redirect('/admin-dashboard/')


# Delete product — removes a product from the database.
@login_required(login_url='login')
def delete_product(request):
    if request.method == 'POST' and request.user.is_staff:
        Product.objects.filter(id=request.POST.get('product_id')).delete()
    return redirect('/admin-dashboard/')
