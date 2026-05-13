# Auth views for login, logout, registration, and profile

import logging

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.shortcuts import render, redirect

logger = logging.getLogger('analyzer')


# login view - checks user and redirects

def user_login(request):
    error = None
    if request.method == 'POST':
        user = authenticate(request, username=request.POST.get('username'), password=request.POST.get('password'))
        if user:
            login(request, user)
            return redirect('home')
        else:
            error = 'Invalid username or password.'
    return render(request, 'login.html', {'error': error})


# simple logout

def user_logout(request):
    logout(request)
    return redirect('home')


# create new user account

def register_user(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirm_password', '')
        full_name = request.POST.get('full_name', '').strip()

        # Validation
        if not username or not password:
            return render(request, 'login.html', {'reg_error': 'Username and password are required.'})

        if password != confirm_password:
            return render(request, 'login.html', {'reg_error': 'Passwords do not match.'})

        if len(password) < 6:
            return render(request, 'login.html', {'reg_error': 'Password must be at least 6 characters.'})

        try:
            user = User.objects.create_user(username=username, password=password)
            # Save the full name if provided
            if full_name:
                parts = full_name.split(' ', 1)
                user.first_name = parts[0]
                user.last_name = parts[1] if len(parts) > 1 else ''
                user.save()
                
            # Save skin_type to profile
            skin_type = request.POST.get('skin_type', '')
            if skin_type:
                from ..models import UserProfile
                profile, _ = UserProfile.objects.get_or_create(user=user)
                profile.skin_type = skin_type
                profile.save()
                
            logger.info(f'New user registered: {username}')
            login(request, user)
            return redirect('home')
        except IntegrityError:
            return render(request, 'login.html', {'reg_error': 'Username already exists.'})

    return redirect('login')


# user profile settings

def profile_view(request):
    """User profile page: account settings, skin type, favorites."""
    if not request.user.is_authenticated:
        return redirect('login')

    from ..models import UserProfile
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    message = None

    if request.method == 'POST':
        action = request.POST.get('action', '')

        if action == 'update_info':
            full_name = request.POST.get('full_name', '').strip()
            if full_name:
                parts = full_name.split(' ', 1)
                request.user.first_name = parts[0]
                request.user.last_name = parts[1] if len(parts) > 1 else ''
                request.user.save()
            message = 'Profile updated successfully.'

        elif action == 'update_skin_type':
            skin_type = request.POST.get('skin_type', '')
            profile.skin_type = skin_type
            profile.save()
            message = 'Skin type updated successfully.'

        elif action == 'change_password':
            current = request.POST.get('current_password', '')
            new_pw = request.POST.get('new_password', '')
            confirm_pw = request.POST.get('confirm_password', '')
            if not request.user.check_password(current):
                message = 'Current password is incorrect.'
            elif new_pw != confirm_pw:
                message = 'New passwords do not match.'
            elif len(new_pw) < 6:
                message = 'Password must be at least 6 characters.'
            else:
                request.user.set_password(new_pw)
                request.user.save()
                login(request, request.user)
                message = 'Password changed successfully.'

        elif action == 'remove_favorite':
            from ..models import Product
            pid = request.POST.get('product_id')
            try:
                product = Product.objects.get(id=pid)
                profile.favorite_products.remove(product)
                message = f'Removed {product.name} from favorites.'
            except Exception:
                message = 'Product not found.'

    favorites = profile.favorite_products.all()
    return render(request, 'profile.html', {
        'profile': profile,
        'favorites': favorites,
        'message': message,
    })
