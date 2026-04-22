import logging

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.shortcuts import render, redirect

logger = logging.getLogger('analyzer')


def user_login(request):
    error = None
    if request.method == 'POST':
        user = authenticate(request, username=request.POST.get('username'), password=request.POST.get('password'))
        if user:
            login(request, user)
            return redirect('admin_dashboard') if user.is_staff else redirect('home')
        else:
            error = 'Invalid username or password.'
    return render(request, 'login.html', {'error': error})


def user_logout(request):
    logout(request)
    return redirect('home')


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
            logger.info(f'New user registered: {username}')
        except IntegrityError:
            return render(request, 'login.html', {'reg_error': 'Username already exists.'})

    return redirect('login')
