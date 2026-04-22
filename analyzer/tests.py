from django.test import TestCase, Client
from django.contrib.auth.models import User
from .models import Product


class ProductModelTest(TestCase):
    """Tests for the Product model."""

    def test_create_product(self):
        """Test that a product can be created with all fields."""
        product = Product.objects.create(
            name='Test Moisturizer',
            category='moisturizer',
            skin_type='oily',
            description='Test description',
            how_to_use='Apply daily'
        )
        self.assertEqual(product.name, 'Test Moisturizer')
        self.assertEqual(product.category, 'moisturizer')
        self.assertEqual(product.skin_type, 'oily')
        self.assertEqual(str(product), 'Test Moisturizer')

    def test_product_str(self):
        """Test the string representation of a Product."""
        product = Product.objects.create(
            name='CeraVe Cleanser',
            category='cleanser',
            skin_type='all',
            description='A good cleanser',
            how_to_use='Massage onto face'
        )
        self.assertEqual(str(product), 'CeraVe Cleanser')


class PageViewTests(TestCase):
    """Tests that all public pages return 200."""

    def setUp(self):
        self.client = Client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_analyzer_page(self):
        response = self.client.get('/analyzer/')
        self.assertEqual(response.status_code, 200)

    def test_catalog_page(self):
        response = self.client.get('/catalog/')
        self.assertEqual(response.status_code, 200)

    def test_login_page(self):
        response = self.client.get('/login/')
        self.assertEqual(response.status_code, 200)


class AuthTests(TestCase):
    """Tests for authentication flow."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpass123')
        self.admin = User.objects.create_superuser(username='admin', password='adminpass123')

    def test_login_valid_user(self):
        response = self.client.post('/login/', {
            'username': 'testuser',
            'password': 'testpass123'
        })
        self.assertEqual(response.status_code, 302)  # redirect after login

    def test_login_invalid_credentials(self):
        response = self.client.post('/login/', {
            'username': 'testuser',
            'password': 'wrongpassword'
        })
        self.assertEqual(response.status_code, 200)  # stays on login page

    def test_register_new_user(self):
        response = self.client.post('/register/', {
            'username': 'newuser',
            'password': 'newpass123',
            'confirm_password': 'newpass123',
            'full_name': 'Jane Doe'
        })
        self.assertEqual(response.status_code, 302)  # redirect to login
        self.assertTrue(User.objects.filter(username='newuser').exists())

    def test_register_duplicate_username(self):
        response = self.client.post('/register/', {
            'username': 'testuser',  # already exists
            'password': 'newpass123',
            'confirm_password': 'newpass123',
        })
        self.assertEqual(response.status_code, 200)  # stays on page with error

    def test_register_password_mismatch(self):
        response = self.client.post('/register/', {
            'username': 'mismatchuser',
            'password': 'pass123',
            'confirm_password': 'pass456',
        })
        self.assertEqual(response.status_code, 200)  # stays on page with error
        self.assertFalse(User.objects.filter(username='mismatchuser').exists())

    def test_register_short_password(self):
        response = self.client.post('/register/', {
            'username': 'shortpwuser',
            'password': '12345',
            'confirm_password': '12345',
        })
        self.assertEqual(response.status_code, 200)  # stays on page with error
        self.assertFalse(User.objects.filter(username='shortpwuser').exists())

    def test_logout(self):
        self.client.login(username='testuser', password='testpass123')
        response = self.client.get('/logout/')
        self.assertEqual(response.status_code, 302)  # redirect to home


class AdminDashboardTests(TestCase):
    """Tests for admin product management."""

    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create_superuser(username='admin', password='adminpass123')
        self.regular_user = User.objects.create_user(username='user', password='userpass123')

    def test_admin_dashboard_requires_login(self):
        response = self.client.get('/admin-dashboard/')
        self.assertEqual(response.status_code, 302)  # redirect to login

    def test_admin_dashboard_requires_staff(self):
        self.client.login(username='user', password='userpass123')
        response = self.client.get('/admin-dashboard/')
        self.assertEqual(response.status_code, 302)  # redirect to home

    def test_admin_dashboard_accessible_by_staff(self):
        self.client.login(username='admin', password='adminpass123')
        response = self.client.get('/admin-dashboard/')
        self.assertEqual(response.status_code, 200)

    def test_add_product(self):
        self.client.login(username='admin', password='adminpass123')
        response = self.client.post('/admin-dashboard/add/', {
            'name': 'New Product',
            'category': 'serum',
            'skin_type': 'oily',
            'description': 'A new serum',
            'how_to_use': 'Apply at night',
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Product.objects.filter(name='New Product').exists())
        product = Product.objects.get(name='New Product')
        self.assertEqual(product.how_to_use, 'Apply at night')

    def test_delete_product(self):
        self.client.login(username='admin', password='adminpass123')
        product = Product.objects.create(
            name='To Delete', category='cleanser', skin_type='dry',
            description='delete me', how_to_use='n/a'
        )
        response = self.client.post('/admin-dashboard/delete/', {
            'product_id': product.id
        })
        self.assertEqual(response.status_code, 302)
        self.assertFalse(Product.objects.filter(id=product.id).exists())


class APIEndpointTests(TestCase):
    """Tests for AI analysis API endpoints."""

    def test_analyze_skin_requires_post(self):
        response = self.client.get('/analyze/')
        self.assertEqual(response.status_code, 405)  # Method Not Allowed

    def test_analyze_product_requires_post(self):
        response = self.client.get('/analyze-product/')
        self.assertEqual(response.status_code, 405)  # Method Not Allowed

    def test_analyze_skin_requires_image(self):
        response = self.client.post('/analyze/')
        self.assertEqual(response.status_code, 400)

    def test_analyze_product_requires_ingredients(self):
        response = self.client.post('/analyze-product/', {
            'skin_type': 'oily'
        })
        self.assertEqual(response.status_code, 400)
