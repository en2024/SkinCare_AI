"""
Re-export all views so the URL config can import everything from one place.
Example: from . import views  →  views.home, views.analyze_skin, etc.
"""
from .pages import home, analyzer_page, catalog
from .auth import user_login, user_logout, register_user, profile_view
from .admin_views import admin_dashboard, add_product, edit_product, delete_product
from .api import analyze_skin, analyze_product, analyze_skin_hybrid, toggle_favorite
