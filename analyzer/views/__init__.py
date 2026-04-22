# Re-export all views so existing URL imports continue to work unchanged
from .pages import home, analyzer_page, catalog
from .auth import user_login, user_logout, register_user
from .admin_views import admin_dashboard, add_product, edit_product, delete_product
from .api import analyze_skin, analyze_product
