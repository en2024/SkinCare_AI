# Here I'm defining the database models for our project.
# Django uses these classes to automatically create the database tables.
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


# Product model — this stores all the skincare products in our catalog.
# Each product has a category (moisturizer, cleanser, etc.) and a target
# skin type, so the AI can recommend the right products to each user.
class Product(models.Model):
    CATEGORY_CHOICES = [
        ('moisturizer', 'Moisturizer'),
        ('cleanser', 'Cleanser'),
        ('serum', 'Serum'),
        ('sunscreen', 'Sunscreen'),
    ]
    SKIN_TYPE_CHOICES = [
        ('oily', 'Oily'),
        ('dry', 'Dry'),
        ('normal', 'Normal'),
        ('combination', 'Combination'),
        ('sensitive', 'Sensitive'),
        ('all', 'All Skin Types'),
    ]
    name = models.CharField(max_length=200)
    category = models.CharField(max_length=100, choices=CATEGORY_CHOICES)
    skin_type = models.CharField(max_length=100, choices=SKIN_TYPE_CHOICES)
    description = models.TextField()
    how_to_use = models.TextField()
    image = models.ImageField(upload_to='products/', blank=True)

    def __str__(self):
        return self.name


# UserProfile model — extends Django's built-in User model with extra fields.
# We store the user's skin type and their favorite products here.
# This is linked 1-to-1 with the User model using OneToOneField.
class UserProfile(models.Model):
    """Extended user profile for skin type and product favorites."""
    SKIN_TYPE_CHOICES = [
        ('oily', 'Oily'),
        ('dry', 'Dry'),
        ('normal', 'Normal'),
        ('combination', 'Combination'),
        ('sensitive', 'Sensitive'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    skin_type = models.CharField(max_length=50, choices=SKIN_TYPE_CHOICES, blank=True, default='')
    favorite_products = models.ManyToManyField(Product, blank=True, related_name='favorited_by')

    def __str__(self):
        return f'{self.user.username} Profile'


# Django signal — whenever a new User is created (e.g. during registration),
# this automatically creates a matching UserProfile for them.
# This way we never have to worry about a user missing their profile.
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Auto-create a UserProfile when a new User is created."""
    if created:
        UserProfile.objects.get_or_create(user=instance)