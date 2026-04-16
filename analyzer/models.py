from django.db import models

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