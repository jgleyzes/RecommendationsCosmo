from app_compare.models import Author
import django_filters

class UserFilter(django_filters.FilterSet):
    class Meta:
        model = Author
        fields = ['name', ]
