import django_filters
from .models import Tasks


class UserFilter(django_filters.FilterSet):
    class Meta:
        model = Tasks
        fields = ['status', ]
