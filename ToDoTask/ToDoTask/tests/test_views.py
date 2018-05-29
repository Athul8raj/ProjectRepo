from django.test import RequestFactory
from django.urls import reverse
from Todos.models import Tasks
from mixer.backend.django import mixer
import pytest
from Todos.views import details


@pytest.mark.django_db
class TestDetailsViews:
    def test_detail_views(self):
        mixer.blend('ToDoTask.Tasks')
        path = reverse('details', kwargs={'id': 1})
        request = RequestFactory().get(path)
        request.user = mixer.blend(Tasks)

        response = details(request, id=1)
        assert response.status_code == 200
