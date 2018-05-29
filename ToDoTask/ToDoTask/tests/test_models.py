from mixer.backend.django import mixer
import pytest


@pytest.mark.django_db
class TestModels:
    def test_title(self):
        title = mixer.blend('ToDoTask.Tasks', title='My First Task')
        assert title.__str__ == 'My First Task'
