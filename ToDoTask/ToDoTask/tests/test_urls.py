from django.urls import reverse, resolve


class TestUrls:
    def test_edit_urls(self):
        path = reverse('edit', kwargs={'username': 'Athul'})
        assert resolve(path).view_name == 'edit'

    def test_delete_urls(self):
        path = reverse('delete', kwargs={'username': 'Athul'})
        assert resolve(path).view_name == 'delete'

    def test_search_urls(self):
        path = reverse('search', kwargs={'username': 'Athul'})
        assert resolve(path).view_name == 'search'

    def test_tasks_urls(self):
        path = reverse('tasks', kwargs={'todo': ''})
        assert resolve(path).view_name == 'tasks'

    def test_details_urls(self):
        path = reverse('details', kwargs={'id': 1})
        assert resolve(path).view_name == 'details'

    def test_register_urls(self):
        path = reverse('register', kwargs={'id': 1})
        assert resolve(path).view_name == 'register'
