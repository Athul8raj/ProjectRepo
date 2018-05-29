from django.db import models
from multiselectfield import MultiSelectField
from datetime import datetime
from django.contrib.auth.models import User
from .util import get_current_user


class Tasks(models.Model):
    title = models.CharField(max_length=200)
    desc = models.TextField()
    created_at = models.DateTimeField(default=datetime.now, blank=True)
    STATUS = (
        ('Done', 'Done'),
        ('Not Done', 'Not Done'))

    status = MultiSelectField(choices=STATUS)

    def __str__(self):
        return self.title

    created_by = models.ForeignKey(User, on_delete=models.PROTECT, null=True, editable=False, related_name='%(class)s_created')
    modified_by = models.ForeignKey(User, on_delete=models.PROTECT, null=True, editable=False, related_name='%(class)s_modified')

    def save(self, *args, **kwargs):
        user = get_current_user()
        if user:
            self.modified_by = user
            if not self.id:
                self.created_by = user
        super(Tasks, self).save(*args, **kwargs)
