# Generated by Django 2.0.5 on 2018-05-29 09:27

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('Todos', '0003_auto_20180529_1301'),
    ]

    operations = [
        migrations.AddField(
            model_name='tasks',
            name='created_by',
            field=models.ForeignKey(editable=False, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='tasks_created', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='tasks',
            name='modified_by',
            field=models.ForeignKey(editable=False, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='tasks_modified', to=settings.AUTH_USER_MODEL),
        ),
    ]
