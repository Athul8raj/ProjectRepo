# Generated by Django 2.0.5 on 2018-05-29 07:31

from django.db import migrations
import multiselectfield.db.fields


class Migration(migrations.Migration):

    dependencies = [
        ('Todos', '0002_tasks_status'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tasks',
            name='status',
            field=multiselectfield.db.fields.MultiSelectField(choices=[('Done', 'Done'), ('Not Done', 'Not Done')], max_length=13),
        ),
    ]
