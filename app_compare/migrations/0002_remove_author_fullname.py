# Generated by Django 2.1.dev20180416173837 on 2018-05-17 19:11

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app_compare', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='author',
            name='fullname',
        ),
    ]