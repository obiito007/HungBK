# Generated by Django 2.2.5 on 2019-11-05 12:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('VCS', '0002_thongtin'),
    ]

    operations = [
        migrations.AddField(
            model_name='teach',
            name='File',
            field=models.ImageField(null=True, upload_to=''),
        ),
    ]
