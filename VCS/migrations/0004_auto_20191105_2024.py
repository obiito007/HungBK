# Generated by Django 2.2.5 on 2019-11-05 13:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('VCS', '0003_teach_file'),
    ]

    operations = [
        migrations.AlterField(
            model_name='thongtin',
            name='Ghichu',
            field=models.CharField(max_length=7),
        ),
        migrations.AlterField(
            model_name='thongtin',
            name='Trangthai',
            field=models.CharField(max_length=20),
        ),
    ]