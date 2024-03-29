# Generated by Django 2.2.5 on 2019-11-03 12:08

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='List',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Giaovien', models.CharField(max_length=100)),
                ('Pubkey', models.CharField(max_length=100)),
                ('PubN', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Taokhoa',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('a', models.FloatField()),
                ('b', models.FloatField()),
                ('Ten', models.CharField(choices=[('Nguyễn Văn A', 'Nguyễn Văn A'), ('Trần Văn B', 'Trần Văn B'), ('Phạm Thị C', 'Phạm Thị C')], max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Teach',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('KhoaE', models.FloatField()),
                ('KhoaN', models.FloatField()),
                ('Link', models.TextField(max_length=1000)),
                ('Ghichu', models.CharField(max_length=100)),
                ('MSSV', models.CharField(default='', max_length=25)),
            ],
        ),
        migrations.CreateModel(
            name='Teacher',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Giaovien', models.CharField(choices=[('Nguyễn Văn A', 'Nguyễn Văn A'), ('Trần Văn B', 'Trần Văn B'), ('Phạm Thị C', 'Phạm Thị C')], max_length=1000)),
                ('image', models.ImageField(blank=True, null=True, upload_to='images/%Y/%m/%d/')),
                ('date', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Xac',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Giaovien', models.CharField(choices=[('Nguyễn Văn A', 'Nguyễn Văn A'), ('Trần Văn B', 'Trần Văn B'), ('Phạm Thị C', 'Phạm Thị C')], max_length=1000)),
            ],
        ),
    ]
