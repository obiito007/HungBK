from django.db import models

# Create your models here.
class Taokhoa(models.Model):
    teach_choice = (
        ('Nguyễn Văn A','Nguyễn Văn A'),
        ('Trần Văn B','Trần Văn B'),
        ('Phạm Thị C','Phạm Thị C'),
    )
    a = models.FloatField()
    b = models.FloatField()
    Ten = models.CharField(max_length=100,choices=teach_choice)
    MSCB = models.CharField(null=True,max_length=7)


class Teach(models.Model):
    KhoaE = models.FloatField()
    KhoaN = models.FloatField()
    Link = models.TextField(max_length=1000)
    Ghichu=models.CharField(max_length=100)
    MSSV = models.CharField(max_length=25,default='')
    File = models.ImageField(null=True,upload_to='Taokhoa/')

    
class Teacher(models.Model):
    teach_choice = (
        ('Nguyễn Văn A','Nguyễn Văn A'),
        ('Trần Văn B','Trần Văn B'),
        ('Phạm Thị C','Phạm Thị C'),
    )
    Giaovien = models.CharField(max_length=1000,choices=teach_choice)
    image= models.ImageField(upload_to='images/%Y/%m/%d/', blank=True, null=True ) #1
    date=models.DateTimeField(auto_now_add=True)
    MSCB = models.CharField(max_length=7,null=True)
    def __str__(self):
        return self.MSCB

class List(models.Model):
    Giaovien = models.CharField(max_length= 100)
    Pubkey = models.CharField(max_length=100)
    PubN = models.CharField(max_length= 100)
    MSCB = models.CharField(max_length=7,null=True)
    UserRegis = models.CharField(max_length=100,null=True)
    def __str__(self):
        return self.Giaovien
    
class Xac(models.Model):
    teach_choice = (
        ('Nguyễn Văn A','Nguyễn Văn A'),
        ('Trần Văn B','Trần Văn B'),
        ('Phạm Thị C','Phạm Thị C'),
    )
    Giaovien = models.CharField(max_length=1000,choices=teach_choice)
    MSCB = models.CharField(max_length=7,null=True)

class Thongtin(models.Model):
    Giaovien=models.CharField(max_length=100)
    MSSV = models.CharField(max_length=100)
    Trangthai =models.CharField(max_length=20)
    Ghichu = models.CharField(max_length=7)
    MSCB = models.CharField(max_length=7,null=True)
    date = models.CharField(max_length=100,null=True)