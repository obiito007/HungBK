from django import forms
from django.forms import ModelForm,Textarea
from .models import Taokhoa,Teach,Teacher,List,Xac
import re
from django.contrib.auth.models import User	
from django.core.exceptions import ObjectDoesNotExist 

class TeachForm(forms.ModelForm):
	class Meta:
		model = Teach
		fields=['KhoaE','KhoaN','Ghichu','MSSV','Link']
		widgets = {
			'Link':Textarea(attrs={'cols':80,'rows':1})		
		}
		labels = {
			'KhoaE': ('Khóa bí mật'),
			'KhoaN': ('Khóa N'),
			'Link':('Path Image'),
		}


class TaokhoaForm(forms.ModelForm):
	class Meta:
		model = Taokhoa
		fields=['a','b','Ten','MSCB']
	def clean_Ten(self):
		Ten = self.cleaned_data['Ten']		
		c=List.objects.filter(Giaovien=Ten)
		if 'Ten' in self.cleaned_data:
			Ten = self.cleaned_data['Ten']
			if len(c)==0:
				return Ten
		raise forms.ValidationError('Mật khẩu không hợp lệ')
        
class UpForm(forms.ModelForm):
    class Meta:
        model = Teacher
        fields=['image','MSCB']

class ListForm(forms.ModelForm):
    class Meta:
        model = Xac
        fields=['MSCB']

class dangnhapForm(forms.Form):
	username = forms.CharField(label='Taikhoan', max_length=30)
	email = forms.EmailField(label='Email')
	password1 = forms.CharField(label='Mật Khẩu',widget=forms.PasswordInput())
	password2 = forms.CharField(label='Nhập lại Mật Khẩu',widget=forms.PasswordInput())

	def clean_password2(self):
		if 'password1' in self.cleaned_data:
			password1 = self.cleaned_data['password1']
			password2 = self.cleaned_data['password2'] 
			if password1==password2 and password1:
				return password2
		raise forms.ValidationError('Mật khẩu không hợp lệ')
		
	def clean_username(self):
		username = self.cleaned_data['username']
		if not re.search(r'^\w+$',username):
			raise forms.ValidationError("Tên tài khoản có kí tự đặc biệt")
		try:
			User.objects.get(username=username)
		except ObjectDoesNotExist:
			return username
		raise forms.ValidationError('Tài khoản đã tồn tại')

	def save(self):
		User.objects.create_user(username=self.cleaned_data['username'], email=self.cleaned_data['email'], password=self.cleaned_data['password1'])