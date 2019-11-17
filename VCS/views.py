from django.shortcuts import render,redirect
import math
from django.template import RequestContext
from django.http import HttpResponse
import random
from hashlib import sha1
import hashlib
from .forms import TeachForm
from .forms import TaokhoaForm,UpForm,ListForm,dangnhapForm
from .models import Teach,Teacher,List,Thongtin
import cv2 
import numpy as np
import sys
from django.template import loader
from django.contrib.auth import authenticate,decorators,login
from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import user_passes_test
import xlwt
from django.contrib.auth.models import User



# Create your views here.
#@decorators.login_required(login_url= '/login/')
def staff_required(login_url=None):
    return user_passes_test(lambda u: u.is_staff, login_url=login_url)

def index(request):
     return render(request,'VCS/home.html')

def danh(request):
    allaccs= Thongtin.objects.all()
    
    context= {'allaccs': allaccs}
    return render(request,'VCS/linhtinh3.html',context)

def coprime(o, p):
    while p!= 0:
        o, p = p, o % p
    return o
    
    
def extended_gcd(aa, bb):
    lastremainder, remainder = abs(aa), abs(bb)
    x, lastx, y, lasty = 0, 1, 1, 0
    while remainder:
        lastremainder, (quotient, remainder) = remainder, divmod(lastremainder, remainder)
        x, lastx = lastx - quotient*x, x
        y, lasty = lasty - quotient*y, y
    return lastremainder, lastx * (-1 if aa < 0 else 1), lasty * (-1 if bb < 0 else 1)
  
def modinv(a, m):
	g, x, y = extended_gcd(a, m)
	if g != 1:
		raise Exception('Modular inverse does not exist')
	return x % m    

        
def is_prime(num):
    if num == 2:
        return True
    if num < 2 or num % 2 == 0:
        return False
    for n in range(3, int(num**0.5)+2, 2):
        if num % n == 0:
            return False
    return True


def generate_keypair(p, q):
    if not (is_prime(p) and is_prime(q)):
        raise ValueError('Both numbers must be prime.')
    elif p == q:
        raise ValueError('p and q cannot be equal')

    n = p * q

    phi = (p-1) * (q-1)

    e = random.randrange(1, phi)

    g = coprime(e, phi)
  
    while g != 1:
        e = random.randrange(1, phi)
        g = coprime(e, phi)

    d = modinv(e, phi)
    pri=(e,n)
    pub=(d,n)
    return ((e), (d),n,pri,pub)
       
def hashFunction(message):
    hashed = sha1(message.encode("UTF-8")).hexdigest()
    #hashed = hash(message)
    return hashed
    
def encrypt(privatek, plaintext):
    key, n = privatek
    key=int(key)
    n=int(n)
    numberRepr = [ord(char) for char in plaintext]
    cipher = [pow(ord(char),key,n) for char in plaintext]
    return cipher
def decrypt(publick, ciphertext):
    key, n = publick
    key=int(key)
    n=int(n)  
    numberRepr = [pow(char, key, n) for char in ciphertext]
    plain = [chr(pow(char, key, n)) for char in ciphertext]

    #print("Decrypted number representation is: ", numberRepr)
    a=''.join(plain)
    #Return the array of bytes as a string
    return a
def hash_file(filename):
   h = hashlib.sha1()
   with open(filename,'rb') as file:
       chunk = 0
       while chunk != b'':
           chunk = file.read(1024)
           h.update(chunk)
   return h.hexdigest()


def taokhoa(form):
    a = form.a
    b = form.b
    Ten = form.Ten
    MSCB = form.MSCB
    public, private,n,t,y= generate_keypair(a, b) 
    #User = get_user_model()
    #user = User.objects.get(id=self.user.id)  
    user =Tennguoiky
    c = List(Giaovien=Ten,Pubkey = int(public),PubN= int(n),MSCB=MSCB,UserRegis=user)
    c.save()
    template = loader.get_template('VCS/linhtinh.html')
    context = {
        'public': int(public),'private': int(private),'n': int(n)
    }
    #response=HttpResponse()
    #response.writelines( "<h3>Khoa bi mat bang=%s </h3><br/>" %(public))
    #response.writelines( "<h3>khoa cong khai bang=%s</h3><br/>" %(private))
    #response.writelines( "<h3>n bằng tích của hai số nguyên tố:%s </h3><br/>" %(n))
    #return response
    return HttpResponse(template.render(context))
@staff_required(login_url= '/login/')
def teach(request):                          
    form = TeachForm(request.POST or None)
    if form.is_valid():
        form=form.save()
        return Ky(form)

    return render(request,'VCS/teach.html', {'form': form})


@staff_required(login_url= '/login/')
def index1(request):
    form = TaokhoaForm(request.POST or None)
    user = User.objects.get(username=request.user.username)
    global Tennguoiky
    Tennguoiky = user
    if request.method == 'POST':
        if form.is_valid():
            form = form.save(commit=False)
            return taokhoa(form)
    else:
        form = TaokhoaForm()

    return render(request,'VCS/khoa.html', {'form': form})

def Ky(form):
    d=form.KhoaE
    n=form.KhoaN
    a = form.Link
    a1=form.MSSV
    a2=form.Ghichu
    y=(d,n)
    np.set_printoptions(threshold=sys.maxsize)
    img = cv2.imread(a,cv2.IMREAD_GRAYSCALE)
    hash_object = hashlib.sha1(img)
    hex_dig = hash_object.hexdigest()
    s=hash_file(a)
    print(hex_dig)
    #print(s)
    #encrypted_msg = encrypt(y, s)
    encrypted_msg = encrypt(y, hex_dig)
    encrypted_msg1 =str( encrypted_msg)
    #print(encrypted_msg1)
    z = nhung(a,encrypted_msg,a1,a2)
    #print(z)
    if not z=='Hình này không phù hợp cho việc nhúng':
        a1 ='-sign'
        a2=a[:len(a)-4]
        #a3 = a[len(a)-4:len(a)]
        a3='.bmp'
        a4 = a2+a1+a3
        cv2.imwrite(a4,z)
        template = loader.get_template('VCS/linhtinh2.html')
        return HttpResponse(template.render())
        #response=HttpResponse()
        #response.writelines( "<h3>khoa bi mat bang=%s </h3><br/>" %(encrypted_msg1))
        #return response 
    else:
        template = loader.get_template('VCS/linhtinh4.html')
        return HttpResponse(template.render())

    

def list(request):
    Teachers = Teacher.objects.all()
    return render(request,'VCS/list.html',{'Teachers':Teachers})
#def Poss(request, id):
    #post = Teacher.objects.get(id=id)
    #return render(request,'VCS/post.html', {'post' : post})

def upload_file(request):
    if request.method == 'POST':                                
        form = UpForm(request.POST , files = request.FILES ) 
        if form.is_valid():
            form = form.save()                                 
            return redirect('upload_file') 
    else:
        form = UpForm()
 
    return render(request,'VCS/up.html',{'form':form})

def bang(request):
    allaccs= List.objects.all()
    
    context= {'allaccs': allaccs}

        
    return render(request, 'VCS/bang.html', context)

def xac(request):
    form = ListForm(request.POST or None)
    if form.is_valid():
        form = form.save()
        return xacthuc(form)

    return render(request,'VCS/xacthuc.html', {'form': form})

def xacthuc(form):
    b=form.MSCB
    a=Teacher.objects.filter(MSCB=b)
    c = List.objects.get(MSCB=b)
    c1=c.Pubkey
    c2=c.PubN
    pub=(float(c1),float(c2))
    for i in range(0,len(a)):
        try:
            [encrypted_msg,Ghichu,MSSV,im] = giainhung(a[i].image.path)
            date=a[i].date
            #print(encrypted_msg)
            d=decrypt(pub,encrypted_msg)
            hash_object = hashlib.sha1(im)
            hex_dig = hash_object.hexdigest()
            s=hex_dig
            #print('Giai mã chữ ký ra=',d)
            #print(encrypted_msg)
            #print('hash file gốc=',s)
            Ghichu=str(Ghichu)
            cv2.imwrite(r'D:\luanvan 11-12\filegoc.bmp',im)
            if s==d:
                Trangthai='File đc xác thực'
            else:
                Trangthai='File đã bị thay đổi'
            
            p=List.objects.get(MSCB=b)
            c = Thongtin(Giaovien=str(p.Giaovien),MSSV = str(MSSV),Trangthai=str(Trangthai) ,Ghichu=str(Ghichu),MSCB=b,date=str(date))
            c.save()

        except Exception:
            p=List.objects.get(MSCB=b)
            Giaovien=str(p.Giaovien)
            MSSV='None'
            MSCB=b
            Ghichu='None'
            date=a[i].date 
            Trangthai ='File đã bị thay đổi'     
            p=List.objects.get(MSCB=b)
            c = Thongtin(Giaovien=str(p.Giaovien),MSSV = str(MSSV),Trangthai=str(Trangthai) ,Ghichu=str(Ghichu),MSCB=b,date=str(date))
            c.save()    
        #(encrypted_msg,Ghichu,MSSV,im) = giainhung(a[i].image.path)
        #d=decrypt(pub,encrypted_msg)
        #hash_object = hashlib.sha1(im)
        #hex_dig = hash_object.hexdigest()
        #s=hex_dig
        #Ghichu=str(Ghichu)

        #if s==d:
            #Trangthai='File đc xác thực'
    #p=List.objects.get(MSCB=b)
    #c = Thongtin(Giaovien=str(p.Giaovien),MSSV = str(MSSV),Trangthai=str(Trangthai) ,Ghichu=str(Ghichu),MSCB=b)
    #c.save()
    
    template = loader.get_template('VCS/linhtinh3.html')
    allaccs= Thongtin.objects.all()
    
    context= {'allaccs': allaccs}

    
    return HttpResponse(template.render(context))
   
    #response=HttpResponse()
    #if s==d:
        #response.writelines( "<h4>file đã đc xác thực <br/>" )
    #else:
        #response.writelines( "<h4>file đã bị thay đổi <br/>" ) 
    #response.writelines( "<h3>hash=%s </h3><br/>" %(d) )
    #response.writelines( "<h3>khoa bi mat bang=%s </h3><br/>" %(c1) )
    #response.writelines( "<h3>khoa bi mat bang=%s </h3><br/>" %(c2) )
    #response.writelines( "<h3>khoa bi mat bang=%s </h3><br/>" %(p) )
    #return response 
#@decorators.login_required(login_url= '/login/')
def register(request):
    form = dangnhapForm()
    if request.method == 'POST':
        form = dangnhapForm(request.POST)
        if form.is_valid():
            form=form.save()
            return HttpResponseRedirect('/')
    return render(request,'VCS/register.html',{'form':form})

def export_users_xls(request):
    response = HttpResponse(content_type='application/ms-excel')
    response['Content-Disposition'] = 'attachment; filename="Danhsach.xls"'

    wb = xlwt.Workbook(encoding='utf-8')
    ws = wb.add_sheet('Users')

    # Sheet header, first row
    row_num = 0

    font_style = xlwt.XFStyle()
    font_style.font.bold = True

    columns = ['MSCB','Giáo viên', 'MSSV', 'Trạng thái', 'Ghi chú', ]

    for col_num in range(len(columns)):
        ws.write(row_num, col_num, columns[col_num], font_style)

    # Sheet body, remaining rows
    font_style = xlwt.XFStyle()

    rows = Thongtin.objects.all().values_list('MSCB','Giaovien', 'MSSV', 'Trangthai', 'Ghichu')
    for row in rows:
        row_num += 1
        for col_num in range(len(row)):
            ws.write(row_num, col_num, row[col_num], font_style)

    wb.save(response)
    return response

def nhung(link,sign,MSSV,Ghichu):
    np.set_printoptions(threshold=sys.maxsize)
    img = cv2.imread(link,cv2.IMREAD_GRAYSCALE)
    im = cv2.imread(link,cv2.IMREAD_GRAYSCALE)
    size = im.shape
    histogram = [0] * 256
    #q=np.histogram(im,bins=1,range=None)
    for row in range(size[0]): # traverse by row (y-axis)
            for col in range(size[1]): # traverse by column (x-axis)
                histogram[im[row, col]] += 1
    a = histogram
    b = np.arange(256)
    #print(histogram)
    #print(b)
    j1 = im[0,0];
    j2 = im[0,1];
    j3 = im[0,2];
    j4 = im[0,3];
    j5 = im[0,4];
    j6 = im[0,5];
    j7 = im[0,6];
    j8 = im[0,7];
    a[j1]=a[j1]-1;
    a[j2]=a[j2]-1;
    a[j3]=a[j3]-1;
    a[j4]=a[j4]-1;
    a[j5]=a[j5]-1;
    a[j6]=a[j6]-1;
    a[j7]=a[j7]-1;
    a[j8]=a[j8]-1;
    #print(a)
    
    #Tim peak va toa do peak
    max = a[0] + a[1]
    for i in range(1,255):
        if max<(a[i] + a[i+1]):
            max = (a[i]+a[i+1])
            peak = i
            giatripeak=a[i]
            giatriketiep=a[i+1]
    #print(peak)
    # chia histogram thanh 2 mien
    sub1 = a[0:peak+1]
    sub2=a[peak:256]

    #Tim minL,R va toa do minL,R
    sub11 = np.flip(sub1)
    q1 = min(sub11)
    w1 = np.argmin(sub11)
    w1 = len(sub11)-w1-1
    q2 = min(sub2)
    w2 = np.argmin(sub2)

    
    #Bieu dien gia tri nhi phan mien I1
    i1 = '{0:08b}'.format(im[0,0])
    i2 = '{0:08b}'.format(im[0,1])
    i3 = '{0:08b}'.format(im[0,2])
    i4 = '{0:08b}'.format(im[0,3])
    i5 = '{0:08b}'.format(im[0,4])
    i6 = '{0:08b}'.format(im[0,5])
    i7 = '{0:08b}'.format(im[0,6])
    i8 = '{0:08b}'.format(im[0,7])
    #Tap thong tin bo tro:
    #8 bit thap cua I1
    V = i1[7]+i2[7]+i3[7]+i4[7]+i5[7]+i6[7]+i7[7]+i8[7]
    #print(V)

    #Gia tri nhi phan diem cuc tieu ben trai peak
    minL = '{0:08b}'.format(w1)
    #print(minL)

    #So diem cuc tieu ben trai peak
    CL = '{0:08b}'.format(q1)
    #print(CL)
    
    #Vi tri cac diem co gia tri minL
    for i in range(0,8):
        im[0][i] = 257
    
    ML = np.argwhere(im == w1)
    #print(ML)
    
    #Gia tri nhi phan diem cuc tieu ben phai peak
    minR = '{0:08b}'.format(w2+peak)
    #print(minR)

    # So diem cuc tieu ben phai peak
    CR = '{0:08b}'.format(q2)
    #print(CR)

    #Vi tri cac diem co gia tri minR
    MR = np.argwhere(im == w2+peak)
    #print(MR)
    bin9 = lambda x : ''.join(reversed( [str((x >> i) & 1) for i in range(9)] ) )
    chieudaiB = giatripeak + giatriketiep
    xixo1 = ''
    xixo2 = ''
    for i in range(1,len(ML)+1):
        xixo1 = xixo1 + bin9(ML[i-1,0]) + bin9(ML[i-1,1])

    for i in range(1,len(MR)+1):
        xixo2 = xixo2 + bin9(MR[i-1,0]) + bin9(MR[i-1,1])
   
    MLL = xixo1
    MRR = xixo2
    #print(MLL,MRR)
    thongtinphu= V + minL + CL + MLL + minR + CR + MRR 
    chieudainhung = chieudaiB - len(thongtinphu)
    #print('chieu dai nhung la :',chieudainhung)

    if chieudainhung > 750:
        chuki=sign
        MSSV=MSSV
        Ghichu=Ghichu
        bin7 = lambda x : ''.join(reversed( [str((x >> i) & 1) for i in range(7)] ) )
        bin10 = lambda x : ''.join(reversed( [str((x >> i) & 1) for i in range(10)] ) )
        MSSV=[ord(c) for c in MSSV]
        MSSV=[(bin7(a)) for a in MSSV]
        MSSV=''.join(MSSV)
        a= [ord(c) for c in Ghichu]
        a=[(bin7(a)) for a in a]
        a=''.join(a)
        b=len(a)
        b1=b
        b=bin10(b)
        a=a+b
        Ghichu=a
        bin14 = lambda x : ''.join(reversed( [str((x >> i) & 1) for i in range(14)] ) )
        W=''
        for i in range(0,40):
            W = W + bin14(chuki[i])

        zeros = [0] * (chieudainhung-560-b1-49-10)
        zeros = ''.join(str(e) for e in zeros)
        #print(MSSV)
        #print(Ghichu)
        W = zeros +MSSV+Ghichu+ W
        #print(W)
        B= thongtinphu + W
        #print(thongtinphu)
        #Tao cac cap histogram(peak,peak-1) và (peak+1,peak+2)bang dich chuyen histogram
        for i in range(0,size[0]):
            for j in range(0,size[1]):
                if im[i,j] in range(w1+1,peak):
                    im[i,j] = im[i,j]-1

        for i in range(0,size[0]):
            for j in range(0,size[1]):
                if im[i,j] in range(peak+2,w2+peak):
                    im[i,j] = im[i,j]+1

        #nhung day bit B
        k=-1
        for i in range(0,size[0]):
            for j in range(0,size[1]):
                if im[i,j] in range(peak,peak+2):
                    k = k+1
                    #print(k)
                    if int(B[k]) == 0:
                        im[i,j] = im[i,j]
                    elif int(B[k]) == 1:
                        if im[i,j] == peak:
                            im[i,j] = im[i,j]-1
                        else:
                            im[i,j] = im[i,j]+1
                            
        for i in range(0,8):
            im[0,i] = img[0,i]




        #nhung peak vao i1
        mask = 1 << 0 
        peak = '{0:08b}'.format(peak)
        for i in range(0,8):
            im[0,i] = (im[0,i] & ~mask) | ((int(peak[i]) << 0) & mask)

        #cv2.imwrite('lan2.tiff',im)
        #print(size)
        #print(len(B))
        #cv2.imshow('image',im)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return im
    else:
        a='Hình này không phù hợp cho việc nhúng'
        return a
    

def giainhung(anh):
        np.set_printoptions(threshold=sys.maxsize)
        img = cv2.imread(anh,cv2.IMREAD_GRAYSCALE)
        im = cv2.imread(anh,cv2.IMREAD_GRAYSCALE)
        size = im.shape

        #buoc2
        i1 = '{0:08b}'.format(im[0,0])
        i2 = '{0:08b}'.format(im[0,1])
        i3 = '{0:08b}'.format(im[0,2])
        i4 = '{0:08b}'.format(im[0,3])
        i5 = '{0:08b}'.format(im[0,4])
        i6 = '{0:08b}'.format(im[0,5])
        i7 = '{0:08b}'.format(im[0,6])
        i8 = '{0:08b}'.format(im[0,7])

        peak = i1[7] + i2[7] + i3[7] + i4[7] + i5[7] + i6[7] + i7[7] + i8[7] 
        peak = int(peak,2)
        #print(peak)


        for i in range(0,8):
                im[0,i] = 256

        #buoc3
        B=''
        k=-1
        for i in range(0,size[0]):
                for j in range(0,size[1]):
                        if im[i,j] in range(peak-1,peak+3):
                                k = k+1
                                if im[i,j] in range(peak,peak+2):
                                        B = B + '0'
                                else:
                                        B = B + '1'

        #print('day nhung B la',B)
        #buoc4
        #tim V
        V=''
        for i in range(0,8):
                V=V + B[i]
        #print('Gia tri của V',V)

        #tim minL
        minL=''
        for i in range(8,16):
                minL=minL + B[i]
        #print('Gia tri của minL',minL)

        #tim CL
        CL=''
        for i in range(16,24):
                CL=CL + B[i]
        #print('Gia tri của CL',CL)

        #tim ML
        lem = 9*2*int(CL,2)
        ML =''
        for i in range(24,24+lem):
                ML=ML + B[i]
        #print('Gia tri của ML',ML)

        #tim minR
        BL=''
        for i in range(24+lem,len(B)):
                BL=BL + B[i]
        
        minR = ''
        for i in range(0,8):
                minR=minR + BL[i]
        #print('Gia tri của minR',minR)

        #tim CR
        CR = ''
        for i in range(8,16):
                CR=CR + BL[i]
        #print('Gia tri của CR',CR)

        #tim MR
        lemm = 9*2*int(CR,2)
        MR =''
        for i in range(16,16+lemm):
                MR=MR + BL[i]
        #print('Gia tri của MR',MR)
        #print(len(MR))

        # tim W
        W=''
        for i in range(16+lemm,len(BL)):
                W=W + BL[i]

        #print(len(W))

        minL = int(minL,2)
        minR = int(minR,2)
        #print(len(W))
        #buoc 5,2
        for i in range(0,8):
                im[0,i] =999

        for i in range(0,size[0]):
                for j in range(0,size[1]):
                        if im[i,j] in range(minL+1,minR):
                                if im[i,j] < peak:
                                        im[i,j] = im[i,j] +1
                                elif im[i,j] > (peak +1):
                                        im[i,j] = im[i,j] -1 


        MLL = ''
        LM1 = ','
        LM=''
        for i in range(0,len(ML),9):
                MLL1 =MLL + ML[i]+ML[i+1]+ML[i+2]+ML[i+3]+ML[i+4]+ML[i+5]+ML[i+6]+ML[i+7]+ML[i+8]
                LM = LM +LM1 + str(int(MLL1,2))
                
        LM = LM[1:]
        LM = LM.split(',')
        #print(LM)


        MRR = ''
        RM1 = ','
        RM = ''
        for i in range(0,len(MR),9):
                MRR1 =MRR + MR[i]+MR[i+1]+MR[i+2]+MR[i+3]+MR[i+4]+MR[i+5]+MR[i+6]+MR[i+7]+MR[i+8]
                RM = RM + RM1 + str(int(MRR1,2))
        RM = RM[1:]
        RM = RM.split(',')
        #print(RM)

        #buoc 5.3 khoi phuc cac diem anh co gia tri minL
        tet = np.argwhere(im == minL)
        #print(tet)
        for i in range(0,len(tet)):
                im[tet[i,0],tet[i,1]] = im[tet[i,0],tet[i,1]]+1
        if LM != ['']:
                for i in range(0,len(LM),2):
                        im[int(LM[i]),int(LM[i+1])] = im[int(LM[i]),int(LM[i+1])]-1


        #buoc 5.3 khoi phuc cac diem anh co gia tri minR
        tet = np.argwhere(im == minR)
        #print(tet)
        #print(len(tet))
        for i in range(0,len(tet)):
                im[tet[i,0],tet[i,1]] = im[tet[i,0],tet[i,1]]-1
        if RM != ['']:
                for i in range(0,len(RM),2):
                        im[int(RM[i]),int(RM[i+1])] = im[int(RM[i]),int(RM[i+1])]+1
        
        tet = np.argwhere(im == minR)
        #print(tet)
        #print(len(tet))

        for i in range(0,8):
                im[0,i] = img[0,i]
        mask = 1 << 0 

        for i in range(0,8):
                im[0,i] = (im[0,i] & ~mask) | ((int(V[i]) << 0) & mask)


        
        sign=W[len(W)-560:]
        lenG=W[len(W)-560-10:len(W)-560]
        lenG=int(lenG,2)
        #print(lenG)
        Ghichu=W[len(W)-560-10-lenG:len(W)-560-10]
        MSSV=W[len(W)-560-10-lenG-49:len(W)-560-10-lenG]
        W1=Ghichu
        WWW=MSSV
        #print(len(W1))
        #print(WWW)


        W=sign
        
        e=''
        p=''
        q=','
        for i in range(0,560,14):
                e = W[i]+W[i+1]+W[i+2]+W[i+3]+W[i+4]+W[i+5]+W[i+6]+W[i+7]+W[i+8]+W[i+9]+W[i+10]+W[i+11]+W[i+12]+W[i+13]
                p = p + str(int(e,2))+q
        p = p.split(',')
        p = p[:40]
        p= [int(i) for i in p]
        chuki=p
        #print(p)

        e=''
        p=''
        q=','
        for i in range(0,len(W1),7):
            e = W1[i]+W1[i+1]+W1[i+2]+W1[i+3]+W1[i+4]+W1[i+5]+W1[i+6]
            p = p + str(int(e,2))+q
        p = p.split(',')
        p = p[:int(len(W1)/7)]
        p= [int(i) for i in p]
        p= [chr(p) for p in p]
        e=''
        for i in range(0,len(p)):
            e=e+str(p[i])
        Ghichu=e

        e=''
        p=''
        q=','
        for i in range(0,len(WWW),7):
            e = WWW[i]+WWW[i+1]+WWW[i+2]+WWW[i+3]+WWW[i+4]+WWW[i+5]+WWW[i+6]
            p = p + str(int(e,2))+q
        p = p.split(',')
        p = p[:int(len(WWW)/7)]
        p= [int(i) for i in p]
        p= [chr(p) for p in p]
        e=''
        for i in range(0,len(p)):
            e=e+str(p[i])
        MSSV=e

       
        #cv2.imwrite(r'D:\luanvan 11-12\hinh4trich.bmp',im)
        #cv2.imshow('image',im)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return (chuki,Ghichu,MSSV,im)

def ok(request):
    return render(request,'VCS/login.html')