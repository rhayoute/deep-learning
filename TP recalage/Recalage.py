import numpy as np
from scipy import signal
from scipy import interpolate
from PIL import Image

def getGaussians() :
    n=21
    sigma=0.3
    [X,Y]=np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n), indexing='xy')
    Z=np.sqrt(X*X+Y*Y)
    im1=np.zeros((n,n))
    im1[Z<=.7]=1.
    im1[Z<=.3]=.5
    im1[Z<=.1]=.7
    im2=np.zeros((n,n));
    Z=np.sqrt((X-.3)**2+(Y+.2)**2)
    im2[Z<=.7]=1
    im2[Z<=.3]=.5
    im2[Z<=.1]=.7
    G=np.fft.fftshift(np.exp(-(X**2+Y**2)/sigma**2))
    f=np.real(np.fft.ifft2(np.fft.fft2(G)*np.fft.fft2(im1)))
    g=np.real(np.fft.ifft2(np.fft.fft2(G)*np.fft.fft2(im2))) 
    f=f/np.max(f)
    g=g/np.max(g)
    return f,g,(X,Y)

def interpol(f,u) :
    # function that computes f \circ Id+u and interpolates it on a mesh
    (ux,uy)=u
    nx,ny=f.shape
    ip=interpolate.RectBivariateSpline(np.arange(nx),np.arange(ny),f)
    [X,Y]=np.meshgrid(np.arange(nx),np.arange(ny), indexing='ij')
    X=X+ux
    Y=Y+uy
    return np.reshape(ip.ev(X.ravel(),Y.ravel()),(nx,ny))

def upscale(f,factor) :
    nx,ny=f.shape
    ip=interpolate.RectBivariateSpline(np.arange(nx),np.arange(ny),f)
    [X,Y]=np.meshgrid(np.arange(factor*nx),np.arange(factor*ny), indexing='ij')
    X=X/factor
    Y=Y/factor
    return np.reshape(ip.ev(X.ravel(),Y.ravel()),(factor*nx,factor*ny))

############################## CORRECTIONS
def dx(im) :
    d=np.zeros(im.shape)
    d[:-1,:]=im[1:,:]-im[:-1,:]
    return d
def dxT(d) :
    im=np.zeros(d.shape)
    return im
def dy(im) :
    d=np.zeros(im.shape)
    return d
def dyT(d) :
    im=np.zeros(d.shape)
    return im 

class R() :
    def __init__(self,lam=10,mu=5) :
        self.lam=lam
        self.mu=mu
        self.nb_eval=0
        self.nb_grad=0
        self.nb_Hess=0
    def eval(self,u) :
        self.nb_eval+=1
        return 
    def grad(self,u) :
        self.nb_grad+=1
        return 
    def Hess(self,x) :
        assert False
        return 

class E() :
    def __init__(self,f,g) :
        self.f=f
        self.g=g
        self.nb_eval=0
        self.nb_grad=0
        self.nb_Hess=0
    def eval(self,u) :
        self.nb_eval+=1
        return 
    def grad(self,u) :
        self.nb_grad+=1
        return 
    def Hess(self,x) :
        assert False
        return 

class objectif() :
    def __init__(self,r,e) :
        self.r=r
        self.e=e
    def eval(self,u) :
        return 
    def grad(self,u) :
        return 

class MoindreCarres() :
    def __init__(self,e,r) :
        self.e=e
        self.r=r
        self.obj=objectif(e,r)
    def compute(self,u) :
        return 
    def JPsi(self,u,h) :
        return 
    def JPsiT(self,u,Psi) :
        return 
    def LM(self,u,h,epsilon=0.) :
        return 
        