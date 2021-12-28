import numpy as np

class MLP() :
	def __init__(self,inp,outp,p=2,q=20,r=1) :
		self.inp=inp
		self.outp=outp
		self.p=p # taille de la première couche
		self.q=q # taille de la deuxième couche
		self.r=1 # taille de la troisième couche
		self.ind_s=(0,p*q,p*q+q,p*q+q+q*r) # indice de debut de stockage des variables
		self.ind_e=(p*q,p*q+q,p*q+q+q*r,p*q+q+q*r+r) # indice de fin de stockage des variables
		self.shapes=((q,p),(q,1),(r,q),(r,1)) #taille des variables
		self.nb_params=self.ind_e[-1] #taille totale du vecteur de paramètres
	def eval(self,theta) :
		(inp1,inp2,inp3,inp4)=self.forward(theta)
		return inp4
	def grad(self,theta) :
		state=self.forward(theta)
		gstate,gtheta=self.backward(theta,state)
		return gtheta

	def get_matrices(self,theta) :
		A,b,C,d=None # TODO
		return (A,b,C,d)
	def get_theta(self,matrices) :
		theta=np.zeros(self.nb_params)
		for (i_s,i_e,m,s) in zip(self.ind_s,self.ind_e,matrices,self.shapes) :
			assert m.shape==s
			theta[i_s:i_e]=m.ravel()
		return theta
	def product(self,A,b,x) :
		return A.dot(x)+np.outer(b,np.ones(x.shape[1]))
	def forward(self,theta) :
		return  
	def diff(self,theta,state,dtheta) :
		return  
	def backward(self,theta,state) :
		return 
