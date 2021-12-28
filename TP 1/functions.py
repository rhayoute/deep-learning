import numpy as np

class square() :
	def __init__(self) :
		print("Fonction (x,y) --> x^2/2+7/2*y^2")
		self.size=2
	def eval(self,x) :
		if not len(x)==self.size :
			print ("Erreur de taille de x, on a ",len(x)," au lieu de ",self.size)
		return 0.5*x[0]**2+7/2.*x[1]**2
	def grad(self,x) :
		print('toto')
		return np.array([x[0],7*x[1]])
	def Hess(self,x) :
		to_return=np.zeros((2,2))
		to_return[0,0]=1
		to_return[1,1]=7
		return to_return


