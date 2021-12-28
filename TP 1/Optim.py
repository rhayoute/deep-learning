import numpy as np

def deriv_num(J,a,d,compute_grad=True,compute_Hess=True) :
	eps_range=[0.1**(i+1) for i in range(12)]
	for eps in  eps_range:
		s='eps {:1.3e}'.format(eps)
		print(s)
