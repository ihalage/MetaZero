''' 
This python script calculates the RCS reduction of a metasurface using both matlab and numpy
Author: Achintha Ihalage
Date:	06/04/2020
'''

# import matlab.engine
import numpy as np
import copy

# state = np.random.randint(4, size=(6,6)).astype(float) 
# state = np.ones((8,8))
# state = np.random.randint(2,size=(8,8)).astype(float)*3
# def build_chessboard(w, h) :
#       re = np.r_[ w*[0,1] ]              # even-numbered rows
#       ro = np.r_[ w*[1,0] ]              # odd-numbered rows
#       return np.row_stack(h*(re, ro))
# state = build_chessboard(8,8)*3.0

# metasurface proposed in the paper
# state = np.array([[1, 0, 1, 0, 1, 1, 0, 0], 
# 					[0, 0, 1, 0, 0, 0, 1, 0], 
# 					[1, 0, 0, 0, 1, 0, 1, 0], 
# 					[1, 0, 0, 1, 0, 0, 1, 1], 
# 					[0, 0, 1, 1, 1, 1, 1, 1], 
# 					[0, 0, 0, 0, 1, 0, 0, 0], 
# 					[1, 1, 0, 1, 0, 1, 0, 1], 
# 					[0, 1, 1, 0, 0, 1, 0, 0]])
# state = state*3.0
class RCS():

	def __init__(self, metasurface_width, metasurface_height):
		###################### all matlab.engine calls had to be commented out due to basic_string::_M_construct null not valid ERROR ####################
		# self.eng = matlab.engine.start_matlab()
		f = 10e9	# 10 GHz frequency
		# self.m = matlab.double([metasurface_width])
		# self.n = matlab.double([metasurface_height])
		c = 3e+8
		# self.lmd = matlab.double([c/f])
		# self.d = self.lmd 	# d=lmd
		# now define the same parameters to be used with numpy
		self.m_np = metasurface_width
		self.n_np = metasurface_height
		self.lmd_np = c/f
		self.d_np = self.lmd_np
		self.k_np = 2*np.pi/self.lmd_np # wave vector

		# self.metasurface_np = state 	# for numpy
		# self.metasurface = matlab.double(state.tolist())	# for matlab

	# def get_RCS_matlab(self, state):
	# 	metasurface_matlab = matlab.double(state.tolist())	# for matlab
	# 	try:
	# 		rcs = self.eng.rcs_calc(metasurface_matlab,self.m,self.n,self.lmd,self.d,nargout=1)
	# 		return rcs
	# 	except Exception as e:
	# 		raise e

	def get_RCS_numpy(self, state):
		Ntheta=180.0
		Nphi=180.0
		dtheta=np.pi/(2*Ntheta)
		dphi=2*np.pi/Nphi
		theta=np.linspace(0,np.pi/2,Ntheta)
		phi=np.linspace(0,2*np.pi,Nphi)
		[THETA, PHI]=np.meshgrid(theta,phi)


		# ######## get the metasurface based on the state returned by python code
		# # 1 ---> 0
		# # 2 ---> pi/4
		# # 3 ---> pi/2
		# # 4 ---> pi
		reflect_phi = copy.deepcopy(state)
		reflect_phi[(reflect_phi!=1) & (reflect_phi!=2) & (reflect_phi!=3)]=0	# assign 0 response to all other places
		reflect_phi[reflect_phi==1]=np.pi/4
		reflect_phi[reflect_phi==2]=np.pi/2
		# reflect_phi[reflect_phi==3]=3*np.pi/4
		reflect_phi[reflect_phi==3]=np.pi

		F=0
		for q in range(self.n_np):
			for p in range(self.m_np):
				F2 = np.exp((-1j)*(reflect_phi[q,p]+np.multiply(self.k_np*self.d_np*np.sin(THETA),((q-0.5)*np.cos(PHI)+(p-0.5)*np.sin(PHI)))))
				F=F+F2

		F3=np.multiply(np.abs(np.multiply(F,F)),np.sin(THETA))
		F4=np.multiply(F,F)

		Ravg=np.sum(np.sum(F3*dtheta)*dphi)
		F5=F4/Ravg

		# maximum directivity
		Rm=np.max(np.abs(F5))
		# RCS reduction
		rcs = (self.lmd_np**2)*Rm/(4*np.pi*self.m_np*self.n_np*self.d_np**2)
		return rcs

# rcs = RCS(8,8)
# # rcs_matlab = rcs.get_RCS_matlab(state)
# rcs_numpy = rcs.get_RCS_numpy(state)

# # print (rcs_matlab)
# print (rcs_numpy)
# print (state)
