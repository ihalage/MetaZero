import numpy as np
import pandas as pd

# width = 8
# height = 8
# # print("{0:4}".format(''), end='')
# # for x in range(width):
# # 	print("{0:8}".format(x), end='')
# # print('\r\n')
# # for i in range(height - 1, -1, -1):
# # 	print("{0:4d}".format(i), end='')
# # 	for j in range(width):
# # 		# print("{0:8}".format(0), end='\r')
# # 		loc = i * width + j
# # 		# p = (1 if loc == 12 else 0)
# # 		# if p == 1:
# # 		#     print('π'.center(8), end='')
# # 		# elif p == player2:
# # 		#     print('O'.center(8), end='')
# # 		# else:
# # 		#     print('_'.center(8), end='')
# # 	print('\r\n\r\n')


# # for i in range(height):
# # 	for j in range(width):
# # 		print("{0:4d}".format(i), end='')

# metasurface = [['0' for i in range(8)] for j in range(8)]
# print (metasurface)


# print (pd.DataFrame(metasurface).iloc[::-1])
# # print (metasurface[3][5])
# metasurface[int(12/8)][int(12%8)] = '𝞹'
# metasurface[int(31/8)][int(31%8)] = '𝞹'
# metasurface[int(56/8)][int(56%8)] = '𝞹'
# # metasurface[4][5] = '8'
# # print (metasurface)
# print (pd.DataFrame(metasurface).iloc[::-1])
import copy
class A:
	def __init__(self,a1,a2):
		self.a1 = a1
		self.a2 = a2

	def pass_fna(self):
		self.a1=self.a2+2

class B:
	def __init__(self,a,b1,b2):
		self.a=a
		self.b1 = b1*3
		self.b2 = b2*2

	def pass_fnb(self):
		self.b1=self.b2

class C:
	def __init__(self,b,c1):
		self.b=b
		self.c1 = c1*3

	def pass_fnc(self):
		# print ('C')
		self.b.pass_fnb()
		return 0

ins_A = A(0,0)
ins_B= B(ins_A,0,0)
ins_C = C(ins_B,0)
print (ins_C)
cpy = copy.deepcopy(ins_C.pass_fnc())