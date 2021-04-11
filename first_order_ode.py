class neural_network:
	def __init__(self, hiden_neurons, learning_rate):
		self.w = np.random.rand(hiden_neurons)
		self.b = np.random.rand(hiden_neurons)
		self.v = np.random.rand(hiden_neurons)
		self.hiden_neurons = hiden_neurons
		self.learning_rate = learning_rate

	def ff(self, x):
		w = self.w; b = self.b; v = self.v;
		y = 1/(1+np.exp(-w*x-b))
		N = sum(y*v)
		return [y, N]

	def ff_d(self, x):
		w = self.w; b = self.b; v = self.v;
		y, N = self.ff(x)
		return sum(y*(1-y)*w*v)

	def N_v(self, x):
		y, N = self.ff(x)
		return y

	def N_w(self, x):
		y, N = self.ff(x)
		return self.v*y*(1-y)*x

	def N_b(self, x):
		y, N = self.ff(x)
		return self.v*y*(1-y)

	def Nd_v(self, x):
		y, N = self.ff(x)
		return self.w*y*(1-y)

	def Nd_w(self, x):
		y, N = self.ff(x)
		return x*self.w*y*(1-y)*(1-2*y)

	def Nd_b(self, x):
		y, N = self.ff(x)
		return self.v*self.w*y*(1-y)*(1-2*y)

	def E(self, x):
		# depends on ODE
		
	def dv(self, x):
		# depends on ODE

	def dw(self, x):
		# depends on ODE
		
	def db(self, x):
		# depends on ODE

	def update(self, x):
		lr = self.learning_rate; 
		self.v += -lr*self.dv(x)
		self.w += -lr*self.dw(x)
		self.b += -lr*self.db(x)

	def sol(self, x):
		y, N = self.ff(x)
		return x*N

	def sol_der(self, x):
		y, N = self.ff(x)
		N_d = self.ff_d(x)
		return x*N_d+N

	def max_E(self):
		error = 0;
		for x in np.linspace(0,1,101):
			temp = (self.E(x)**2)/2
			if(temp>error):
				error = temp
		return error
