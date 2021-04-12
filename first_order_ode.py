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

	def ff_dd(self, x):
		w = self.w; b = self.b; v = self.v;
		y, N = self.ff(x)
		return sum(v*w*w*y*(1-y)*(1-2*y))

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

	def Ndd_v(self, x):
		w = self.w; b = self.b; v = self.v;
		y, N = self.ff(x)
		return w*w*y*(1-y)*(1-2*y)

	def Ndd_w(self, x):
		w = self.w; b = self.b; v = self.v;
		y, N = self.ff(x)
		return v*w*y*(1-y)*(2-4*y+w*x*(6*y*y-6*y+1))

	def Ndd_b(self, x):
		w = self.w; b = self.b; v = self.v;
		y, N = self.ff(x)
		return v*w*w*y*(1-y)*(6*y*y-6*y+1)

	def E(self, x):
		y, N = self.ff(x); Nd = self.ff_d(x); Ndd = self.ff_dd(x)
		# return Ndd + 2*Nd/x+N**5
		return x*x*Ndd+4*x*Nd+2*N-6*x

	def dv(self, x):
		Nd_v = self.Nd_v; N_v = self.N_v; E = self.E; Ndd_v = self.Ndd_v
		y, N = self.ff(x)
		# return (Ndd_v(x)+2*Nd_v(x)/x+N_v(x)*5*N**4)
		return x*x*Ndd_v(x)+4*x*Nd_v(x)+2*N_v(x)

	def dw(self, x):
		Nd_w = self.Nd_w; N_w = self.N_w; E = self.E; Ndd_v = self.Ndd_v; ff = self.ff
		y, N = self.ff(x); Ndd_w = self.Ndd_w; Nd_w = self.Nd_w; N_w = self.N_w
		# return (Ndd_w(x)+2*Nd_w(x)/x+N_w(x)*5*N**4)
		# return Ndd_w(x)
		return x*x*Ndd_w(x)+4*x*Nd_w(x)+2*N_w(x)

	def db(self, x):
		Nd_b = self.Nd_b; N_b = self.N_b; E = self.E; Ndd_v = self.Ndd_v; ff = self.ff
		y, N = self.ff(x); Ndd_b = self.Ndd_b; Nd_b = self.Nd_b; N_b = self.N_b
		# return (Ndd_b(x)+2*Nd_b(x)/x+N_b(x)*5*N**4)
		# return Ndd_b(x)
		return x*x*Ndd_b(x)+4*x*Nd_b(x)+2*N_b(x)

	def update(self, x):
		lr = self.learning_rate; E = self.E(x)
		self.v += -lr*E*self.dv(x)
		self.w += -lr*E*self.dw(x)
		self.b += -lr*E*self.db(x)

	def sol(self, x):
		y, N = self.ff(x)
		# return x*N
		return 1-x+x*x*N

	def sol_der(self, x):
		y, N = self.ff(x)
		N_d = self.ff_d(x)
		N_dd = self.ff_dd(x)
		return N_dd*x*x+4*x*N_d+2*N

	def max_E(self):
		error = 0;
		for x in np.linspace(0,1,101):
			temp = (self.E(x)**2)/2
			if(temp>error):
				error = temp
		return error
