from __future__ import unicode_literals, print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
np.random.seed(123)
class EncoderVAE(nn.Module):
	def __init__(self, x_dim, h_dim, z_dim):
		super(EncoderVAE, self).__init__()
		
		self.linear_h = nn.Linear(x_dim, h_dim)
		self.linear_mu = nn.Linear(h_dim, z_dim)
		self.linear_var = nn.Linear(h_dim, z_dim)
		
	def forward(self, x):
		h, mu, logvar = self._encode(x)
		z = self._reparameterize(mu, logvar)
		return h, z, mu, logvar
		
	def _encode(self, x):
		h = F.relu(self.linear_h(x))
		mu = self.linear_mu(h)
		logvar = self.linear_var(h)
		return h, mu, logvar
		
	def _reparameterize(self, mu, logvar):
		eps = torch.from_numpy(np.random.normal(0,1, size=logvar.size())).float().cuda()
		z = mu + torch.sqrt(torch.exp(logvar)) * eps
		return z
		
class DecoderVAE(nn.Module):
	def __init__(self, z_dim, h_dim, x_dim):
		super(DecoderVAE, self).__init__()
		
		self.linear_h = nn.Linear(z_dim, h_dim)
		self.linear_x = nn.Linear(h_dim, x_dim)
		
	def forward(self, z):
		return self.decode(z)
		
	def decode(self, z):
		h = F.relu(self.linear_h(z))
		x = torch.sigmoid(self.linear_x(h))
		return x
		
	