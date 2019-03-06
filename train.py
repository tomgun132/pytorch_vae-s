from __future__ import unicode_literals, print_function, division

import os
import time

import torch
import data
from model import Model
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

# from data_util import *

# cudnn.benchmark = True
USE_CUDA = True


class Train(object):
	def __init__(self, base_path, topic):
		i2w, w2i, sents, lines, w_pos,  entities = data.load(base_path + "/docs/" + topic)
		if USE_CUDA:
			sents = sents.cuda()
		self.i2w = i2w
		self.w2i = w2i
		self.sents = sents
		self.lines = lines
		# self.w_concept = w_concept
		
		train_dir = 'log/'
		self.model_dir = os.path.join(train_dir,'model')
		
	def save_model(self, running_avg_loss, iter):
		state = {
			'iter' : iter,
			'encoder_state_dict' : self.model.encoder.state_dict(),
			'decoder_state_dict' : self.model.decoder.state_dict(),
			'salience_attn_state_dict' : self.model.salience_attn.state_dict(),
			'optimizer' : self.optimizer.state_dict(),
			'current_loss' : running_avg_loss
		}
		
		model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
		torch.save(state, model_save_path)
		
	def setup_train(self, lr, model_file_path=None):
		x_dim = len(self.w2i)
		y_dim = len(self.w2i)
		hidden_size = 500
		latent_size = 100
		num_summs = 5
		self.model = Model(x_dim, hidden_size, latent_size, num_summs, model_file_path)
		
		self.params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + list(self.model.salattn.parameters())
		initial_lr = 0.001
		self.optimizer = torch.optim.Adam(self.params, lr=lr) #edit later
		iter, loss = 0,0
		if model_file_path is not None:
			state = torch.load(model_file_path, map_location= lambda storage, location: storage)
			iter = state['iter']
			loss = state['current_loss']
			
			# load optimizer here
			# edit later
			
		return iter, loss
	
	def multivariate_bernoulli(self, y_pred, y_true):
		return torch.sum(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred), dim = 1)
		
	def kld(self, mu, logvar):
		return 0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1)
		
	def mse(self, pred, label):
		loss = torch.mean((pred - label) ** 2, dim=1)
		return torch.sum(loss)
		
	def train(self, lr, iter, model_file_path=None):
		iter_, loss = self.setup_train(lr, model_file_path)
		if iter_ > iter: iter=iter_
		for i in range(iter):
			self.optimizer.zero_grad()
			
			h, z, mu, logvar = self.model.encoder(self.sents)
			recons = self.model.decoder(z)
			# print(recons)
			sh, sx, ax = self.model.salattn(self.sents,h)
			# print(sh)
			# print(sx)
			# print(sx)
			# print(self.model.salattn.sz)
			recons_z = torch.mm(ax.t(), self.model.salattn.sz)
			recons_h = torch.mm(ax.t(), sh)
			recons_x = torch.mm(ax.t(), sx)
			loss_vae = -torch.mean(self.kld(mu,logvar) + self.multivariate_bernoulli(recons, self.sents))
			a = self.mse(recons_z, z)
			b = self.mse(recons_h, h)
			c = self.mse(recons_x, self.sents)
			loss_sal = a + 400 * b + 800 * c
			total_loss = loss_vae + loss_sal
			# print(a)
			# print(b)
			# print(c)
			total_loss.backward()
			clip_grad_value_(self.params, 10)
			self.optimizer.step()
			
			print('iteration: ',i)
			print('loss: ',total_loss)
			
		return sx.cpu().data.numpy(), ax.cpu().data.numpy()