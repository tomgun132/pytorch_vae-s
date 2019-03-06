from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vae import EncoderVAE, DecoderVAE
from torch.autograd import Variable

USE_CUDA = True
np.random.seed(123)
torch.manual_seed(123)
if USE_CUDA:
	torch.cuda.manual_seed(123)
class VAEs_A(nn.Module):
	
	def __init__(self, decoder, asp_num, z_dim, h_dim, out_dim):
		super(VAEs_A, self).__init__()
		self.decoder = decoder
		self.z_dim = z_dim
		
		self.sz = nn.Parameter(torch.FloatTensor(asp_num, z_dim))
		self.sz.data.uniform_(-0.1, 0.1)
		
		self.enc_linear = nn.Linear(h_dim, h_dim, bias=False) #W_ha * h_enc
		self.aspdec_linear = nn.Linear(h_dim, h_dim, bias=False) #U_ha * s_h
		self.v = nn.Linear(h_dim,1,bias=False) #v_ha * tanh
		
		self.context_linear = nn.Linear(h_dim, h_dim, bias=False)
		self.updatedec_linear = nn.Linear(h_dim,h_dim,bias=False)

		# self.wa = nn.Parameter(torch.rand(1))
		
		# Lipiji implementation use two different weight to update output vector
		# self.out_linear = nn.Linear(out_dim, out_dim, bias=False)
		self.wa1 = nn.Parameter(torch.rand(asp_num, 1))
		self.wa2 = nn.Parameter(torch.rand(asp_num, 1))
		
	def forward(self, input, enc_state):
		sh = F.relu(self.decoder.linear_h(self.sz))
		H_attn = self.getAttn(sh, enc_state, 'decode') # M x batch
		# print('attn: ', H_attn.size())
		# print('enc_state: ', enc_state.size())
		ch = H_attn.mm(enc_state) # M x hidden
		sh_ = torch.tanh(self.context_linear(ch) + self.updatedec_linear(sh))
		sx_ = torch.sigmoid(self.decoder.linear_x(sh_)) # M x V
		out_attn = self.getAttn(sx_, input, 'out')
		cx = out_attn.mm(input) # M x V
		# print('cx: ', cx)
		# print('sx: ', sx_)
		# sx = self.wa * cx + (1-self.wa) * sx_
		"""Lipiji implementation use tanh on top of linear function for last sx"""
		# a = torch.tanh(self.out_linear(sx_))
		sx = self.wa1 * cx + self.wa2 * sx_
		# print('sx: ', sx)
		# print(self.wa1)
		return sh_, sx, out_attn
		
	def getAttn(self, s, enc_state, type):
		input_size = enc_state.size(0)
		asp_num, h_dim = s.size()
		# e = Variable(torch.zeros(self.z_dim, input_size))
		# print(e_hij.size())
		# for i in range(self.z_dim):
			# score = self.score(enc_state, s[i], type)
			# print(score.size())
			# e_hij[i] = score
		if type == 'decode':
			e = Variable(torch.zeros(asp_num, input_size)).cuda()
			for i in range(asp_num):
				h_ = s[i].unsqueeze(0).expand(input_size, h_dim)
				M = torch.tanh(self.aspdec_linear(h_) + self.enc_linear(enc_state))
				e[i] = self.v(M).t()
				
			# s_feat = self.aspdec_linear(s)
			# s_feat_expanded = s.unsqueeze(1).expand(asp_num, input_size, h_dim).contiguous()
			# enc_feat = self.enc_linear(enc_state)
			# e = self.v(torch.tanh(enc_feat + s_feat_expanded))
			# print('e size', e.shape)
		elif type=='out':
			e = s.mm(enc_state.t())
		
		return F.softmax(e.squeeze(),dim=1) 
		
	# def score(self, enc, s, type='decode'):
		# if type=='decode':
			# enc_feat = self.enc_linear(enc)
			# dec_feat = self.aspdec_linear(s)
			# return self.v(enc_feat + dec_feat)
		# elif type=='out':
			# return s @ enc.t()
			
class Model(object):
	def __init__(self, x_dim, h_dim, z_dim, asp_num, model_file_path=None):
		encoder = EncoderVAE(x_dim, h_dim, z_dim)
		decoder = DecoderVAE(z_dim, h_dim, x_dim)
		sal_attn = VAEs_A(decoder, asp_num, z_dim, h_dim, x_dim)
		
		if USE_CUDA:
			encoder = encoder.cuda()
			decoder = decoder.cuda()
			sal_attn = sal_attn.cuda()
			
		self.encoder = encoder
		self.decoder = decoder
		self.salattn = sal_attn
		
		if model_file_path is not None:
			state = torch.load(model_file_path, map_location= lambda storage, location: storage)
			self.encoder.load_state_dict(state['encoder_state_dict'])
			self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
			self.salattn.load_state_dict(state['salience_attn_state_dict'])
	
if __name__ == '__main__':
	encoder_test = EncoderVAE(10,15,5)
	decoder_test = DecoderVAE(5,15,10)
	print(encoder_test)
	print(decoder_test)
	
	test_input = Variable(torch.FloatTensor([[0,0,1,0,1,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0]]))
	print(test_input.size())
	if USE_CUDA:
		encoder_test.cuda()
		test_input = test_input.cuda()
	
	h_enc, z, mu, logvar = encoder_test(test_input)
	# print(z)
	# print(mu)
	# print(logvar)
	if USE_CUDA:
		decoder_test.cuda()
	recons = decoder_test(z)
	# print(recons)
	print(h_enc.size())
	print(h_enc)
	model_test = VAEs_A(decoder_test,3,5,15,10)
	print(model_test)
	if USE_CUDA:
		model_test.cuda()
	sh, sx, attn = model_test(test_input, h_enc)
	print(sh)
	print(sx)
	print(attn)