#pylint: skip-file
cudaid = 0
import os
import time
import sys
import numpy as np
import pickle

from model import Model
import data
from train import Train

hidden_size = 500
latent_size = 100
num_summs = 10
lr = 0.001
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
optimizer = "adam" 

# base_path = "../data/TAC2011/"
base_path = "../data/DUC2006/"
# base_path = "../data/sensei_sent/"

top_w = 50
top_k = 10

class ScoreContainer(object):

	def __init__(self, sal, recons):
		self.salience_score = sal
		self.reconstruct_vector = recons
		
if __name__ == '__main__':
	print('Running on ', base_path)
	f = open(base_path + "topics_tiny.txt", "r") # or use topics_tiny.txt for development

	score_dict = dict()
	for topic in f:
		topic = topic.strip('\n')
		# print(topic, "============")
		# i2w, w2i, sents, lines, w_pos, w_concept,  entities = data.load(base_path + "/docs/" + topic)
		# dim_x = len(w2i)
		# dim_y = len(w2i)
		# num_sents = len(lines)
		# paragraph_info = np.zeros((num_sents, 1), dtype=theano.config.floatX)
		# for si in range(num_sents):
			# paragraph_info[si, 0] = 1#w_pos[si]
		# print("#features = ", dim_x, "#labels = ", dim_y)
		print("compiling...")
		# model = VAE(dim_x, dim_y, hidden_size, latent_size, num_sents, num_summs, optimizer)
		
		train_proc = Train(base_path, topic)
		num_sents = len(train_proc.lines)
		start = time.time()
		w_summs, Ax = train_proc.train(lr, 500)

		# for i in range(500):
			# error = 0.0
			# in_start = time.time()
			# cost, kld, lh, c, d, e,   w_summs,  Ax = model.train(sents, lr)
			# in_time = time.time() - in_start
			# print i, cost, c, d, e,  in_time

		o = open(base_path + "/vae/rouge/" + topic, "w") 
		Xx = np.linalg.norm(Ax, axis=0)
		ind = np.argpartition(Xx, -top_k)[-top_k:]
		for k in ind:
			print(train_proc.lines[k])
		print("==================")


		# Xk = train_proc.w_concept
		# ind = np.argpartition(Xk, -top_k)[-top_k:]
		# for k in ind:
			# print(train_proc.lines[k])
			# #o.write(lines[k])
		# print("=================")

		X = Xx
		np.savetxt(base_path + "/vae/salience/" + topic + ".atten", X)
		#X = np.linalg.norm(A , axis=0)
		ind = np.argpartition(X, -top_k)[-top_k:]
		for k in ind:
			print(train_proc.lines[k])
			o.write(train_proc.lines[k] + "\n")
		print("=================")
		#######################################

		# top sents
		o_sents = open(base_path + "/vae/salience/" + topic + ".sent", "w")
		for i in range(num_sents):
			o_sents.write(str(X[i]) + "\n")
			#print Xi[i] , Xj[i] , X1[i], X2[i], Xk[i]
		o_sents.close()
		print("=================")

		# top words
		o_words = open(base_path + "/vae/salience/" + topic + ".word", "w")
		top_w = 20
		Xm = np.sum(w_summs, axis=0)
		ind = np.argsort(-Xm)
		for k in range(top_w):
			print(train_proc.i2w[ind[k]])
		for k in range(len(train_proc.i2w)):
			o_words.write(train_proc.i2w[ind[k]] + " " + str(Xm[ind[k]]) + "\n")
		o_words.close()

		# top words for each top
		o_words = open(base_path + "/vae/salience/" + topic + ".tword", "w")
		for i in range(num_summs):
			s_i = w_summs[i,:]
			ind = np.argsort(-s_i)
			for k in range(top_w):
				print(train_proc.i2w[ind[k]] + ", ",)
			print("\n")
			ws = ""
			for k in range(len(train_proc.i2w)):
				ws += train_proc.i2w[ind[k]] + " "
			o_words.write(ws + "\n")
		o_words.close()
		
		o.close() # end of summary
		score = ScoreContainer(X, w_summs)
		score_dict[topic] = score
		
	with open('VAEsalience_scoreTAC2011.pkl', 'wb') as pkl_file:
		pickle.dump(score_dict, pkl_file, -1)
		
	f.close()
	print("Finished.")