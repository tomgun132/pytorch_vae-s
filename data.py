#/usr/bin/env python
# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os
import numpy as np
import encodings
import pkgutil
import torch
from nltk.corpus import stopwords

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
# for paragraph position
par_const = 0.5
par_bar = 4
forgotten_punct= ['-lrb-', '-rrb-']
def all_encodings():
	modnames = set(
		[modname for importer, modname, ispkg in pkgutil.walk_packages(
			path=[os.path.dirname(encodings.__file__)], prefix='')])
	aliases = set(encodings.aliases.aliases.values())
	return modnames.union(aliases)
	
def load_stop_words():
    return stopwords.words('english')

def load(f_path):
	dic = {}
	i2w = {}
	w2i = {}
	lines = []
	w_pos = []
	# w_concept = []

	# load dic
	encodings = all_encodings()
	# for enc in encodings:
		# try:
			# with open(filename, encoding=enc) as f:
				# # print the encoding and the first 500 characters
				# print(enc, f.read(500))
		# except Exception:
			# pass
	chosen_encodings = []
	for enc in encodings:
		try:
			f = open(f_path + ".dic", "r", encoding=enc)
			for line in f:
				line
			chosen_encodings.append(enc)
		except Exception:
			pass
			
	
	f = open(f_path + ".dic", "r", encoding=chosen_encodings[0])
	for line in f:
		line = line.strip('\n')
		fields = line.split("=====")
		if len(fields) < 2:
			print("Error.")
			continue
		dic[fields[0]] = float(fields[1])
		i2w[len(w2i)] = fields[0]
		w2i[fields[0]] = len(w2i)
	f.close()

	# load sents
	f = open(f_path + ".sent", "r", encoding=chosen_encodings[0])
	i = 0
	sent = 0
	for line in f:
		line = line.strip('\n')
		i += 1
		if '***' in line:
			# sent += 1
			# if sent > 10: break
			continue
		lines.append(line)
	f.close()

	# num_x = len(lines)

	# load bows
	# sents = np.zeros((num_x, len(w2i)), dtype = theano.config.floatX) 
	sents = list()
	f = open(f_path + ".tf", "r", encoding=chosen_encodings[0])
	i = 0
	for line in f:
		sent = torch.zeros(len(w2i), dtype=torch.float)
		line = line.strip('\n')
		tfs = line.split("|")
		i += 1
		if '***' in line:
			continue
		for tf in tfs:
			fields = tf.split("=====")
			if len(fields) < 2:
				#print "Error.tf"
				continue
			if fields[0] in forgotten_punct: continue
			sent[w2i[fields[0]]] = float(fields[1])
		sents.append(sent)

	num_x = len(sents)
	sents = torch.stack(sents)

	f.close()
	#normalization
	for i in range(num_x):
		norm2 = torch.norm(sents[i, :])
		if norm2 == 0:
			norm2 = 1
		sents[i, :] = sents[i, :] / norm2

	# load entities
	# entities = np.zeros((1, len(w2i)), dtype = theano.config.floatX)
	entities = torch.zeros((1, len(w2i)), dtype = torch.float)
	f = open(f_path + ".entities", "r", encoding=chosen_encodings[0])
	for line in f:
		line = line.strip('\n')
		fields = line.split("=====")
		if len(fields) < 2:
			print("Error.entities")
			continue
		entities[0, w2i[fields[0]]] = float(fields[1])
	f.close()
	entities = entities / torch.norm(entities)

	# load weights
	# f = open(f_path + ".weight", "r")
	# for line in f:
		# line = line.strip('\n')
		# fields = line.split()
		# if len(fields) < 2:
			# print("Error.weight")
			# continue
		# w_pos.append(float(fields[0]))
		# w_concept.append(float(fields[1]))
	# f.close()

	return i2w, w2i, sents, lines, w_pos, entities


def word_sequence(f_path, batch_size = 1):
	stop_words = load_stop_words()
	seqs = []
	i2w = {}
	w2i = {}
	lines = []
	tf = {}
	para_info = []

	tmp_sents = []
	last_sents = []

	f = open(curr_path + "/" + f_path, "r")
	paragraph_now = 0
	for line in f:
		line = line.strip('\n').lower()
		words = line.split()

		if line == "====":
			paragraph_now = 0
			continue
		if len(words) < 3:
			if len(words) == 0 and paragraph_now < par_bar:
				paragraph_now += 1
			continue
		words.append("<eoss>") # end symbol
		words.append("<eoss>") 
		tmp_sents.append(line)
		lines.append(words)
		para_info.append(paragraph_now)
		for w in words:
			if w not in w2i:
				i2w[len(w2i)] = w
				w2i[w] = len(w2i)
				tf[w] = 1
			else:
				tf[w] += 1
	f.close()

	num_x = len(lines) 
	#num_x = 30
	for i in xrange(len(para_info)):
		para_info[i] = par_const ** para_info[i]
	# represent sentences with word-bag model
	# sents = np.zeros((num_x, len(w2i)), dtype = theano.config.floatX) # row: each sentence
	sents = torch.zeros((num_x, len(w2i)), dtype=torch.float)
	for i in range(0, num_x):
		last_sents.append(tmp_sents[i])

		each_sent = lines[i]
		# x = np.zeros((len(each_sent), len(w2i)), dtype = theano.config.floatX) # row: each word position
		x = torch.zeros((len(each_sent), len(w2i)), dtype = torch.float) # row: each word position
		for j in range(0, len(each_sent)):
			each_word = each_sent[j]
			x[j, w2i[each_word]] = 1
			if each_word not in stop_words:
				sents[i, w2i[each_word]] += 1

		seqs.append(x)
	seqs = torch.stack(seqs)

	#normalization
	for i in xrange(num_x):
		norm2 = torch.norm(sents[i, :])
		if norm2 == 0:
			norm2 = 1
		sents[i, :] = sents[i, :] / norm2


	data_xy = batch_sequences(seqs, i2w, w2i, para_info, batch_size)
	print("#dic = " + str(len(w2i)))
	return seqs, i2w, w2i, data_xy, sents, last_sents

def batch_sequences(seqs, i2w, w2i, paragraph_info, batch_size):
	data_xy = {}
	batch_x = []
	batch_y = []
	batch_paragraph_info = []
	seqs_len = []
	batch_id = 0
	dim = len(w2i)
	# zeros_m = np.zeros((1, dim), dtype = theano.config.floatX)
	zeros_m = torch.zeros((1, dim), dtype = torch.float)
	for i in xrange(len(seqs)):
		seq = seqs[i]
		X = seq[0 : len(seq) - 1, ]
		Y = seq[1 : len(seq), ]
		batch_x.append(X)
		seqs_len.append(X.shape[0])
		batch_y.append(Y)
		batch_paragraph_info.append(paragraph_info[i])

		if len(batch_x) == batch_size or (i == len(seqs) - 1):
			# tmp_sum = float(sum(batch_paragraph_info))
			# batch_paragraph_info = [(lambda x : x / tmp_sum)(x) for x in batch_paragraph_info]

			max_len = np.max(seqs_len)
			# mask = np.zeros((max_len, len(batch_x)), dtype = theano.config.floatX)
			mask = torch.zeros((max_len, len(batch_x)), dtype = torch.float)

			# concat_X = np.zeros((max_len, len(batch_x) * dim), dtype = theano.config.floatX)
			concat_X = torch.zeros((max_len, len(batch_x) * dim), dtype = torch.float)
			concat_Y = concat_X.copy()
			for b_i in xrange(len(batch_x)):
				X = batch_x[b_i]
				Y = batch_y[b_i]
				mask[0 : X.shape[0], b_i] = 1
				for r in xrange(max_len - X.shape[0]):
					X = np.concatenate((X, zeros_m), axis=0)
					Y = np.concatenate((Y, zeros_m), axis=0)
				concat_X[:, b_i * dim : (b_i + 1) * dim] = X 
				concat_Y[:, b_i * dim : (b_i + 1) * dim] = Y
			data_xy[batch_id] = (concat_X, concat_Y, mask, len(batch_x), np.array(batch_paragraph_info))
			batch_x = []
			batch_y = []
			batch_paragraph_info = []
			seqs_len = []
			batch_id += 1
	return data_xy

if __name__ == '__main__':
	base_path = "../data/TAC2011/Amish Shooting"
	i2w, w2i, sents, lines, w_pos, w_concept, entities = load(base_path)