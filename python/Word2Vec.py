from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import pickle
import os
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class Word2Vec:
	def __init__(self, args):

		# hyperparameters
		self.learning_rate = 1e-1
		self.num_steps = 1e5
		self.embeddings = None
		self.dictionary = {}
		self.reverse_dictionary = {}

	@property
	def embeddings(self):
		return self.embeddings

	@embeddings.setter
	def embeddings(self, val):
		assert type(val.__module__) == np.__name__
		# define some error checking
		self.embeddings = val  	


	def read_corpus(self):
		try:
			with open(self.filepath, 'r') as fp:
				self.text = fp.read()
				print('Successfully read file')
		except IOError as e:
			print(e)
		finally:
			print('The length of the corpus was: ', len(self.text))
	
	@property
	def text(self):
		return self.text

	@text.deleter
	def text(self):
		print('Deleting text attribute')
		self.text = ''	# or None	


	def load(self, pk):
		# generator
		assert(os.path.exists(pk), 'File {} does not exist'.format(pk))
		with open(pk, 'r') as pk_handle:
			while True:
				try:
					yield pickle.load(pk_handle)
				except EOFError:
					break
				
	def save(self, data, filepath):
		if not os.path.exists(filepath):	
			with open(filepath, 'w') as fout:
				pickle.dump(data, fout)
		else:
			print('File already exists!')
			raise FileExistsError('File already exists! Please try again.') # Python3 only
	
	def embed(self, loop_function=Word2Vec.loop):
		with tf.Graph().as_default():
			
			# placeholders for word and context, fed in batches to tf.nce_loss
			inputs = tf.placeholder(tf.int32, [self.batch_size])
			labels = tf.placeholder(tf.int32, [self.batch_size, 1])
			
			with tf.device('/cpu:0'):
				embeddings = tf.get_variable('embeddings', tf.random_normal_initializer())
				_embed = tf.nn.embedding_lookup(embeddings, inputs)
				softmax_w = tf.get_variable('softmax_w', [], tf.random_normal_initializer()))
				softmax_b = tf.get_variable('softmax_b', [], tf.constant_initializer(0.0))
				nce_loss = tf.nn.nce_loss()
				loss = tf.reduce_mean(nce_loss)
				optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
		
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			for step in range(self.num_steps):
				batch_inputs = loop()
				feed_dict = {}
				_, loss_val = sess.run([], feed_dict=feed_dict)
				if step % 1000 == 0 and step > 0:
					print('Average loss at step {:d}: {:f}'.format(step, loss_val/2000))
			self.embeddings = pass
			
	# implement deleters to save memory
	@property.deleter
	pass

	@staticmethod
	def loop(data, ngrams, nsamples, batch_size):
		offset = 2 * (ngrams - 1)
		x = np.ndarray([batch_size], np.int32)
		y = np.ndarray([batch_size, 1], np.int32)
		
		for i in range(batch_size // nsamples):
			try:
				sublist = data[index: index+offset]
				samples = random.sample(range(offset), nsamples)
				for j in range(nsamples):
					x[i*nsamples + j] = sublist[samples[j]][1]
					y[i*nsamples + j, 0] = sublist[samples[j]][0]
				loop.index += offset
			except IndexError:
				loop.index = 0
				continue
		yield x,y 
				
				



	@staticmethod
	def tokenize_sentences(data):
		# takes too long, only use if handling abbreviations 
		return nltk.tokenize.sent_tokenize(data)

	def tokenize(self, remove_stopwords=True, remove_punct=True):
		tokens = nltk.tokenize.word_tokenize(self.text)
		if not remove_stopwords and not remove_punct:
			self.tokens = tokens
			return tokens

		filter_tokens = []
		if remove_stopwords:
			filter_tokens.extend(nltk.corpus.stopwords('english'))
		if remove_punct:
			filter_tokens.extend(string.punct)
		# will return lowercase
		tokens = [i.lower() for i in tokens if i not in set(filter_tokens)]
		self.tokens = tokens
		return tokens
		
	def abbreviation_handler(self):
		stopwords = set(nltk.corpus.stopwords.words('english'))
		sentences = Word2Vec.tokenize_sentences(self.text)
		abbreviations = {}
		query1 = re.compile(r'\((\w+)\)')
		for sentence in sentences:
			matches = re.findall(query1, sentence)
			if matches is not None:
			 	for match in matches:
				# intersperse '|' (regex OR) in reducing substrings
					match = # remove numbers
					tmp1 = [match[i:] for i, _ in enumerate(match)] # same as tmp1 = [match[i:] for i in range(len(match))]
					tmp2 = ['|'.join(i) for i in tmp1]		# same as lambda elem, arr = sum([[elem, ind] for ind in arr], [])
					tmp3 = [ i[0] + '(?:\w+|\b' + i[1:] + ')' for i in tmp2] 	
					query2 = r'\b' + '?(?:.)'.join(tmp3) 
					possible_matches = re.findall(query2, sentence, re.IGNORECASE) 
					if not possible_matches:
						trimmed_str = ' '.join([i for i in word_tokenize(sentence) if i not in stopwords])	
						possible_matches = re.findall(query2, trimmed_str, re.IGNORECASE)
		
	def visualize_embeddings(self, num_plotted=500):
		tsne = TSNE(perplexity=30, n_components=2, init='pca')
		low_dim_embeddings = tsne.fit_transform(self.embeddings[:num_plotted, :]
		labels = [self.reverse_vocabulary[i] for i in xrange(num_plotted)]
		plt.figure(figsize=[18, 18])
		for i, label in enumerate(self.dictionary):
			x, y = low_dim_embeddings[i, :]
			plt.scatter(x, y)
			plt.annotate(labels, xy=[x, y], xytex[5, 2], textcoords='offset points')	
		plt.show()
	
