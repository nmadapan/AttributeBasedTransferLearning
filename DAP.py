'''
	This script has a class (DAP) that facilitates attribute prediction
	using Direct Attribute Prediction (DAP) approach proposed by 
	Lampert et al. in 2009 and 2014. 

	It is adapted from the GitHub repository of Charles. 
	https://github.com/chcorbi/AttributeBasedTransferLearning.git

	Author: Naveen Madapana
'''

import os, sys
from os.path import isdir, join, basename, dirname
from time import time
import random

# NumPy and plotting
import numpy as np
import matplotlib.pyplot as plt

## Scipy and sklearn
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import roc_curve, auc

## Custom modules
from utils import *
from SVMClassifier import SVMClassifier
from SVMRegressor import SVMRegressor

import warnings
warnings.filterwarnings('ignore')

class DAP(object):
	def __init__(self, data_dict, predicate_type = 'binary', rs = 42):
		'''
			data_dict:
				A dictionary with the following keys:
					1. seen_data_input: np.ndarray (train_num_instances x feature_size)
					2. unseen_data_input: np.ndarray (test_num_instances x feature_size)
					3. seen_data_output: np.ndarray (train_num_instances, )
					4. unseen_data_output : np.ndarray (test_num_instances, )
					5. seen_class_ids: np.ndarray (num_seen_classes, )
					6. unseen_class_ids: np.ndarray (num_unseen_classes, )
					7. seen_attr_mat: np.ndarray (num_seen_classes, num_attributes)
					8. unseen_attr_mat: np.ndarray (num_unseen_classes, num_attributes)				
			predicate_type:
				A string. For instance, it can be 'binary'. 
				The file, 'predicate-matrix-binary.txt' should exist. 
			rs:
				An integer indicating the random_state. This will be passed to the 
				functions that has randomness involved. If None, then, there is 
				no random state. 
		'''
		self.rs = rs
		self.predicate_type = predicate_type
		self.binary = (predicate_type == 'binary')

		## Write the files and results into this directory
		self.write_dir = './DAP_' + self.predicate_type
		try:
			if(not isdir(self.write_dir)): os.makedirs(self.write_dir)
		except Exception as exp: 
			print(exp)

		self.seen_data_input = data_dict['seen_data_input']
		self.seen_data_output = data_dict['seen_data_output']
		
		self.seen_attr_mat = data_dict['seen_attr_mat']
		self.unseen_attr_mat = data_dict['unseen_attr_mat']

		self.seen_class_ids = data_dict['seen_class_ids']
		self.unseen_class_ids = data_dict['unseen_class_ids']
		
		self.unseen_data_input = data_dict['unseen_data_input']
		self.unseen_data_output = data_dict['unseen_data_output']		

		self.num_attr = self.seen_attr_mat.shape[1]
		self.num_seen_classes = self.seen_attr_mat.shape[0]
		self.num_unseen_classes = self.unseen_attr_mat.shape[0]

		# Create training Dataset
		print ('Creating training dataset...')
		self.X_train, self.a_train = self.seen_data_input, self.seen_attr_mat[self.seen_data_output, :]

		print ('Creating test dataset...')
		self.X_test, self.a_test = self.unseen_data_input, self.unseen_attr_mat[self.unseen_data_output, :]
		
		self.clfs = []
		if(self.binary): 
			# skewedness = 80., n_components = 200, C = 300. # For cust data
			for _ in range(self.num_attr): self.clfs.append(SVMClassifier(skewedness = 80., n_components = 200, C = 10., rs = rs))
		else: 
			for _ in range(self.num_attr): self.clfs.append(SVMRegressor())

		self.pprint()

	def pprint(self):
		print('######################')
		print('### Seen Classes ###')
		print('No. of seen classes: ', self.num_seen_classes)
		print('Seen data input: ', self.seen_data_input.shape)
		print('Seen data output: ', self.seen_data_output.shape)
		print('Seen attribute matrix:', self.seen_attr_mat.shape)
		print('Seen class IDs:', self.seen_class_ids.shape)

		print('### Unseen Classes ###')
		print('No. of unseen classes: ', self.num_unseen_classes)
		print('Uneen data input: ', self.unseen_data_input.shape)
		print('Unseen data output: ', self.unseen_data_output.shape)
		print('Unseen attribute matrix:', self.unseen_attr_mat.shape)
		print('Unseen class IDs:', self.unseen_class_ids.shape)
		print('######################\n')

	def requirements(self):
		print('######################')
		print('data dictionary should contain 8 np.ndarray variables. ')
		print('###### Seen Classes ###')
		print('1. seen_data_input: (#train_samples x #features)')
		print('2. seen_data_output: (#train_samples, )')
		print('3. seen_attr_mat: (#seen_classes, #attributes)')
		print('4. seen_class_ids: (#seen_classes, )')

		print('\n###### Uneen Classes ###')
		print('5. unseen_data_input: (#test_samples x #features)')
		print('6. unseen_data_output: (#test_samples, )')
		print('7. unseen_attr_mat: (#unseen_classes, #attributes)')
		print('8. unseen_class_ids: (#unseen_classes, )')
		print('######################\n')

	def fit(self):
		Xplat_train, Xplat_val, yplat_train, yplat_val = train_test_split(
			self.X_train, self.a_train, test_size=0.10, random_state = self.rs)
	
		a_pred = np.zeros(self.a_test.shape)
		a_proba = np.copy(a_pred)

		platt_params = []
		for idx in range(self.num_attr):
			print ('--------- Attribute %d/%d ---------' % (idx+1, self.num_attr))
			t0 = time()

			if(self.binary):
				temp = self.a_train[:,idx]
				alph = temp.sum() / len(temp)
				# c_wt = {0: alph, 1: (1 - alph)}
				c_wt = {0: 0.5/(1-alph), 1: 0.5/alph} # 'balanced'
				# c_wt = {0: 5/(1-alph), 1: 5/alph} # 10*'balanced'

				# print(self.clfs[idx].clf.get_params().keys())
				self.clfs[idx].clf.set_params(svm__class_weight = c_wt)

			# Training
			self.clfs[idx].fit(self.X_train, self.a_train[:,idx])
			print ('Fitted classifier in: %fs' % (time() - t0))
			if(self.binary): self.clfs[idx].set_platt_params(Xplat_val, yplat_val[:,idx])

			# Predicting
			print ('Predicting for attribute %d...' % (idx+1))
			a_pred[:,idx] = self.clfs[idx].predict(self.X_test)
			if(self.binary): a_proba[:,idx] = self.clfs[idx].predict_proba(self.X_test)

			print ('Saving files...')
			np.savetxt(join(self.write_dir, 'prediction_SVM'), a_pred)
			if(self.binary):
				np.savetxt(join(self.write_dir, 'platt_params_SVM'), platt_params) ## REDUNDANT
				np.savetxt(join(self.write_dir, 'probabilities_SVM'), a_proba)
		
		self.a_pred = a_pred
		self.a_proba = a_proba

		return a_pred, a_proba

	def evaluate(self):
		# P = self.a_proba
		P = np.loadtxt(join(self.write_dir, 'probabilities_SVM')) # (n, 85)
		self.a_proba = P

		prior = np.mean(self.seen_attr_mat, axis=0)
		prior[prior==0.] = 0.5
		prior[prior==1.] = 0.5    # disallow degenerated priors

		M = self.unseen_attr_mat # (10,85)

		prob=[] # (n, 10)
		for p in P:
			prob.append( np.prod(M*p + (1-M)*(1-p),axis=1)/np.prod(M*prior+(1-M)*(1-prior), axis=1) )

		MCpred = np.argmax(prob, axis = 1) # (n, )

		d = self.num_unseen_classes
		confusion=np.zeros([d, d])
		for pl, gt in zip(MCpred, self.unseen_data_output):
			confusion[gt, pl] += 1.

		confusion /= confusion.sum(axis = 1, keepdims = True)
	    
		L = self.unseen_data_output

		return confusion, np.asarray(prob), L

if __name__ == '__main__':
	data_path = r'/home/isat-deep/Desktop/Naveen/fg2020/data/cust_feat_data/data_0.61305.mat'
	classes = ['A', 'B', 'C', 'D', 'E']
	data = reformat_dstruct(data_path)

	# classes = loadstr('testclasses.txt')
	# data = awa_to_dstruct()

	p_type = 'binary'
	dap = DAP(data, predicate_type = p_type)
	dap.pprint()

	dap.fit()

	attributepattern = 'DAP_' + p_type + '/probabilities_' + 'SVM'
	confusion, prob, L = dap.evaluate()
	
	wpath = join(dap.write_dir, 'AwA-ROC-confusion-DAP-'+p_type+'-SVM.pdf')
	plot_confusion(confusion, classes, wpath)

	wpath = join(dap.write_dir, 'AwA-ROC-DAP-SVM.pdf')
	plot_roc(prob, L, classes, wpath)

	wpath = join(dap.write_dir, 'AwA-AttAUC-DAP-SVM.pdf')
	plot_attAUC(dap.a_proba, dap.a_test, wpath)
	print ("Mean class accuracy %g" % np.mean(np.diag(confusion)*100))
