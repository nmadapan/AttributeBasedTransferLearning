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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import roc_curve, auc, f1_score, make_scorer, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

## Custom modules
from utils import *
from SVMClassifier import SVMClassifier
from SVMRegressor import SVMRegressor

import warnings
warnings.filterwarnings('ignore')
# GridSearchCV prints warnings irrespective. 

class DAP(object):
	def __init__(self, data_dict, predicate_type = 'binary', rs = None, normalize = True):
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

		if(normalize):
			self.X_train, self.a_train, self.X_test, self.a_test = self.preprocess(self.X_train, \
										self.a_train, self.X_test, self.a_test, clamp_thresh = 3.0)
		
		self.clfs = []
		if(self.binary): 
			# skewedness = 80., n_components = 200, C = 300. # For cust data
			# skewedness = 75, n_components = 200, C = 10., rs = rs
			for _ in range(self.num_attr): self.clfs.append(SVMClassifier())
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

	def preprocess(self, seen_in, seen_out, unseen_in, unseen_out, clamp_thresh = 3.):
		'''
			Description:
				Does mean normalization and clamping. 
			Inputs:
				seen_in: np.ndarray (num_seen_samples x num_features)
				seen_out: np.ndarray (num_seen_samples x num_seen_classes)
				unseen_in: np.ndarray (num_unseen_samples x num_features)
				unseen_out: np.ndarray (num_unseen_samples x num_unseen_classes)
			Returns
				seen_in: np.ndarray (num_seen_samples x num_features)
				seen_out: np.ndarray (num_seen_samples x num_seen_classes)
				unseen_in: np.ndarray (num_unseen_samples x num_features)
				unseen_out: np.ndarray (num_unseen_samples x num_unseen_classes)
		'''
		seen_mean = np.mean(seen_in, axis = 0)
		seen_std = np.std(seen_in, axis = 0)
		
		## Mean normalization
		seen_in -= seen_mean
		seen_in /= seen_std
		## Clamping
		seen_in[seen_in > clamp_thresh] = clamp_thresh
		seen_in[seen_in < -1 * clamp_thresh] = -1 * clamp_thresh
		
		## Mean normalization
		unseen_in -= seen_mean
		unseen_in /= seen_std
		## Clamping
		unseen_in[unseen_in > clamp_thresh] = clamp_thresh
		unseen_in[unseen_in < -1 * clamp_thresh] = -1 * clamp_thresh

		return seen_in, seen_out, unseen_in, unseen_out


	def train(self, model, x_train, y_train, cv_parameters = None):
		# Binary classification with cross validation
		if(cv_parameters is None): 
			model.fit(x_train, y_train)
			return model
		else:
			clf = GridSearchCV(model, cv_parameters, cv = 5, n_jobs = -1, scoring = make_scorer(roc_auc_score))
			clf.fit(x_train, y_train)
			print(clf.best_params_)		
			return clf.best_estimator_


	def fit(self, cv_parameters = None):
		Xplat_train, Xplat_val, aplat_train, aplat_val = train_test_split(
			self.X_train, self.a_train, test_size=0.10, random_state = self.rs)
	
		a_pred = np.zeros(self.a_test.shape)
		a_proba = np.copy(a_pred)

		platt_params = []
		for idx in range(self.num_attr):
			print ('--------- Attribute %d/%d ---------' % (idx+1, self.num_attr))
			t0 = time()

			# Training and do hyper-parameter search
			self.clfs[idx].clf = self.train(self.clfs[idx].clf, Xplat_train, aplat_train[:,idx], cv_parameters)

			print ('Fitted classifier in: %fs' % (time() - t0))
			a_pred_train = self.clfs[idx].predict(Xplat_train)
			if(self.binary):
				self.clfs[idx].set_platt_params(Xplat_val, aplat_val[:,idx])
				f1_score_c0 = f1_score(aplat_train[:, idx], a_pred_train, pos_label = 0)
				f1_score_c1 = f1_score(aplat_train[:, idx], a_pred_train, pos_label = 1)
				print('Train F1 scores: %.02f, %.02f'%(f1_score_c0, f1_score_c1))			

			# Predicting
			a_pred[:,idx] = self.clfs[idx].predict(self.X_test)
			if(self.binary):
				a_proba[:,idx] = self.clfs[idx].predict_proba(self.X_test)
				f1_score_c0 = f1_score(self.a_test[:, idx], a_pred[:,idx], pos_label = 0)
				f1_score_c1 = f1_score(self.a_test[:, idx], a_pred[:,idx], pos_label = 1)
				print('Test F1 scores: %.02f, %.02f'%(f1_score_c0, f1_score_c1))
			
			print ('Saving files...')
			np.savetxt(join(self.write_dir, 'prediction_SVM'), a_pred)
			if(self.binary): np.savetxt(join(self.write_dir, 'probabilities_SVM'), a_proba)
		
		self.a_pred = a_pred
		self.a_proba = a_proba

		return a_pred, a_proba

	def evaluate(self, prob_fpath = None):
		if(prob_fpath is None):
			P = np.loadtxt(join(self.write_dir, 'probabilities_SVM')) # (n, 85)
			self.a_proba = P
		else:
			P = self.a_proba

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
	### To test on gestures ###
	# data_path = r'/home/isat-deep/Desktop/Naveen/fg2020/data/cust_feat_data/data_0.61305.mat'
	# classes = ['A', 'B', 'C', 'D', 'E']
	# data = reformat_dstruct(data_path)
	# normalize = True
	# parameters = {'fp__skewedness': [4., 10., 20.],
	# 			  'fp__n_components': [50],
	# 			  'svm__C': [1., 10.]}		
	###########################

	####### To test on awa #######
	# # This is to convert awa data to a compatible format.
	# data = awa_to_dstruct()
	# parameters = None	
	# normalize = False
	##############################

	p_type = 'binary'
	dap = DAP(data, predicate_type = p_type, normalize = normalize)
	dap.fit(parameters)

	attributepattern = 'DAP_' + p_type + '/probabilities_' + 'SVM'
	confusion, prob, L = dap.evaluate()
	
	wpath = join(dap.write_dir, 'AwA-ROC-confusion-DAP-'+p_type+'-SVM.pdf')
	plot_confusion(confusion, classes, wpath)

	wpath = join(dap.write_dir, 'AwA-ROC-DAP-SVM.pdf')
	plot_roc(prob, L, classes, wpath)

	wpath = join(dap.write_dir, 'AwA-AttAUC-DAP-SVM.pdf')
	plot_attAUC(dap.a_proba, dap.a_test, wpath)
	print ("Mean class accuracy %g" % np.mean(np.diag(confusion)*100))
