'''
	This script has a class (IAP) that facilitates attribute prediction
	using Indirect Attribute Prediction (IAP) approach proposed by 
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
from sklearn.metrics import roc_curve, auc, f1_score, make_scorer, roc_auc_score, accuracy_score
from sklearn.exceptions import ConvergenceWarning

## Custom modules
from utils import *
from SVMClassifier import SVMClassifierIAP
from SVMRegressor import SVMRegressorIAP

import warnings
warnings.filterwarnings('ignore')
# GridSearchCV prints warnings irrespective. 

class IAP(object):
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
				If it is binary, we will use a classifier, otherwise, we will use a regressor. 
			rs:
				An integer indicating the random_state. This will be passed to the 
				functions that has randomness involved. If None, then, there is 
				no random state. 
		'''
		self.rs = rs
		self.predicate_type = predicate_type
		self.binary = (predicate_type == 'binary')

		## Write the files and results into this directory
		self.write_dir = './IAP_' + self.predicate_type
		try:
			if(not isdir(self.write_dir)): os.makedirs(self.write_dir)
		except Exception as exp: 
			print(exp)

		## Seen Unseen data
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

		## Create training Dataset
		print ('Creating training dataset...')
		self.X_train = self.seen_data_input
		self.y_train = self.seen_data_output

		## Create testing Dataset
		print ('Creating test dataset...')
		self.X_test = self.unseen_data_input
		self.y_test = self.unseen_data_output

		if(normalize):
			self.X_train, _, self.X_test, _ = self.preprocess(\
				self.X_train, None, self.X_test, None, clamp_thresh = 3.0)
		
		# Irrespective of binary true or false, we should do classification first. 
		self.clf = SVMClassifierIAP()

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
			## When n_jobs = None, it is taking slightly more time to run than n_jobs = -1 (Equivalent to #processors)
			# However, when n_jobs is -1, it prints a lot of ConvergenceWarning by sklearn. 
			clf = GridSearchCV(model, cv_parameters, cv = 5, n_jobs = None, \
								scoring = make_scorer(accuracy_score)) ## TODO: CHeck it. which score. 
			clf.fit(x_train, y_train)
			print(clf.best_params_)		
			return clf.best_estimator_

	def fit(self, cv_parameters = None):
		y_pred = np.zeros(self.y_test.shape)
		y_proba = np.zeros((y_pred.shape[0], self.num_seen_classes))

		print('Training model... (takes around 10 min)')
		t0 = time()
		self.clf.clf = self.train(self.clf.clf, self.X_train, self.y_train, cv_parameters)
		# self.clf.fit(self.X_train, self.y_train)
		print('Training finished in %.02f secs'%(time() - t0))

		## Train evaluation
		y_pred_train = self.clf.predict(self.X_train)
		y_proba_train = self.clf.predict_proba(self.X_train)		
		acc = accuracy_score(self.y_train, y_pred_train)
		print('Train Accuracy: %.02f'%acc)

		## Testing evaluation
		y_pred = self.clf.predict(self.X_test)
		y_proba = self.clf.predict_proba(self.X_test)		
		acc = accuracy_score(self.y_test, y_pred)
		print('Test Accuracy: %.02f'%acc)

		print ('Saving files...')
		np.savetxt(join(self.write_dir, 'prediction_SVM'), y_pred)
		np.savetxt('./IAP/prediction_SVM', y_pred)
		np.savetxt('./IAP/probabilities_SVM', y_proba)
		
		self.y_pred = y_pred
		self.y_proba = y_proba

		return y_pred, y_proba

	def evaluate(self, prob_fpath = None):
		M = self.unseen_attr_mat # (10,85)
		prob=[] # (n, 10)

		if(prob_fpath is None): P = self.y_proba # (n, 85)
		else: P = np.loadtxt(join(self.write_dir, 'probabilities_SVM')) # (n, 85)
		P = np.dot(P, self.seen_attr_mat)

		if(self.binary):
			prior = np.mean(self.seen_attr_mat, axis=0)
			prior[prior==0.] = 0.5
			prior[prior==1.] = 0.5    # disallow degenerated priors
			for p in P:
				prob.append(np.prod(M*p + (1-M)*(1-p),axis=1)/\
							np.prod(M*prior+(1-M)*(1-prior), axis=1) )			
		else:
			Md = np.copy(M).astype(np.float)
			Md /= np.linalg.norm(Md, axis = 1, keepdims = True)
			for p in P:
				p /= np.linalg.norm(p)
				prob.append(np.dot(Md, p))

		MCpred = np.argmax(prob, axis = 1) # (n, )

		d = self.num_unseen_classes
		confusion=np.zeros([d, d])
		for pl, gt in zip(MCpred, self.unseen_data_output):
			confusion[gt, pl] += 1.

		confusion /= confusion.sum(axis = 1, keepdims = True)

		return confusion, np.asarray(prob), self.unseen_data_output

if __name__ == '__main__':
	### To test on gestures ###
	data_path = r'/home/isat-deep/Desktop/Naveen/fg2020/data/cust_feat_data/data_0.61305.mat'
	classes = ['A', 'B', 'C', 'D', 'E']
	data = reformat_dstruct(data_path)
	normalize = True
	parameters = {'fp__skewedness': [4.], # 4., 6., 10.
				  'fp__n_components': [50],
				  'svm__C': [10.]} # 1., 10.
	p_type = 'binary2'
	print('Gesture Data ... ', p_type)
	###########################

	####### To test on awa #######
	# # This is to convert awa data to a compatible format.
	# print('AwA data ...')
	# classes = loadstr('testclasses.txt')
	# data = awa_to_dstruct()
	# parameters = None	
	# normalize = False
	# p_type = 'binary'
	##############################

	iap = IAP(data, predicate_type = p_type, normalize = normalize)
	start = time()
	iap.fit(parameters)
	print('Total time taken: %.02f secs'%(time()-start))

	attributepattern = 'IAP_' + p_type + '/probabilities_' + 'SVM'
	confusion, prob, L = iap.evaluate()
	
	wpath = join(iap.write_dir, 'AwA-ROC-confusion-IAP-'+p_type+'-SVM.pdf')
	plot_confusion(confusion, classes, wpath)
	print ("Mean class accuracy %g" % np.mean(np.diag(confusion)*100))

	# wpath = join(iap.write_dir, 'AwA-ROC-IAP-SVM.pdf')
	# plot_roc(prob, L, classes, wpath)

	## TODO: Why are these two not working. 
	# wpath = join(iap.write_dir, 'AwA-AttAUC-IAP-SVM.pdf')
	# plot_attAUC(iap.y_proba, iap.y_test, wpath)
