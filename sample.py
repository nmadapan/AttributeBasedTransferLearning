from scipy.io import savemat, loadmat
import numpy as np
import sys
from os.path import join
from sklearn.kernel_approximation import SkewedChi2Sampler
import time

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

from sklearn.metrics.pairwise import linear_kernel

from SVMClassifier import SVMClassifier

from sklearn.exceptions import ConvergenceWarning

import pickle

a = np.random.randint(0, 10, (3, 4))
print(a)
for p in a:
	p = p / 2
	print(p)
print(a)
# from DAP import DAP
# from utils import *

# with open('./DAP_binary/p_data_0.61305.pickle', 'rb') as fp:
# 	obj = pickle.load(fp)
# dap = obj['self']

# print_dict(dap.__dict__)

# digits = load_digits()
# X, y = digits.data,digits.target
# print(X.shape)

# start = time.time()

# multi_core_learning = cross_validate(SVC(), X, y, cv=150, n_jobs=1)

# print('Time taken: %.02f secs'%(time.time()-start))

# seed = 23

# sk = SkewedChi2Sampler(skewedness=0.1,	n_components=1) # , random_state = seed

# # np.random.seed(seed)
# # X = np.random.rand(6, 3)
# X = np.array(
# 	[[0.51729788, 0.9469626,  0.76545976],
#  	[0.28239584, 0.22104536, 0.68622209],
#  	[0.1671392,  0.39244247, 0.61805235],
#  	[0.41193009, 0.00246488, 0.88403218],
#  	[0.88494754, 0.30040969, 0.58958187],
#  	[0.97842692, 0.84509382, 0.06507544]]
# 	)

# Xp = sk.fit_transform(X)

# print(X)
# print(Xp)

# data_path = r'/home/isat-deep/Desktop/Naveen/fg2020/data/raw_feat_data/data_0.11935.mat'
# x = loadmat(data_path, struct_as_record = False, squeeze_me = True)['dstruct']

# imp_keys = ['unseen_class_ids', 'seen_class_ids', 'seen_data_input', 'unseen_data_input', \
# 			'seen_data_output', 'unseen_data_output', 'seen_attr_mat', 'unseen_attr_mat']

# data = {}
# for key in imp_keys:
# 	data[key] = getattr(x, key)
# del x

# data['seen_data_input'] = data['seen_data_input'].astype(np.float)
# data['unseen_data_input'] = data['unseen_data_input'].astype(np.float)	
# data['seen_class_ids'] -= 1
# data['unseen_class_ids'] -= 1
# data['seen_data_output'] -= 1
# data['unseen_data_output'] -= 1

# for key, value in data.items():
# 	if(isinstance(value, np.ndarray)):
# 		print(key, value.shape, value.dtype)
# 	else:
# 		print(key, type(value))

# print(data['unseen_data_output'].min(), data['unseen_data_output'].max())
# print(data['seen_data_output'].min(), data['seen_data_output'].max())
# print(data['unseen_class_ids'], data['seen_class_ids'])