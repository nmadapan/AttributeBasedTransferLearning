import numpy as np
import pickle as cPickle
import bz2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.io import loadmat, savemat

######################################################
#### Process Animal with Attributes (AwA) Dataset ####
######################################################

'''
Required files
	* ./classes.txt
	* ./numexamples.txt
	* ./testclasses.txt
	* ./trainclasses.txt
	* ./predicate-matrix-{binary}.txt # when predicate_type = 'binary'
	* ./CreatedData/train_features_index.txt
	* ./CreatedData/test_features_index.txt
	* ./CreatedData/train_featuresVGG19.pic.bz2
	* ./CreatedData/test_featuresVGG19.pic.bz2
'''

def nameonly(x):
	return x.split('\t')[1]

def loadstr(filename,converter=str):
	return [converter(c.strip()) for c in open(filename).readlines()]

def loaddict(filename,converter=str):
	D={}
	for line in open(filename).readlines():
		line = line.split()
		D[line[0]] = converter(line[1].strip())
	
	return D

def get_full_animals_dict(path):
	animal_dict = {}
	with open(path) as f:
		for line in f:
			(key, val) = line.split()
			animal_dict[val] = int(key)
	return animal_dict

def get_animal_index(path, filename):
	classes = []
	animal_dict = get_full_animals_dict(path + "classes.txt")
	with open(path+filename) as infile:
		for line in infile:
			classes.append(line[:-1])
	return [animal_dict[animal]-1 for animal in classes]

def get_attributes():
	attributes = []
	with open('attributes.txt') as infile:
		for line in infile:
			attributes.append(line[:-1])
	return attributes

def get_class_attributes(path, name='train', predicate_type='binary'):
	animal_index = get_animal_index(path, name+'classes.txt')
	classAttributes = np.loadtxt(path + "predicate-matrix-" + predicate_type + ".txt", comments="#", unpack=False)
	return classAttributes[animal_index]

def create_data(path, sample_index, attributes):
  
	X = bzUnpickle(path)
	
	nb_animal_samples = [item[1] for item in sample_index]
	for i,nb_samples in enumerate(nb_animal_samples):
		if i==0:
			y = np.array([attributes[i,:]]*nb_samples)
		else:
			y = np.concatenate((y,np.array([attributes[i,:]]*nb_samples)), axis=0)
	
	return X,y


def autolabel(rects, ax):
	"""
	Attach a text label above each bar displaying its height
	"""
	for rect in rects:
		if np.isnan(rect.get_height()):
			continue
		else:
			height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%0.3f' % height, 
			ha='center', va='bottom', rotation=90)


def awa_to_dstruct(predicate_type = 'binary'):

	# Get features index to recover samples
	train_index = bzUnpickle('./CreatedData/train_features_index.txt')
	test_index = bzUnpickle('./CreatedData/test_features_index.txt')

	# Get classes-attributes relationship
	train_attributes = get_class_attributes('./', name='train', predicate_type=predicate_type)
	test_attributes = get_class_attributes('./', name='test', predicate_type=predicate_type)

	# Create training Dataset
	print ('Creating training dataset...')
	X_train, y_train = create_data('./CreatedData/train_featuresVGG19.pic.bz2',train_index, train_attributes)
	
	# Convert from sparse to dense array
	print ('X_train to dense...')
	X_train = X_train.toarray()

	print ('Creating test dataset...')
	X_test, y_test = create_data('./CreatedData/test_featuresVGG19.pic.bz2',test_index, test_attributes)

	# Convert from sparse to dense array
	print ('X_test to dense...')
	X_test = X_test.toarray()    

	classnames = loadstr('classes.txt', nameonly)
	numexamples = loaddict('numexamples.txt', int)
	test_classnames=loadstr('testclasses.txt')
	train_classnames=loadstr('trainclasses.txt')

	test_classes = [ classnames.index(c) for c in test_classnames]
	train_classes = [ classnames.index(c) for c in train_classnames]

	test_output = []
	for idx, c in enumerate(test_classes):
		test_output.extend( [idx]*numexamples[classnames[c]] )

	train_output = []
	for idx, c in enumerate(train_classes):
		train_output.extend( [idx]*numexamples[classnames[c]] )

	data = {}

	data['seen_class_ids'] = np.array(train_classes).astype(np.uint8)
	data['unseen_class_ids'] = np.array(test_classes).astype(np.uint8)

	data['seen_data_input'] = X_train
	data['seen_data_output'] = np.array(train_output).astype(np.uint8)
	
	data['unseen_data_input'] = X_test
	data['unseen_data_output'] = np.array(test_output).astype(np.uint8)
	
	data['seen_attr_mat'] = train_attributes
	data['unseen_attr_mat'] = test_attributes

	return data

######################################################
######### Plotting confusion and roc curves ##########
######################################################

def plot_confusion(confusion, classes, wpath = ''):
	fig=plt.figure()
	plt.imshow(confusion,interpolation='nearest',origin='upper')
	plt.clim(0,1)
	plt.xticks(np.arange(0,len(classes)),[c.replace('+',' ') for c in classes],rotation='vertical',fontsize=24)
	plt.yticks(np.arange(0,len(classes)),[c.replace('+',' ') for c in classes],fontsize=24)
	plt.axis([-.5, len(classes)-.5, -.5, len(classes)-.5])
	plt.setp(plt.gca().xaxis.get_major_ticks(), pad=18)
	plt.setp(plt.gca().yaxis.get_major_ticks(), pad=12)
	fig.subplots_adjust(left=0.30)
	fig.subplots_adjust(top=0.98)
	fig.subplots_adjust(right=0.98)
	fig.subplots_adjust(bottom=0.22)
	plt.gray()
	plt.colorbar(shrink=0.79)
	if(len(wpath) == 0):
		plt.show()
	else:
		plt.savefig(wpath)
	return 

def plot_roc(P, GT, classes, wpath = ''):
	AUC=[]
	CURVE=[]
	for i,c in enumerate(classes):
		fp, tp, _ = roc_curve(GT == i,  P[:,i])
		roc_auc = auc(fp, tp)
		print ("AUC: %s %5.3f" % (c,roc_auc))
		AUC.append(roc_auc)
		CURVE.append(np.array([fp,tp]))

	print ("----------------------------------")
	print ("Mean classAUC %g" % (np.mean(AUC)*100))

	order = np.argsort(AUC)[::-1]
	styles=['-','-','-','-','-','-','-','--','--','--']
	plt.figure(figsize=(9,5))
	for i in order:
		c = classes[i]
		plt.plot(CURVE[i][0],CURVE[i][1],label='%s (AUC: %3.2f)' % (c,AUC[i]),linewidth=3,linestyle=styles[i])
	
	plt.legend(loc='lower right')
	plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0], [r'$0$', r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'],fontsize=18)
	plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0], [r'$0$', r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'],fontsize=18)
	plt.xlabel('false positive rate',fontsize=18)
	plt.ylabel('true positive rate',fontsize=18)
	if(len(wpath) == 0): plt.show()
	else: plt.savefig(wpath)
	return AUC, CURVE


def plot_attAUC(P, y_true, wpath, attributes = None):
	AUC=[]
	if(attributes is None):
		attributes = map(str, range(y_true.shape[1]))

	for i in range(y_true.shape[1]):
		fp, tp, _ = roc_curve(y_true[:,i],  P[:,i])
		roc_auc = auc(fp, tp)
		AUC.append(roc_auc)
	print ("Mean attrAUC %g" % (np.nanmean(AUC)) )

	xs = np.arange(y_true.shape[1])
	width = 0.5

	# fig = plt.figure(figsize=(15,5))
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	rects = ax.bar(xs, AUC, width, align='center')
	ax.set_xticks(xs)
	ax.set_xticklabels(attributes,  rotation=90)
	ax.set_ylabel("area under ROC curve")
	autolabel(rects, ax)
	if(len(wpath) == 0): plt.show()
	else: plt.savefig(wpath)
	return AUC

######################################################
################## General functions #################
######################################################

def bzPickle(obj,filename):
	f = bz2.BZ2File(filename, 'wb')
	cPickle.dump(obj, f)
	f.close()

def bzUnpickle(filename):
	return cPickle.load(bz2.BZ2File(filename))

def print_dict(dict_inst, idx = 1):
	for key, value in dict_inst.items():
		if(isinstance(value, dict)):
			print('\t'*(idx-1), key, ': ')
			print_dict(value, idx = idx+1)
		else:
			print('\t'*idx, key, ': ', end = '')
			if(isinstance(value, np.ndarray)):
				print(value.shape)
			else: print(value)

######################################################
####################### Others #######################
######################################################

def reformat_dstruct(data_path):
	'''
		Description:
			Convert the ZSL data file that we have in Windows/Matlab into a
			format compatible with python DAP codes. 
		Input:
			data_path: path to the .mat file. This file has a matlab struct variable 'dstruct'
				Example: # r'/home/isat-deep/Desktop/Naveen/fg2020/data/raw_feat_data/data_0.11935.mat'
		Output: 
			data: dictionary with the following keys
				1. seen_data_input: np.ndarray (train_num_instances x feature_size)
				2. unseen_data_input: np.ndarray (test_num_instances x feature_size)
				3. seen_data_output: np.ndarray (train_num_instances, )
				4. unseen_data_output : np.ndarray (test_num_instances, )
				5. seen_class_ids: np.ndarray (num_seen_classes, )
				6. unseen_class_ids: np.ndarray (num_unseen_classes, )
				7. seen_attr_mat: np.ndarray (num_seen_classes, num_attributes)
				8. unseen_attr_mat: np.ndarray (num_unseen_classes, num_attributes)

	'''
	x = loadmat(data_path, struct_as_record = False, squeeze_me = True)['dstruct']

	imp_keys = ['unseen_class_ids', 'seen_class_ids', 'seen_data_input', 'unseen_data_input', \
				'seen_data_output', 'unseen_data_output', 'seen_attr_mat', 'unseen_attr_mat']

	data = {}
	for key in imp_keys:
		data[key] = getattr(x, key)
	del x

	data['seen_data_input'] = data['seen_data_input'].astype(np.float)
	data['unseen_data_input'] = data['unseen_data_input'].astype(np.float)

	data['seen_data_output'] = data['seen_data_output'].astype(np.uint8) - 1 # Matlab indices start from 1
	data['unseen_data_output'] = data['unseen_data_output'].astype(np.uint8) - 1 # Matlab indices start from 1
	
	data['seen_class_ids'] = data['seen_class_ids'].astype(np.uint8) - 1 # Matlab indices start from 1
	data['unseen_class_ids'] = data['unseen_class_ids'].astype(np.uint8) - 1 # Matlab indices start from 1

	return data