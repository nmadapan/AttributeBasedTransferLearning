import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn

class SVM(nn.Module):
    """
    Linear Support Vector Machine
    -----------------------------
    This SVM is a subclass of the PyTorch nn module that
    implements the Linear  function. The  size  of  each 
    input sample is 2 and output sample  is 1.
    """
    def __init__(self, din, dout = 1):
        super().__init__()  # Call the init function of nn.Module
        self.fully_connected = nn.Linear(din, dout)  # Implement the Linear function
        
    def forward(self, x):
        fwd = self.fully_connected(x)  # Forward pass
        return fwd

if __name__ == '__main__':
	# import some data to play with
	iris = datasets.load_iris()
	x = iris.data[:, :2]  # we only take the first two features. 
	y = iris.target

	print(x.shape)
	temp = np.logical_or(y == 0, y == 1)
	y = y[temp]
	y[y == 0] = -1
	x = x[temp, :]

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

	C = 1.0  # SVM regularization parameter
	svc = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)

	y_pred = svc.predict(x_test)
	f1 = f1_score(y_test, y_pred)

	print(f1)

	din = x_train.shape[1]
	
	device = torch.device('cuda')

	x_train = torch.FloatTensor(x_train).cuda()
	y_train = torch.FloatTensor(y_train).cuda()
	x_test = torch.FloatTensor(x_test).cuda()
	y_test = torch.FloatTensor(y_test).cuda()

	model = SVM(din = din).cuda()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

	for _ in range(5000):
		optimizer.zero_grad()
		output = model(x_train)
		loss = torch.mean(torch.clamp(1 - output * y_train, min=0))  # hinge loss
		loss.backward()  # Backpropagation
		optimizer.step()  # Optimize and adjust weights

	y_pred = model(x_train)
	y_pred[y_pred > 0] = 1
	y_pred[y_pred <= 0] = 0
	y_train[y_train == -1] = 0
	# print(y_pred)
	# print(y_train)
	y_train, y_pred = y_train.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
	f1 = f1_score(y_train, y_pred)
	print(f1_score)