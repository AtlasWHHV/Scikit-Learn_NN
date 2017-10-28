import scipy
import numpy as np
import logging
import root_numpy as rnp
from sklearn import neural_network as nn
import time
import matplotlib.pyplot as plt
import tqdm
import os
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.feature_selection import f_classif, SelectPercentile


logging.basicConfig(level = logging.DEBUG)


def loader(file_type, data):
	test_sig = "/phys/users/arbo94/Decision_Tree/Inputs/testingsignalII.root"
	train_sig = "/phys/users/arbo94/Decision_Tree/Inputs/trainingsignalII.root"
	test_bkg = "/phys/users/arbo94/Decision_Tree/Inputs/testingbkg.root"
	train_bkg = "/phys/users/arbo94/Decision_Tree/Inputs/trainingbkg.root"
	uncut = "/phys/users/arbo94/Decision_Tree/Inputs/Hv_Uncutrad_signal.root"
	branchNames =["Event_Num","Jet_Num","Jet_Pt","Cal_Ratio","Track_Count"]

	if (file_type == "train"):
		if (data == "sig"):
			array = rnp.root2array(train_sig, "Training_Variables", branchNames)
			return array
		elif(data == "bkg"):
			array = rnp.root2array(train_bkg, "Training_Variables", branchNames)
			return array
		else:
			logging.info("Not a valid data option.")

	elif (file_type == "test"):
		if (data == "sig"):
			array = rnp.root2array(test_sig, "Training_Variables", branchNames)
			return array
		elif(data == "bkg"):
			array = rnp.root2array(test_bkg, "Training_Variables", branchNames)
			return array
		elif(data =="uncut"):
			array = rnp.root2array(uncut, "Training_Variables", branchNames)
			return array
		else:
			logging.info("Not a valid data option.")

	else:
		logging.info("Not a valid Train/Test option")

def cuts(array):
	primer = []
	for event in tqdm.tqdm(array):
		if (event[1] >=25000):
			primer.append(event)
		else:
			pass

	return primer

def reshape(array):
	primer  = np.column_stack((array["Cal_Ratio"],array["Jet_Pt"],array["Track_Count"]))
	return primer
	#Im removing event number and jet number to not confuse the Tree, as we cant hide variables



''' Loading arrays '''
logging.info("Loading Arrays from .root file")

logging.info("testingbkg.root")
test_bkg = loader("test","bkg")

logging.info("uncut rootfile")
uncut = loader("test","uncut")

logging.info("Loading completed")



''' Reshaping Arrays'''
logging.info("Reshaping Arrays for correct input")

test_bkg = reshape(test_bkg)
logging.info("Done with test_bkg")

uncut = reshape(uncut)
logging.info("Done with uncut")


logging.info("Applying cuts to data")
test_bkg= cuts(test_bkg)
uncut = cuts(uncut)


'''Standardization'''
logging.info("Standardizing Data")
scaler = joblib.load("/phys/groups/tev/scratch4/users/arbo94/NN/NNScaler2/NNScaler2.pkl")
uncut = scaler.transform(uncut)
test_bkg = scaler.transform(test_bkg)


'''Loading Classifier '''

NN = joblib.load("/phys/groups/tev/scratch4/users/arbo94/NN/NNClassifier2/NNClassifier2.pkl")


'''Predictions'''

logging.info("Probability Predicting")
prob_sig = NN.predict(uncut)#NN.predict_proba(uncut)
prob_bkg = NN.predict(test_bkg)#NN.predict_proba(test_bkg)

print (prob_sig)

'''GET PARAMS'''
logging.info("Getting Params")
params = NN.get_params()

print(params)

#input("Press Enter To Continue")


''' classification_report '''
logging.info("Getting Classification Report")


merged_prob = np.concatenate((prob_sig,prob_bkg), axis=0)


class_prob_sig = [1]*len(prob_sig)
class_prob_bkg = [0]*len(prob_bkg)
merged_prob_class = class_prob_sig + class_prob_bkg




labels = ["Background", "Signal"]

report = classification_report(merged_prob_class, merged_prob, target_names = labels )
print(report)

''' Looking At Classifier Attributes'''
print("Classifier Attributes:")
print(NN.coefs_)
print(NN.intercepts_)

''' testing feature selection '''
print("Testing Feature Selection:")
feat_select = SelectPercentile()
