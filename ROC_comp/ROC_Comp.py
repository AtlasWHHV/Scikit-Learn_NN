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

start_time =time.time()

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
prob_sig = NN.predict_proba(uncut)
prob_bkg = NN.predict_proba(test_bkg)


'''SCIKIT ROC CURVE'''

logging.info("Creating ROC Curve")

merged_prob = np.concatenate((prob_sig[:,1],prob_bkg[:,1]),axis=0)

#####################################################
#                                                   #
#   NOTE: [:,1] is for all of the second column of  #
#     proba_predict function's output. we are       #
#     sending the probabilities of the event being  #
#     a signal event through the ROC Curve          #
#                                                   #
#####################################################


class_prob_sig = [1]*len(prob_sig[:,1])
class_prob_bkg = [0]*len(prob_bkg[:,1])
merged_prob_class = class_prob_sig + class_prob_bkg


fpr_sig, tpr_sig, thresholds_sig = roc_curve(merged_prob_class, merged_prob)
auc_score= roc_auc_score(merged_prob_class, merged_prob)
text = "Scikit AUC score is: "+ str(auc_score)



''' Loading TMVA Info '''
logging.info("setting up TMVA Data")
tmva_data_loc = "Hv_NN_2tree.root"
tmva_tree = "TestTree"
tmva_MLP = ["MLP"]
tmva_classID = ["classID"]

tmva_data = rnp.root2array(tmva_data_loc,tmva_tree,tmva_MLP)
tmva_classification = rnp.root2array(tmva_data_loc,tmva_tree,tmva_classID)

tmva_classification = tmva_classification.astype(int)
tmva_data = tmva_data.astype(float)


def tmva_converter(array):

	primer = []
	for i in range(0,len(array)):
		if (array[i]==1):
			primer.append(0) 
		else:
			primer.append(1)

	return primer

np.set_printoptions(threshold=np.inf)
tmva_classification = tmva_converter(tmva_classification)

'''TMVA ROC CURVE'''
logging.info("Setting up TMVA Roc Curve")
tmva_fpr_sig, tmva_tpr_sig, tmva_thresholds_sig = roc_curve(tmva_classification, tmva_data)
tmva_auc_score= roc_auc_score(tmva_classification, tmva_data)
tmva_text = "TMVA AUC score is: "+ str(tmva_auc_score)



'''Setting up Histograms'''
logging.info("Setting up Histograms")

fig7 = plt.figure()
plot7 = fig7.add_subplot(1,1,1)
plot7.plot(fpr_sig,tpr_sig,"b", label = "Sci-Kit ROC")
plot7.plot(tmva_fpr_sig, tmva_tpr_sig, "r", label = "TMVA ROC")
plot7.set_title("Scikit vs TMVA Neural Net ROC comparisons")
plot7.set_xlabel("False Positve Rate")
plot7.set_ylabel("True Positive Rate")
plot7.plot([0,1],[0,1],"r--", label = "Random Guess Line")
plot7.set_xlim([0,1])
plot7.set_ylim([0,1])
plot7.grid(True)
plot7.legend(bbox_to_anchor=(.6,.4))
plot7.annotate(text, xy = (0.5,0.1))
plot7.annotate(tmva_text, xy = (.5,.05))



save7 = fig7.savefig("ROC_Comparison.png")


end_time = time.time()
print("completed in", (end_time - start_time)/60, "minutes")

plt.draw()