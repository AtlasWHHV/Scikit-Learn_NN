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

logging.info("trainingsignal.root")
train_sig = loader("train","sig")

logging.info("trainingbkg.root")
train_bkg = loader("train","bkg")

logging.info("testsignal.root")
test_sig = loader("test","sig")

logging.info("testingbkg.root")
test_bkg = loader("test","bkg")

logging.info("uncut rootfile")
uncut = loader("test","uncut")

logging.info("Loading completed")



''' Reshaping Arrays'''
logging.info("Reshaping Arrays for correct input")

train_sig = reshape(train_sig)
logging.info("Done with Train_sig")

test_sig= reshape(test_sig)
logging.info("Done with test_sig")

train_bkg= reshape(train_bkg)
logging.info("Done with train_bkg")

test_bkg = reshape(test_bkg)
logging.info("Done with test_bkg")

uncut = reshape(uncut)
logging.info("Done with uncut")


logging.info("Applying cuts to data")
train_sig = cuts(train_sig)
test_sig = cuts(test_sig)
train_bkg = cuts(train_bkg)
test_bkg= cuts(test_bkg)
uncut = cuts(uncut)


'''Classification Array '''

logging.info("Creating classification Arrays")

sig_class = [1]*len(train_sig)
bkg_class = [0]*len(train_bkg)
class_array = sig_class + bkg_class
class_array = np.array(class_array).astype(np.float32)

'''Standardizing Data'''
logging.info("Standardizing Data")
#From the tips on pratical use section of scikit NN documentation, this is for standardizing the data. ONLY FIT ON TRAINING DATA

training_array = np.concatenate((train_sig, train_bkg), axis = 0)
scaler = StandardScaler()
scaler.fit(training_array)
training_array = scaler.transform(training_array)
uncut = scaler.transform(uncut)
test_bkg = scaler.transform(test_bkg)



''' Creating Classifier '''
logging.info("Creating Neural Net")

NN = nn.MLPClassifier(hidden_layer_sizes=(5,))


'''Training Classifier'''
logging.info("Time to train")

train_start = time.time()
NN.fit(training_array,class_array)
train_end = time.time()

print("Signal training took:",(train_end- train_start)/60, "minutes")


'''Saving Training Classifier'''
logging.info("Saving Classifier")
from sklearn.externals import joblib

if(os.path.isdir("NNClassifier3")==True):
	shutil.rmtree("NNClassifier3")
	os.mkdir("NNClassifier3")
else:
	os.mkdir("NNClassifier3")

joblib.dump(NN, "NNClassifier3/NNClassifier3.pkl")



'''Predictions'''
logging.info("Predicting")
predict_sig = NN.predict(uncut)
predict_bkg = NN.predict(test_bkg)

logging.info("Probability Predicting")
prob_sig = NN.predict_proba(uncut)
prob_bkg = NN.predict_proba(test_bkg)


'''ROC Curve'''
logging.info("Creating ROC Curve")
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

merged_prob = np.concatenate((prob_sig[:,1],prob_bkg[:,1]),axis=0)

print("The Length of Prob_Sig: ")
print(len(prob_sig[:,1]))
print("The Length of Prob_Bkg: ")
print(len(prob_bkg[:,1]))

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

print("The Length of Probability array: ")
print(len(merged_prob))
print("The Length of class array: ")
print(len(merged_prob_class))

'''logging.debug("The length of probability array: ", +len(merged_prob))
logging.debug("The length of class array: ", +len(merged_prob_class))'''

fpr_sig, tpr_sig, thresholds_sig = roc_curve(merged_prob_class, merged_prob)
auc_score= roc_auc_score(merged_prob_class, merged_prob)
text = "AUC score is: "+ str(auc_score)


''' Making Histograms and Trees'''
logging.info("Making Histograms")

fig = plt.figure()
plot1 = fig.add_subplot(1,1,1)
plot1.hist(predict_sig,  histtype ="bar", log = True)
plot1.set_title("NN Response (Signal)")
plot1.set_xlabel("Response (0 = Background, 1= Signal)")
plot1.set_ylabel("# of Events")

fig2 = plt.figure()
plot2 = fig2.add_subplot(1,1,1)
plot2.hist(predict_bkg, histtype ="bar", log = True)
plot2.set_title("NN Response (Background)")
plot2.set_xlabel("Response (0 = Background, 1= Signal)")
plot2.set_ylabel("# of Events")

fig3 = plt.figure()
plot3 = fig3.add_subplot(1,1,1)
plot3.hist(prob_sig[:,0], histtype ="bar", log = True)
plot3.set_title("NN Probability of event being bkg(sig)")
plot3.set_xlabel("Probability of Being Background")
plot3.set_ylabel("# of Events")

fig4 = plt.figure()
plot4 = fig4.add_subplot(1,1,1)
plot4.hist(prob_sig[:,1], histtype ="bar", log = True)
plot4.set_title("NN Probability of event being signal(sig)")
plot4.set_xlabel("Probability of Being Signal")
plot4.set_ylabel("# of Events")

fig5 = plt.figure()
plot5 = fig5.add_subplot(1,1,1)
plot5.hist(prob_bkg[:,0], histtype ="bar", log = True)
plot5.set_title("NN Probability of event being bkg(bkg)")
plot5.set_xlabel("Probability of Being Background")
plot5.set_ylabel("# of Events")

fig6 = plt.figure()
plot6 = fig6.add_subplot(1,1,1)
plot6.hist(prob_bkg[:,1], histtype ="bar", log = True)
plot6.set_title("NN Probability of event being signal(bkg)")
plot6.set_xlabel("Probability of Being Signal")
plot6.set_ylabel("# of Events")

fig7 = plt.figure()
plot7 = fig7.add_subplot(1,1,1)
plot7.plot(fpr_sig,tpr_sig,"b", label = "ROC")
plot7.set_title("Reciever Operating Characteristic for Neural Net(w/ Uncut Data)")
plot7.set_xlabel("False Positve Rate")
plot7.set_ylabel("True Positive Rate")
plot7.plot([0,1],[0,1],"r--", label = "Random Guess Line")
plot7.set_xlim([0,1])
plot7.set_ylim([0,1])
plot7.grid(True)
plot7.legend(bbox_to_anchor=(0.46,1))
plot7.annotate(text, xy = (0.5,0.1))



if(os.path.isdir("graphs3")==True):
	pass
else:
	os.mkdir("graphs3")



save = fig.savefig("graphs3/NN_Predict_Sig.png")
save2 = fig2.savefig("graphs3/NN_Predict_Bkg.png")
save3 = fig3.savefig("graphs3/NN_sig_col0.png")
save4 = fig4.savefig("graphs3/NN_sig_col1.png")
save5 = fig5.savefig("graphs3/NN_bkg_col0.png")
save6 = fig6.savefig("graphs3/NN_bkg_col1.png")
save7 = fig7.savefig("graphs3/NN_ROC_Curve.png")



end_time = time.time()
print("completed in", (end_time - start_time)/60, "minutes")

plt.draw()
