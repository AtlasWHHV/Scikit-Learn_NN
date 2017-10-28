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

logging.basicConfig(level = logging.info)

""" The logging library allows me to recieve outputs faster than the print function for debugging. At the moment I have removed most of the debug outputs, and left in the info outputs,
but should I want too, I can have everything export into a log file."""

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

""" The loader function here is to take a root file, and put it into a numpy structured array using the root_numpy libary [emphasis on the fact it is a structured array]. 
The basic set up of the function is: 
array = rnp.root2array(File_location, Name_of_TTree, List_of_branch_names). All values should be strings, except of the list, which should be a list of strings. 
The huge if/else is to make typing it out below easier."""



def cuts(array):
	primer = []
	for event in tqdm.tqdm(array):
		if (event[1] >=25000):
			primer.append(event)
		else:
			pass

	return primer

""" This function is to add our Pt cut of 25 GeV]. This could be done with splicing, but I am not comfortable with splicing, hence this horrendous for loop."""


def reshape(array):
	primer  = np.column_stack((array["Cal_Ratio"],array["Jet_Pt"],array["Track_Count"]))
	return primer
	

	"""This function removes event number and jet number to not confuse the Tree, as the classifier cant hide variables from being ran,(unlike TMVA)"""


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


""" Loading array and reshaping array loads the arrray into numpy, reshapes the array, then applies the cut."""

'''Classification Array '''

logging.info("Creating classification Arrays")

sig_class = [1]*len(train_sig)
bkg_class = [0]*len(train_bkg)
class_array = sig_class + bkg_class
class_array = np.array(class_array).astype(np.float32)


""" This grouping of functions creates a classification array, to determine whether an event is signal, or background. 1 is signal, 0 is background events """

'''Standardizing Data'''
logging.info("Standardizing Data")
#From the tips on pratical use section of scikit NN documentation, this is for standardizing the data. ONLY FIT ON TRAINING DATA

training_array = np.concatenate((train_sig, train_bkg), axis = 0)
scaler = StandardScaler()
scaler.fit(training_array)
training_array = scaler.transform(training_array)
uncut = scaler.transform(uncut)
test_bkg = scaler.transform(test_bkg)

""" This standaradization step is very important for neural networks within scikit. The standard scalar function creates a normalized set of data. Sadly I forget which version of 
normalization it uses, but this is important. It takes in all of the data for training, both signal and background, to create the scalar. Then you apply it to the test data with the 
transform method. """



''' Creating Classifier '''
logging.info("Creating Neural Net")

batch_size = len(class_array)

NN = nn.MLPClassifier()
#NN.set_params(hidden_layer_sizes=(5,))
#NN.set_params(alpha = 1)
NN.set_params(hidden_layer_sizes=(5,), activation ="logistic", solver="sgd", batch_size=batch_size,
	learning_rate="adaptive", learning_rate_init = .02, max_iter= 600, random_state = None, shuffle = True, momentum =.01,
	nesterovs_momentum = False)


"""This step is creating the classifier, with the paramaters I would like. The Hidden layer size is odd, because you end the dictionary with an empty entry. What (5,) is one hidden layer
with 5 nodes. (5,5,) would be 2 hidden layers, with 5 nodes each. This hidden layers do not include the input and output layers. The activation function is logistic.
This means that activating the node runs on a unit logistic function. SGD is one of the many training methods Scikit uses. After many different tests, I have determine this 
is the most accurate for our data. The rest can be read from the documentation on scikits website. """


'''Training Classifier'''
logging.info("Time to train")

train_start = time.time()
NN.fit(training_array,class_array)
train_end = time.time()

print("Signal training took:",(train_end- train_start)/60, "minutes")

""" Fit is the training function in scikitlearn """


'''Saving Training Classifier'''
logging.info("Saving Classifier")
from sklearn.externals import joblib

if(os.path.isdir("NNClassifier4")==True):
	shutil.rmtree("NNClassifier4")
	os.mkdir("NNClassifier4")
else:
	os.mkdir("NNClassifier4")

joblib.dump(NN, "NNClassifier4/NNClassifier4.pkl")


""" This set of code saves the classifier as a special pickle type to be used later on as needed"""

'''Saving Scaler'''

logging.info("Saving Scaler")
if(os.path.isdir("NNScaler4")==True):
	pass
else:
	os.mkdir("NNScaler4")

joblib.dump(scaler,"NNScaler4/NNScaler4.pkl")

""" This set of code saves the scalar as a special pickle for later use as needed"""

'''Predictions'''
logging.info("Predicting")
predict_sig = NN.predict(uncut)
predict_bkg = NN.predict(test_bkg)

logging.info("Probability Predicting")
prob_sig = NN.predict_proba(uncut)
prob_bkg = NN.predict_proba(test_bkg)


""" The predict method takes the testing data, and outputs whether or not it is a "0" or a "1". I.e, is it a signal, or background. The outputs are only 0 or 1. No inbetween.
predict_proba is similar to the predict method, but outputs an array, with columns equal to the number of possible outputs in predicit. In this instance, the array is nx2, 
where n = # of events. Each column shows the probability that it is that type of classification. For us, the first column is the probability it is a 0, or background, and the
second column is hte probability it is a signal event, or 1. Adding these two numbers should =1. """


'''ROC Curve'''
logging.info("Creating ROC Curve")
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

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
text = "AUC score is: "+ str(auc_score)


""" The purpose of this group of code is to create the data necessary for a ROC curve. The method to create the roc curve data is as follows:
false_positive, true_positives, thresholds = roc_curve(classification_array, array_of_test_data). This only works because our "test" date is actually known to us. 
This is how we are able to create the ROC curve. All the code above that line is just creating the arrays necessary. the AUC score, or area under the curve
gives you a single number. This number is considered the "accuaracy" of the classifier, which takes into account how well a classifier can identify a true event, and identify 
a background event.

A ROC curve is made with the x axis as the flase positive arrays, and the y axis as the true positive arrays. """


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


""" These 7 graphs are confusing, but they are described as follows. All plots except for 7 are log graphs for the Y axis:
1. This plots the outputs the results from predict purely on the signal data.
2. This plots the outputs of predict purely on the background data.
3. This plots the probabilities of events that are known to be signal to be a background event from predict_proba
4. This plots the probabilities of events that are known to be signal to be a signal event from predict_proba
5. This plots the probabilities of events that are known to be background to be a background event from predict_proba
6. This plots the probabilities of events that are known to be background to be a signal event from predict_proba
7. This plots the ROC curve, with the AUC score written on it. There is a Random guess line, which shows what the ROC score would be if the classifier would just randomly
	guess whether the even was signal or background """


if(os.path.isdir("graphs4")==True):
	pass
else:
	os.mkdir("graphs4")



save = fig.savefig("graphs4/NN_Predict_Sig.png")
save2 = fig2.savefig("graphs4/NN_Predict_Bkg.png")
save3 = fig3.savefig("graphs4/NN_sig_col0.png")
save4 = fig4.savefig("graphs4/NN_sig_col1.png")
save5 = fig5.savefig("graphs4/NN_bkg_col0.png")
save6 = fig6.savefig("graphs4/NN_bkg_col1.png")
save7 = fig7.savefig("graphs4/NN_ROC_Curve.png")



end_time = time.time()
print("completed in", (end_time - start_time)/60, "minutes")

plt.draw()


""" This set of codes saves the graphs as PNG's, and plt.draw should show the pictures up on your screen. """