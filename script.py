import sys
from optparse import OptionParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import preprocessing 
from sklearn import linear_model
from sklearn import tree


from colnames import col_names, col_dict
from helper import edu_dict, citizen_dict

parser = OptionParser()
parser.add_option('-a','--algo', dest='algorithm', type ='string')


def encode(feature_name, data):

	binaryfeatures = pd.get_dummies(learn[feature_name])
	# modify the column names to avoid overlaps
	binaryfeatures.columns = binaryfeatures.columns.map(lambda name: feature_name + name.replace(" ", "_"))   
	data = data.drop(feature_name, axis=1).join(binaryfeatures)
	
	return data

def encode_all(feature_names, data):
	for name in feature_names:
		binaryfeatures = pd.get_dummies(learn[name])
		# modify the column names to avoid overlaps
		binaryfeatures.columns = binaryfeatures.columns.map(lambda name: name + name.replace(" ", "_"))   
		data = data.drop(name, axis=1).join(binaryfeatures)
	
	return data


def normalize(feature_names, data):
	for name in feature_names:
		col = data[name].values.astype(float) #returns a numpy array
		x_scaled = preprocessing.MinMaxScaler().fit_transform(col.reshape(-1,1))
		data = data.drop(name, axis=1)
		data[name] = pd.DataFrame(x_scaled)
	return data



algo = sys.argv[1]
print 'going to predict using:', algo, '\n'

learn = pd.read_csv( 'us_census_full/census_income_learn.csv', names=col_names, header=None )
test = pd.read_csv( 'us_census_full/census_income_test.csv', names=col_names, header=None )


# clean/prepare the dataset:    ************************************************


# select the features to use:
feature_list = [
	'AGE',
	'EDU',
	'AHRSPAY',
	'MAJIND',
	'ACLSWKR',
	'MARITAL',
	'SEX',
	'RACE',
	'isHISPANIC',
	'SCHED',
	'CAPGAIN',
	'CAPLOSS',
	'DIVVAL',
	'FILESTAT',
	'CITIZEN',
	'BIRTHSELF',
	'WKSWORK',
	'TAXINC'
]

learn_reduced = learn[feature_list]
test_reduced = test[feature_list]
#learn_reduced = learn[['AGE','EDU','AHRSPAY','MAJIND','CAPGAIN', 'CAPLOSS', 
#	'MARITAL','SEX','RACE','isHISPANIC',
#	'SCHED','DIVVAL','FILESTAT','CITIZEN','BIRTHSELF','WKSWORK','TAXINC']]
#test_reduced = test[['AGE','EDU','AHRSPAY','MAJIND','CAPGAIN', 'CAPLOSS', 
#	'MARITAL','SEX','RACE','isHISPANIC',
#	'SCHED','DIVVAL','FILESTAT','CITIZEN','BIRTHSELF','WKSWORK','TAXINC']]


# the education is ordered (similar to age), so i'm fine with considering it a continuum:

for title,value in edu_dict.iteritems():
	learn_reduced['EDU'][learn_reduced['EDU'] == title] = value
	test_reduced['EDU'][test_reduced['EDU'] == title] = value

# make citizenship feature more readable:

for citi,value in citizen_dict.iteritems():
	learn_reduced['CITIZEN'][learn_reduced['CITIZEN'] == citi] = value
	test_reduced['CITIZEN'][test_reduced['CITIZEN'] == citi] = value

# translate target column into 0,1:

le = preprocessing.LabelEncoder()
learn_reduced.TAXINC  = le.fit_transform(learn.TAXINC)
test_reduced.TAXINC  = le.fit_transform(test.TAXINC)


# encode categorical data:

# sex, race and hispanic origin:

to_encode = ['SEX',
	'RACE',
	'isHISPANIC',
	'BIRTHSELF',
	'SCHED',
	'MARITAL',
	'CITIZEN',
	'FILESTAT',
	'MAJIND',
	'ACLSWKR',
	]

learn_reduced = encode_all(to_encode, learn_reduced)
test_reduced = encode_all(to_encode, test_reduced)

# normalize the numerical features

to_normalize = [
	'AGE',
	'EDU',
	'AHRSPAY',
	'CAPGAIN',
	'CAPLOSS',
	'DIVVAL',
	'WKSWORK',
]

# do not normalize when using a tree:

if algo == 'logistic':
	learn_reduced = normalize(to_normalize, learn_reduced)
	test_reduced = normalize(to_normalize, test_reduced)


target_train = learn_reduced['TAXINC'].values
learn_reduced = learn_reduced.drop('TAXINC',axis=1)
xvalues_train =  learn_reduced.values

target_test = test_reduced['TAXINC'].values
test_reduced = test_reduced.drop('TAXINC',axis=1)
xvalues_test =  test_reduced.values

# run the logistic regression:    ************************************************

if algo == 'logistic':

	C_parms = [0.001, 0.01, 0.1, 1.0, 10, 1000, 1e5, 1e7]

	for ic in C_parms:

		regr = linear_model.LogisticRegression(C=ic)
		regr.fit(xvalues_train, target_train)

		# test the performance and pick a C value:    ****************************

		accuracy_on_train = regr.score(xvalues_train, target_train)
		accuracy_on_test = regr.score(xvalues_test, target_test)

		pred_train = regr.predict(xvalues_train)
		pred_test = regr.predict(xvalues_test)

		truepos_train = ((target_train==1)&(pred_train==1))
		truepos_test = ((target_test==1)&(pred_test==1))

		purity_train = float(truepos_train.sum())/float(pred_train.sum())
		purity_test = float(truepos_test.sum())/float(pred_test.sum())

		eff_train = float(truepos_train.sum())/float(target_train.sum())
		eff_test = float(truepos_test.sum())/float(target_test.sum())

		print "*** C = %10s: accuracy: %6.4f %6.4f purity: %6.4f %6.4f efficiency: %6.4f %6.4f" % (
				str(ic), accuracy_on_train, accuracy_on_test, purity_train, purity_test, eff_train, eff_test
			)




# run the tree:    ************************************************

if algo == 'tree':

	treeclass = tree.DecisionTreeClassifier()
	treeclass.fit(xvalues_train, target_train)


	accuracy_on_train = treeclass.score(xvalues_train, target_train)
	accuracy_on_test = treeclass.score(xvalues_test, target_test)

	pred_train = treeclass.predict(xvalues_train)
	pred_test = treeclass.predict(xvalues_test)

	truepos_train = ((target_train==1)&(pred_train==1))
	truepos_test = ((target_test==1)&(pred_test==1))

	purity_train = float(truepos_train.sum())/float(pred_train.sum())
	purity_test = float(truepos_test.sum())/float(pred_test.sum())

	eff_train = float(truepos_train.sum())/float(target_train.sum())
	eff_test = float(truepos_test.sum())/float(target_test.sum())


	print "***tree accuracy: %6.4f %6.4f purity: %6.4f %6.4f efficiency: %6.4f %6.4f" % (
			accuracy_on_train, accuracy_on_test, purity_train, purity_test, eff_train, eff_test
		)

	











