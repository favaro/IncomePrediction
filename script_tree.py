import sys
from optparse import OptionParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import preprocessing 
from sklearn import linear_model
from sklearn import tree


from helper import col_names, col_dict, edu_dict, citizen_dict
from helper import encode, encode_all, normalize


learn = pd.read_csv( 'us_census_full/census_income_learn.csv', names=col_names, header=None )
test = pd.read_csv( 'us_census_full/census_income_test.csv', names=col_names, header=None )


# clean/prepare the dataset:    ************************************************


# select the features to use:
feature_list = [
	'AGE',
	'EDU',
	'SEX',
	'ACLSWKR',
	'RACE',
	'SCHED',
	'CAPGAIN',
	'CAPLOSS',
	'DIVVAL',
	'CITIZEN',
	'BIRTHSELF',
	'WKSWORK',
	'TAXINC'
]

learn_reduced = learn[feature_list]
test_reduced = test[feature_list]

# the education is ordered (similar to age), so i'm fine with considering it a continuum:

learn_reduced = learn_reduced.copy()
for title,value in edu_dict.iteritems():
	learn_reduced.loc[learn_reduced.loc[:,'EDU'] == title,'EDU'] = value
	test_reduced.loc[test_reduced.loc[:,'EDU'] == title,'EDU'] = value

# make citizenship feature more readable:

for citi,value in citizen_dict.iteritems():
	learn_reduced.loc[learn_reduced['CITIZEN'] == citi,'CITIZEN'] = value
	test_reduced.loc[test_reduced['CITIZEN'] == citi,'CITIZEN'] = value


# translate target column into 0,1:

le = preprocessing.LabelEncoder()
learn_reduced.loc[:,'TAXINC']  = le.fit_transform(learn.loc[:,'TAXINC'])
test_reduced.loc[:,'TAXINC']  = le.fit_transform(test.loc[:,'TAXINC'])


# encode categorical data:

to_encode = ['SEX',
	'RACE',
	'CITIZEN',
	'SCHED',
	'BIRTHSELF',
	'ACLSWKR'
	]

learn_reduced = encode_all(to_encode, learn_reduced)
test_reduced = encode_all(to_encode, test_reduced)

# (do not normalize when using a tree)


target_train = learn_reduced['TAXINC'].values
learn_reduced = learn_reduced.drop('TAXINC',axis=1)
xvalues_train =  learn_reduced.values

target_test = test_reduced['TAXINC'].values
test_reduced = test_reduced.drop('TAXINC',axis=1)
xvalues_test =  test_reduced.values



# run the tree:    ************************************************

print "training on learning set..."


print "\n"
print "*** decision tree performance ***"
print "\n"

	

for idepth in range(3,15):

	print "*** depth = ", idepth

	treeclass = tree.DecisionTreeClassifier(max_depth = idepth)
	treeclass.fit(xvalues_train, target_train)


	accuracy_train = treeclass.score(xvalues_train, target_train)
	accuracy_test = treeclass.score(xvalues_test, target_test)

	pred_train = treeclass.predict(xvalues_train)
	pred_test = treeclass.predict(xvalues_test)

	truepos_train = ((target_train==1)&(pred_train==1))
	truepos_test = ((target_test==1)&(pred_test==1))

	purity_train = float(truepos_train.sum())/float(pred_train.sum())
	purity_test = float(truepos_test.sum())/float(pred_test.sum())

	eff_train = float(truepos_train.sum())/float(target_train.sum())
	eff_test = float(truepos_test.sum())/float(target_test.sum())

	print "   * on training set: accuracy: %6.4f purity: %6.4f efficiency: %6.4f" % (
		accuracy_train, purity_train, eff_train
	)
	print "   * on test set: accuracy: %6.4f purity: %6.4f efficiency: %6.4f" % (
		accuracy_test, purity_test, eff_test
	)
	#print "\n"

	# print the relevance of each feature:
	fimp = treeclass.feature_importances_

	print "*** feature relevance ***"
	#print "\n"
	for ifeature in range(len(fimp)):
		print "feature: %s * importance: %6.4f" % (learn_reduced.columns[ifeature], fimp[ifeature])












