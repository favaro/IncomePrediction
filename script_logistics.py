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

# the education is ordered (similar to age), so i'm fine with considering it a continuum:

for title,value in edu_dict.iteritems():
	learn_reduced.loc[learn_reduced.loc[:,'EDU'] == title,'EDU'] = value
	test_reduced.loc[test_reduced.loc[:,'EDU'] == title,'EDU'] = value


# make citizenship feature more readable:

for citi,value in citizen_dict.iteritems():
	learn_reduced.loc[learn_reduced['CITIZEN'] == citi,'CITIZEN'] = value
	test_reduced.loc[test_reduced['CITIZEN'] == citi,'CITIZEN'] = value

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

# for logistic regression normalize the numerical features:

to_normalize = [
	'AGE',
	'EDU',
	'AHRSPAY',
	'CAPGAIN',
	'CAPLOSS',
	'DIVVAL',
	'WKSWORK',
]


#learn_reduced = normalize(to_normalize, learn_reduced)
#test_reduced = normalize(to_normalize, test_reduced)


target_train = learn_reduced['TAXINC'].values
learn_reduced = learn_reduced.drop('TAXINC',axis=1)
xvalues_train =  learn_reduced.values

target_test = test_reduced['TAXINC'].values
test_reduced = test_reduced.drop('TAXINC',axis=1)
xvalues_test =  test_reduced.values

# run the logistic regression:    ************************************************

print "\n"
print "*** logistic regression performance, loop over C values ***"
print "\n"


C_parms = [0.001, 0.01, 0.1, 1.0, 10, 1000, 1e5, 1e7]

for ic in C_parms:

	regr = linear_model.LogisticRegression(C=ic)
	regr.fit(xvalues_train, target_train)

	# test the performance and pick a C value:    ****************************

	accuracy_train = regr.score(xvalues_train, target_train)
	accuracy_test = regr.score(xvalues_test, target_test)

	pred_train = regr.predict(xvalues_train)
	pred_test = regr.predict(xvalues_test)

	truepos_train = ((target_train==1)&(pred_train==1))
	truepos_test = ((target_test==1)&(pred_test==1))

	purity_train = float(truepos_train.sum())/float(pred_train.sum())
	purity_test = float(truepos_test.sum())/float(pred_test.sum())

	eff_train = float(truepos_train.sum())/float(target_train.sum())
	eff_test = float(truepos_test.sum())/float(target_test.sum())

	print "   * C = %6.3f" % (ic)
	print "   * on training set: accuracy: %6.4f purity: %6.4f efficiency: %6.4f" % (
			accuracy_train, purity_train, eff_train
		)

	print "   * on test set: accuracy: %6.4f purity: %6.4f efficiency: %6.4f" % (
			accuracy_test, purity_test, eff_test
		)









