import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import preprocessing 
from sklearn import linear_model


from helper import col_names, col_dict, edu_dict, citizen_dict
from helper import encode, encode_all, normalize, is_missing



learn = pd.read_csv( 'us_census_full/census_income_learn.csv', names=col_names, header=None )

for title in edu_dict:
	learn['EDU'][learn['EDU'] == title] = edu_dict.get(title)
	
# make citizenship feature more readable:

for citi in citizen_dict:
	learn['CITIZEN'][learn['CITIZEN'] == citi] = citizen_dict.get(citi)
	
# translate target column into 0,1:

le = preprocessing.LabelEncoder()
learn.TAXINC  = le.fit_transform(learn.TAXINC)


# PLOTTING *********************************************************************

pp = PdfPages('explore.pdf')

features_num = ['AGE','EDU','AHRSPAY','NOEMP','CAPGAIN','CAPLOSS','DIVVAL','WKSWORK','SEOTR','VETYN']
features_text = ['MAJIND','ACLSWKR','MARITAL','SEX','AUNMEM','AUNTYPE','RACE','isHISPANIC','PARENT','SCHED','FILESTAT','CITIZEN','BIRTHSELF','MIGSAME','VETQVA']

print "*** Feature Summary *** \n"

for ifeature in features_text:

	ismiss = learn[ifeature].map(is_missing)
	
	print "******************"
	print ifeature
	print learn[ifeature].value_counts()
	print "number of missing values =", ismiss.sum(), " percentage of missing values = ", float(ismiss.sum())/float(len(learn[ifeature]))*100.0
	

	fig, axes = plt.subplots(2, sharex=True)
	fig.subplots_adjust(top=1.0)

	learn[ifeature][learn.TAXINC == 1].value_counts().plot( ax=axes[0], kind='bar', label='positive', alpha=0.5 )
	learn[ifeature][learn.TAXINC == 0].value_counts().plot( ax=axes[1], kind='bar', label='negative', alpha=0.5 )
	title_pos = ifeature+' positive'
	title_neg = ifeature+' negative'
	
	axes[0].set_title(title_pos)
	axes[1].set_title(title_neg)
	plt.show()
	#plt.savefig(pp, format='pdf')


for jfeature in features_num:

	print "******************"
	print jfeature
	print learn[jfeature].value_counts()
	#print "most frequent value = %6.3f * max = %6.3f * min = %6.3f * percentage of missing values = %2.3%" % ( learn[ifeature].median(), learn[ifeature].max(), learn[ifeature].min(), 10.0 )   
	print "most frequent value =", learn[jfeature].median(), "* min =", learn[jfeature].min(), "* max =", learn[jfeature].max(), " * fraction of missing values =", learn[jfeature].isnull().sum()



	fig, axes = plt.subplots(2, sharex=True)
	learn[jfeature][learn.TAXINC == 1].plot( ax=axes[0], kind='hist', label='positive', alpha=0.5, normed=True )  # normalize to compare shapes
	learn[jfeature][learn.TAXINC == 0].plot( ax=axes[1], kind='hist', label='negative', alpha=0.5, normed=True ) 
	title_pos = jfeature+' positive'
	title_neg = jfeature+' negative'

	axes[0].set_title(title_pos)
	axes[1].set_title(title_neg)

	plt.show()

	#plt.savefig(pp, format='pdf')	


# END PLOTTING *****************************************************************







