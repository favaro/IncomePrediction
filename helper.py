
# helper file containing objects and functions used by all scripts
import pandas as pd
from sklearn import preprocessing


names = [
	("AGE", "age"),
	("ACLSWKR", "class of worker"),   # ?
	("ADTIND", "industry code"),	# ?
	("ADTOCC", "occupation code"),  # ?
	("EDU", "education"),
	("AHRSPAY", "wage per hour"),       # ??
	("AHSCOL", "enrolled in edu inst last wk"),  # ?
	("MARITAL", "marital status"),
	("MAJIND", "major industry code"),   # ?
	("MAJOCC", "major occupation code"), # ?
	("RACE", "race"),    # typo! (corrected)
	("isHISPANIC", "hispanic Origin"),
	("SEX", "sex"),
	("AUNMEM", "member of a labor union"),  #???? 
	("AUNTYPE", "reason for unemployment"),
	("SCHED", "full or part time employment stat"),
	("CAPGAIN", "capital gains"),
	("CAPLOSS", "capital losses"),
	("DIVVAL", "divdends from stocks"),
	("FILESTAT", "tax filer status"),
	("GRINREG", "region of previous residence"),
	("GRINST", "state of previous residence"),
	("HHDFMX", "detailed household and family stat"),  
	("HHDREL", "detailed household summary in household"),  
	("FEDTAX", "federal income tax liability"),   #???? not sure this is the right one...
	("MIGMTR1", "migration code-change in msa"),
	("MIGMTR3", "migration code-change in reg"),
	("MIGMTR4", "migration code-move within reg"),
	("MIGSAME", "live in this house 1 year ago"), 
	("MIGSUN", "migration prev res in sunbelt"),  
	("NOEMP", "num persons worked for employer"), 
	("PARENT", "family members under 18"), 		 
	("BIRTHFATHER", "country of birth father"),
	("BIRTHMOTHER", "country of birth mother"),
	("BIRTHSELF", "country of birth self"),
	("CITIZEN", "citizenship"),
	("SEOTR", "own business or self employed"),
	("VETQVA", "fill inc questionnaire for veteran's admin"),
	("VETYN", "veterans benefits"),	
	("WKSWORK", "weeks worked in year"),
	("YEAR", "year"),
	("TAXINC", "taxable income amount")  # last
]	

col_names = [a[0] for a in names]
col_dict = dict(names)	



edu_dict = {

 	" Children": 0,
	" Less than 1st grade": 1,
	" 1st 2nd 3rd or 4th grade": 2,
	" 5th or 6th grade": 3,
	" 7th and 8th grade": 4,
	" 9th grade": 5,
	" 10th grade": 6,
	" 11th grade": 7,
	" 12th grade no diploma": 8,
	" High school graduate": 9,
	" Some college but no degree": 10,
	" Associates degree-occup /vocational": 11,
	" Associates degree-academic program": 12,
	" Bachelors degree(BA AB BS)": 13,
	" Prof school degree (MD DDS DVM LLB JD)": 14,
	" Masters degree(MA MS MEng MEd MSW MBA)": 15,
	" Doctorate degree(PhD EdD)": 16
}
 
edu_ranking = [
 	" Children",
	" Less than 1st grade",
	" 1st 2nd 3rd or 4th grade",
	" 5th or 6th grade",
	" 7th and 8th grade",
	" 9th grade",
	" 10th grade",
	" 11th grade",
	" 12th grade no diploma",
	" High school graduate",
	" Some college but no degree" ,
	" Associates degree-occup /vocational",
	" Associates degree-academic program",
	" Bachelors degree(BA AB BS)",
	" Prof school degree (MD DDS DVM LLB JD)",
	" Masters degree(MA MS MEng MEd MSW MBA)",
	" Doctorate degree(PhD EdD)",
]

edu_dict2 = {n:r for n,r in enumerate(edu_ranking)}
 
citizen_dict = {
	" Native- Born in the United States": "_native",
	" Foreign born- Not a citizen of U S ": "_foreign",
	" Foreign born- U S citizen by naturalization": "_naturalized",
	" Native- Born abroad of American Parent(s)": "_native_bornabroad",
	" Native- Born in Puerto Rico or U S Outlying": "_native_outlying"
}



def encode(feature_name, data):

	binaryfeatures = pd.get_dummies(data.loc[:,feature_name])
	# modify the column names to avoid overlaps
	binaryfeatures.columns = binaryfeatures.columns.map(lambda fname: feature_name + fname.replace(" ", "_"))   
	data = data.drop(feature_name, axis=1).join(binaryfeatures)
	
	return data


def encode_all(feature_names, data):

	for name in feature_names:
		binaryfeatures = pd.get_dummies(data.loc[:,name])
		# modify the column names to avoid overlaps
		binaryfeatures.columns = binaryfeatures.columns.map(lambda fname: name + fname.replace(" ", "_"))
		data = data.drop(name, axis=1).join(binaryfeatures)
	
	return data


def normalize(feature_names, data):

	for name in feature_names:
		col = data[name].values.astype(float) 
		x_scaled = preprocessing.MinMaxScaler().fit_transform(col.reshape(-1,1))
		data = data.drop(name, axis=1)
		data.name = pd.DataFrame(x_scaled)

	return data


def is_missing(value):

	ismiss = 0
	if value.find("Not in universe") == 1 or value.find("?") == 1: ismiss = 1
	
	return ismiss



