
DESCRIPTION OF FILES AND INSTRUCTIONS:

The directory contains four source files: two scripts to access the data and apply the two learning algorithms, 
one to explore the variables and visualize some of them with histograms, one with a few common functions  

* script_logistics.py
accesses the data, performs the logistic regression with several values of the regularization parameter C, and 
prints on screen the performance on the training and test datasets.

* script_tree.py
accesses the data, creates a normal decision tree and applies it with different depths, and prints on screen the performance 
on the training and test datasets

* plotter.py
prints on screen the statistical summary for most variables and produces a few histograms

* helper.py 
function container

The scripts don't require any input from the user.


WHAT I DID:

1) Data cleaning

I import the two datasets as pandas dataframes, extracting the column names from the metadata file. 
I have to do some manual cleaning, since there are a few inconsistencies in the metadata file (missing or swapped features).
Some of the features have a very high fraction of missing values (actually "Not in universe"), so I prefer not to replace those with
values extrapolated from the rest of the data, as it probably wouldn't add much information, and I just avoid using them. 

I only use a subset of the features, basically the cleanest and the ones I believe are more important for the classification. 
It's not the best approach, so with more time and more requirements on the performance I would probably spend more time
and energy on this step. 
I leave the numerical features as they are. I encode the categorical features to binaries, including the target. The education 
is converted into an integer, ordered by the height of the degree. 

2) Algorithm and performance

I implement a logistic regression and a decision tree, and tune a bit their parameters to try to improve their performance.
The data is skewed, with only ~6% of records with income > 50'000 (positives), and ~94% of records with income < 50'0000 (negatives), 
so a dummy algorithm always predicting a negative would already have high accuracy. I try to optimize purity and efficiency instead. 


* Logistic regression
I have in total about 120 features. I train several regressions with different values of the regularization parameter between 0.001 and 10e7.
For low values of C, corresponding to strong regularization, the purity is around 80%, but the efficiency is very low. With weaker regularization the efficiency improves, while the purity remains around 70%. The algorithm has similar performance on the test set. 
One could do a more detailed purity/efficiency study on a cross-validation dataset to determine the best working point.

* Decision tree
I use fewer input features than for the logistic regression, to make it as simple as possible. I try different depths between 3 and 30, and print out the purity and efficiency. Deeper trees have better performance on the training set, but tend to overfit the data and to fail the classification of the 
test set. 
To improve the performance I would also try to use a boosted decision tree or a random forest.
The most relevant features for the categorization seem to be the education, the gains and dividends from investments, the number of weeks worked, 
and the sex. 







