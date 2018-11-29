# Notes from the course "Introduction to Machine Learning for Coders" (2018)

## Lesson 1

For price/sales target variables, one usually only cares about the percentage error, not the absolute one. Furthermore, the log transform is suitable for target variables when individual data points have different order of magnitude.

In order to deal with missing values of continuous features, one could create a separate bool feature "is_na" for that feature and replace the nan values with the feature's median. For missing values of discrete/categorical features, one could introduce a separate categorical value for missing values.

## Lesson 2

In machine learning modelling, creating a "good" validation set is one of the most important tasks. For example, if the data has a time component, it is usually a good idea to split the data based on time (e.g. the training set for 2009-2012 and the validation set for 2013-2014).