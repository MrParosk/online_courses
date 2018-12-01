# Notes from the course "Introduction to Machine Learning for Coders" (2018)

## Lesson 1

For price/sales target variables, one usually only cares about the percentage error, not the absolute one. Furthermore, the log transform is suitable for target variables when individual data points have different order of magnitude.

In order to deal with missing values of continuous features, one could create a separate bool feature "is_na" for that feature and replace the nan values with the feature's median. For missing values of discrete/categorical features, one could introduce a separate categorical value for missing values.

## Lesson 2

In machine learning modelling, creating a "good" validation set is one of the most important tasks. For example, if the data has a time component, it is usually a good idea to split the data based on time (e.g. the training set for 2009-2012 and the validation set for 2013-2014).

When we have a lot of data and doing feature/model exploration, it is a good practice to use a subset of the training data during this phase to speed up the exploration.

### Random forest's hyperparameters

One should use as many trees as possible, given computational constraints.

If we have a small dataset, oob score can be used to estimate the out of sample performance without splitting the dataset into a training and validation set.

For "normal" sized datasets, 1-25 minimum leaf size is usually good. For larger sized dataset, up to 1000 might be needed.

Random forest bootstraps data per tree, and feature subsets per split.

Choosing random feature subsets for split consideration can help with interactions between features (i.e. a feature can be sub-optimial locally, but good "globally"). Good values of the feature subset parameter is sqrt(num_features), log2(num_features) and 0.5 * num_features.

Encoding nominal categorical features is usually okay in random forest, since multiple splits can be made on one feature. However, it's a bad practice for most models, e.g. linear regression.