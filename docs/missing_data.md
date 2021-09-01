# Dealing with Missing Data in Machine Learning Datasets

Many times we need to deal with data in which values in some features is missing.  

The typical approaches of dealing with missing values for a feature include:

-   Removing the examples with missing features from the data-set. That can be done if data-set is big enough so we can sacrifice some training examples.
-   Using a machine learning algorithm which implicitly deals with missing feature values.
-   Using a data imputation techniques (e.g. Predicting the missing value using regression techniques)

Before Learning methods of data imputation, we need to understand the reason why data goes missing.

-   Missing at Random (MAR): Missing at random means that the tendency for a data point to be missing is not related to the missing data, but it is related to some of the observed data.
-   Missing Completely at Random (MCAR): The fact that a certain value is missing has nothing to do with its hypothetical value or with the values of other variables.
-   Missing not at Random (MNAR): Two possible reasons are that the missing value depends on the hypothetical value or missing value is dependent on some other variable's value.

Data Imputation techniques for categorical features
---------------------------------------------------

-   Most Frequent is a statistical strategy to impute missing values which works with categorical features (strings or numerical representations) by replacing missing data with the most frequent values within each column.

-   Missing values can be treated as a separate category by itself. We can create another category for the missing values and use them as a different level.

-   We can create a predictive model to estimate values that will substitute the missing data In this case, we divide our data set into two sets: One set with no missing values for the variable (training) and another one with missing values (test).  We can use methods like logistic regression for prediction

Data Imputation techniques for continuous features
--------------------------------------------------

-   This technique consists of replacing the missing value of a feature by an Mean/Median/Mode value of this feature.

-   Another technique is to replace the missing value by the same value outside the normal range of values. For example, if the normal range is [20, 50], then we can set the missing value equal to 100 or -20. The idea is that the learning algorithm will learn what is it better to do when the feature has a value significantly different from other values.

-   Alternatively, we can replace the missing value by a value in the middle of the range. For example, if the range for a feature is [-100, 100], we can set the missing value to be equal to 0. Here, the idea is that if we use the value in the middle of the range to replace missing features, such value will not significantly affect the prediction.

-   An advanced technique is to use the missing value as the target variable for a regression (e.g. k-NN) problem. To build training examples, we only use those examples from the original data-set, in which the value of feature is present.

-   Finally, if we have a significantly large data-set and just a few features with missing values, we can increase the dimensionality of feature vectors by adding a binary indicator feature for each feature with missing values. The missing feature value then can be replaced by 0 or any number of our choice.

During prediction  we should use the same data imputation technique to fill the missing features as the technique we used to complete the training data.

Before we start working on the learning problem, we cannot tell which data imputation technique will work the best. Try several techniques, build several models and select the one that works the best.
