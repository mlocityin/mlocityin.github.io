# Steps for building effective machine learning models

## Gathering and Analyzing Input Data

First step of machine learning is **gathering data**.

This step is very important because the quality and quantity of data that we gather will directly determine how good our predictive model can be.

Statistical analysis of input data is essential to understand which method is better for making the data stationary.

Stationary input data improves the effectiveness of prediction algorithms.

Visualizations can be generated using mechanisms like seasonal decomposition.

This can provide inputs for the parameters used by time-series prediction algorithms like ARIMA.

Also, Ad-Fuller and Markov hypothesis can be used for statistical analysis of input data.

## Exploratory Data Analysis (EDA) with Visualization

In this step, we analyze relationship between different features with respect to outcome.

We see how different feature values show different outcome.We also plot different kinds of diagrams to **visualize** our data and findings.

The following steps can be followed for EDA:

-   Loading Modules for Data Analysis

-   Loading Data sets

-   Looking into the training data set

-   Looking into the testing data set

-   Relationship between Features

-   Relationship between Features and Outcome

-   Correlating Features

Heat-map can be used Correlation between different features:

-   Positive numbers = Positive correlation, i.e. increase in one feature will increase the other feature & vice-versa.
-   Negative numbers = Negative correlation, i.e. increase in one feature will decrease the other feature & vice-versa.

## Feature Selection

In this step, We drop unnecessary columns/features and keep only the useful ones for our experiment.

Feature selection is for filtering irrelevant or redundant features from your data-set.

The key difference between feature selection and extraction is that feature selection keeps a subset of the original features while feature extraction creates brand new ones.

Some Techniques for feature selection:

-   Backward Elimination
-   Forward Selection
-   Bidirectional Elimination
-   Score Comparison

## Feature Extraction

In this step, Feature extraction is used for creating a new, smaller set of features that still captures most of the useful information.

Again, feature selection keeps a subset of the original features while feature extraction creates new ones.

-   Principal Component Analysis (PCA)

-   Linear Discriminant Analysis (LDA)

-   t-distributed stochastic neighbor embedding (t-SNE)

-   Multiple correspondence analysis (MCA, used to deal with categorical data)

## Data preparation and processing

Next step of machine learning is **Data preparation**, where we ***prepare our data for use in our machine learning training***.

we need to put all our data together, and then randomize the ordering. We don't want the order of our data to affect what we learn.

Then we apply certain processing techniques such as dealing with missing data, Scaling, Normalizing data.

We also need to deal with categorical data, such as labeling and one hot encoding.

## Model Selection

In this step we train various machine learning models and check which can be used for your data set.

This is where test data-set that we set aside earlier comes into play, it allows us to test our model against data that has never been used for training.

It allows us to see how the model might perform against data that it has not yet seen.

This is meant to be representative of how the model might behave in the real world.

## Evaluating Models Performance and Comparison

### Cross-validation


Cross-validation is a statistical method used to estimate the skill of machine learning models.

It is commonly used in applied machine learning to compare and select a model for a given predictive modeling problem because it is easy to understand, easy to implement, and results in skill estimates that generally have a lower bias than other methods.

### Classification Metrics

Classification problems are perhaps the most common type of machine learning problem and as such there are a lots of metrics that can be used to evaluate predictions for these problems.

-   Classification Accuracy.
-   Logarithmic Loss.
-   Area Under ROC Curve.
-   Confusion Matrix.
-   Classification Report.

### Regression Metrics

The most common metrics for evaluating predictions on regression machine learning problems:

-   Mean Absolute Error.
-   Mean Squared Error.
-   R^2.

## Hyper-parameter tuning

Once we've done evaluation, it's possible that we want to see if you can further improve your training in any way.

We can do this by **tuning hyper-parameters**. There were a few parameters we implicitly assumed when we did our training, and now is a good time to go back and test those assumptions and try other values.

The following Techniques can be used:

-   Grid Search Cross Validation
-   Random Search Cross Validation

## Building a Pipeline

The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters.

There are standard workflows in applied machine learning. Standard because they overcome common problems like data leakage in our test harness.

Many ML frameworks provides a Pipeline utility to help automate machine learning workflows.

Pipelines work by allowing for a linear sequence of data transforms to be chained together culminating in a modeling process that can be evaluated.

## Making the Prediction

Machine learning is using data to answer questions. So **Prediction**, or inference, is the step where we get to answer some questions.

This is the point of all this work, where the value of machine learning is realized.
