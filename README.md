# MMAI_NN_housing_energy_prediction
The objective is to use the features in 2009 RECS Survey Data to predict the KWH (Kilowatts per hour) in a residential place. I attempt to compare regression models complexity vs accuracy
There was 12,083 number of responses collected in the survey using a sophisticated multistage, weight probability sampling design.

files:
1. recs2009_public - main feature file
2. recs2009_public_codebook - data dictionary
3. public_layout - data type

download public data here: https://catalog.data.gov/dataset/residential-energy-consumption-survey-recs-files-energy-consumption-2009/resource/954d476e-313b-41f2-9f40-32ebe0c4346c


Analysis and Code Process – 2009 RECS Survey

Executive Summary

The objective of the assignment is to use the features in 2009 RECS Survey Data to predict the KWH (Kilowatts per hour) in a residential place. There was 12,083 number of responses collected in the survey using a sophisticated multistage, weight probability sampling design.

Assumptions

The intricacy of the sampling methodology and the weighting factors for each household survey should be considered in comprehensive modelling processes. For instance, the weight factor of the survey adjusts for population discrepancy between a survey of a house located in Alaska compared to a house in New York City.

Some of the features in the dataset has a very high correlation against the target feature (KWH). For example, DOLELSPH (Electricity cost for space heating, in whole dollars, 2009) and DOLLAREL (Total Electricity cost, in whole dollars, 2009). In my humble opinion, these features are a proxy for KWH. In addition, these data appear to be ex post facto since the objective is to predict the 2009 household KWH. If I were to use the 2009 energy usage features in my models, the prediction would be very accurate (with KWHOTH in the model, the models achieve 0.93 R-squared). However, I feel it did not capture the spirit of the challenge and the for those reasons, I decided not to use any features that are deemed retroactive.

In the survey dataset a large portion contains imputation flags. Some of the survey metrics were based on imputation instead of actual surveyed value. Due to the interest of time, I did not consider incorporating any imputation flag features into my models. Did not conduct comprehensive feature engineering and create new features, for example: (number of garage/number of stoves) ratio, (Money spend on cooling – Money spend on heating)/ total money spend on energy.

Assumption Summary:

    did not take into consideration of survey weights
    did not use any features that are ex post facto (if it is used, it defeats the purpose of prediction)
    imputation flag not used
    did not create additional features

Step 1- Reading Documentation

Read the documentation (recs2009_public_codebook, public layout) of the data. Get an understanding of data types, response codes and labels. After reading the data dictionary, I noticed that the survey feature were corresponding to major housing characteristics. Most of the features are labelled without any ranking relationship.

Step 2- Import data, Descriptive Statistics

I coded 3 functions. Import_data, Descriptive_Statistics and Feature_visualization. I wanted to inspect the data at a high level for any major discrepancy with the data dictionary. The Descriptive_Statistics outputs key descriptive statistics about the features in the dataframe. The function also detects for objects in the dataframe, if object exists it will also display categorical frequency statistics. At this stage, I am scrolling through the df_stats and paying attention to the max, min values, and the NA counts. The max and min inform me of the nature of the data, whether it’s a scalable feature or a labelled feature (candidate of one-hotencoding). For the df_cat_stats I am inspecting the parentage count to get an understanding which categorical features are evenly distributed, and which are sparse. In the event of a sparsely distributed categorical features, there are a few ways to treat the data. 1) Use target encoding 2) hierarchical clustering (sacrifice some explainability). In the interest of time and abundance of other more significant features, I skipped that step. In the Feature_visualization section, I built the functions that take a sample (0,1) of the data to visually inspect the raw distributions of the features and further confirming the candidate features for one-hotencoding.

Output: 12083 observations, 940 features

Step 3- First Iteration of Feature Importance

In this section, I built three functions: random_forest (regression), corr_target and lasso as a primary guide to assist me in filtering out the insignificant features. The functions take a sample (0,1) of the data due to run time constraints. The output displays the raw value of Gini-importance (random forest output, Gini impurity, more weight on categorical), KWH_corr_abs (target feature absolute correlation), lasso_coefficient (L1 penalty forces coefficient to zero) and their respective rankings (descending). The initial screening of the feature importance led me to the conclusion that the most significant features are the in-retrospect feature such as ‘total money spent on energy in 2009'.

With the assistance of the feature importance data, I manually selected 2-3 primary/least imputed features from each of the major housing characteristic categories for dummy feature candidates. For example air conditioner yes/no, car garage yes/no, high ceiling yes/no

I did not dummy all the categorical/int variable:

    some of them are sequentially dependent, if you already have a detached car garage, then it assumes you have a regular car garage

    some of the features are too specific and already captured in a proxy feature. For example: ’Solar used, other than for space heating or water heating’ or ‘Charging patterns for rechargeable electronic devices’

    some of the labelled features naturally scales well, thus no need to hot-encode. For example: number of rooms or age of the occupant

I kept almost all of the float data type features and features that naturally scales. In the keep_and_hotencode function. I dummy encoded the selected feature and kept the numeric features.

Output: 12083 observations, 163 features

Step 4- Second Iteration of Descriptive Statistics, Manually Outlier/Data Adjustments

The number of features has been dramatically cut from 940 to 163 (include dummies). It is currently a manageable number of features to manually inspect. Once again, I called the descriptive_stats function because I wanted to analyze the features for any potential outliers and to make certain the zero_count statistic makes sense. I isolated a few features and used a boxplot to analyze the distribution visually. I decided to remove outlier values of KWH that are greater than 60000 from the dataset. In addition, I also checked for cases where ‘Total heated square footage’ is 0 with positive KWH. But it did makes sense as utilization of the space is attributed to cooling. Furthermore, I replaced value -2 with 0; -2 corresponds to N/A in labelled features. Fortunately, there are no 0 labelled indicators when there is -2 indictor in my dataframe.

Output: 12074 observations, 163 features

Step 5- Second Iteration of Feature Importance

In this step, I elected to check the correlation matrix (163 x 163) because some of the features exhibit high collinearity, in particular, the binary categorical variables when dummied are 100% correlated with other. I took the opportunity to remove high multicollinearity features in this step. Next, I called the feature importance function once again with the 155 features left in the dataset. I built a feature_selection function to select the features that are ranked in the top 40 (2 out of 3 in Gini-importance_rank, lasso_coefficient_rank, KWH_corr_abs_rank). The application of this function returned 30 features that satisfied these criteria. Some of the feature included in the 30 features are ‘total square feet’, ‘washing clothes 5 to 9 loads each week indicator’ and ‘gross household income’.

Output: 12074 observations, 30 features

Step 6- Standardize Data, Check Standardized Distribution, Split_Data

Since this is a regression exercise, standardizing data is important to reduce Multicollinearity, because Multicollinearity could create overfitting in training, it also obscures the statistical significance of other model features. Standardizing data also helps to centre the distribution of features which allows the model to scale more efficiently when calculating the MAE. There is a multitude of ways to standardize/normalize/transform data, log normal, max-min, z-score. Next step, I split the data into training and testing.

Output: X_train: 8451 observations, 29 features, X_test: 3623 observations, 29 features

Modelling Algorithms

Neural Network regression

XGBoost regressor

Random forest regressor

ElasticNet

Lasso

OLS

The above is a hierarchical ranking of some of the popular regression model used in terms of complexity and accuracy (in general). In this assignment, I tried these 3 different models: Neural Network, XGBoost regressor and OLS in an attempt to demonstrate and contrast some of the model’s capability, explainability.

Neural Network

I built a 4-layer nn_model function with 2 drop out layers. The layers use ‘relu’ activation function because Neural networks are trained using stochastic gradient descent. This involves first calculating the prediction error made by the model and using the error to estimate a gradient used to update each weight in the network so that less error is made next time. This error gradient is propagated backwards through the network from the output layer to the input layer. When there are many layers in the network, the gradient diminishes via backpropagation and the error terms are so small when it reaches the output layer. This indicates the model learns a lot in the shallow layers with diminishing return as it gets into deeper layers. The random weight initiated for each layer is represented by kernel_initializer. We are training on the loss ‘mean absolute error’ and optimizer ‘adam’. I prefer ‘adam’ because it combines the attributes of both Adagrad and RMSprop. It is also a newer method, and in my case, it exhibited less fluctuation in validation MAE. The drop out rate randomly drops out neurons to help with overfitting. The epochs define how many times the model runs through the entire train data, greater the number the better the training accuracy (may not be for validation). The batch size indicates how many samples to train on each iteration. The smaller the batch size the higher propensity to overfit. I stopped training the model after 20 epochs because it started to display overfitting tendencies after 20 epochs. I instructed the model to save the best parameters. The r-square for testing data is 0.541, which is slightly below the training data at 0.567. This difference indicates that there is no overfitting. The testing and training R-square value fluctuate slightly due to the randomly initialized weights at each layer. Furthermore, the r-squared is not comparable to classification accuracy.

In practice, I typically start deliberately overfitting (more features) my NN model. It is not a hindrance for a NN model since there are many built-in tools that can reduce overfitting in a NN. Contrast that with OLS model, it is usually not prudent to start training the model with an abundance of features. However, to compare the effectiveness of each model in this exercise, the initial feature inputs are the exact same 30 features for all 3 models.

OLS

There are two popular packages that can be used to build an OLS model, statsmodels and sklearn linear regression. I prefer the statsmodels because it display more inference outputs. In the OLS function we used the same 29 features from NN to get an initial result for the model goodness of fit. The R-squared and the adjusted R-squared are very similar to each other. This suggests the model is not burdened by a lot of excess features. The large F-statistics indicates the overall effectiveness (combine features) of the model is significant. There are 3 features with large p-values (WASHLOAD_-2, MONEYPY, ATHOME_0) we may need to consider removing. Next, I used a forward_regression function to add each feature sequentially and only keeping the feature with p-value smaller than 0.01. There are other feature selection methods for OLS such as backward selection, stepwise selection, selectKbest. In the interest of time, I only attempted to demonstrate with forward selection. After the forward feature selection, 4 features (17, 18, 21, 26) were excluded from the model. The new model’s R-square was almost the same, which suggest effective feature selection in the previous step. A few tasks were omitted from building this model in the interest of time: 1) Autocorrelation in the residual for each feature 2) Heteroskedasticity for each feature.

XGBoost

This is an ensemble model; it is derived and built upon a family of gradient boosting algorithms. The boosting mechanism is decision trees that are built sequentially such that each subsequent tree intents to reduce the error of the predecessors. Each tree that follows the next tree will continue to update and learn from the previous tree’s residuals. Compared against random forest, the algorithm uses bagging method and it aggregates the mean loss of n trees. Considering the dataset contain a significant number of dummied features, XGBoost algorithm works well because it has a built-in sparsity-aware split algorithm, it handles sparse data split very well. Furthermore, XGBoost is relatively fast because the sequential gradient descend optimizer allows the depth of the trees in XGBoost to be relatively shallow compared to random forest where in some instances a deep grown tree is required. Other benefits of XGBoost are related to its coding structure to leverage parallel processing, distributed computing and cache optimization.

In the code, the class XGBoostWithEarlyStop function has a built-in grid search to find the best parameters. It starts off searching the first combination of specified parameters until MAE has not improved in 5 iterations. Then it will save the best parameter. The algorithm will continue to minimize MAE on the next combination of parameters in the param_grid. The final output is the best parameter (lowest MAE) out of the grid search. I manually copied the parameters into a regular XGBoost model and trained, tested on the same dataset (random seed=42) as NN and OLS. Due to time constraints, I only introduced some of the main tunable hyperparameters and significantly limited the search space.

Result Comparison and Interpretation

In this section, I reverse scaled the standardized KWH to the original value. I predicted the testing dataset using all 3 models and complied them into model_compare dataframe. As mentioned above, it is impractical to recreate the same result each time with (Randomness in initialization weights, Regularization dropout) NN. After running the models a few times, the result in general has NN with the highest R-squared score 0.545 slightly above OLS at 0.531 (reproducible). XGBoost comes in at a close third place at 0.522(reproducible).

The NN consistently has the highest performance and it is ‘an overkill’ usage of the model since the task at hand does not require the complexity. It is more advantageous to use NN models on NLP or computer vision task. However, what is surprising is the performance of XGBoost. One reason I believe the XGBoost consistently produced slightly inferior R-squared score against OLS is that I used a very limited search grid for hypermeter tuning due to time constraints. I believe, if I were to expand the search grid, XGBoost should obtain the second highest R-squared value. Nonetheless, the OLS performed very well is only giving up a percentage point in R-squared value. The r-square value means how much of the target feature is explainable by the features in the model. With the NN model, explainability of each feature is limited. An only localized explanation is available using partial dependency plot or Lime (local interpretable Model-Agnostic). Second place, XGBoost offers more in-depth explainability as feature importance is calculated, aggregated for a single tree via the amount that each feature split improves the Gini score. The most explainable model is OLS. The coefficient explanation is relatively straight forward for a standardized version of OLS. For example, for an increase of 1 standard deviation in the (2: total cooling square feet), we can expect a 0.2119 standard deviation increase in KWH. We know the mean values of each of these features, we can simply inverse calculate the actual value.

If the client has a large underlying portfolio (For example:1 Billion) and they don’t require explanation, then the choice of model is NN. One-percent improvement is very significant when the asset base is large.

If the client has a medium to large underlying asset and they require some level of explanation, then the model choice is XGBoost. We can point to the feature importance that is contributing to the output.

If the client has a regulatory constraint or small-medium size portfolio, then OLS is the model of choice. We can describe the impact of each feature in the model.

Data Engineering/ Deployment

Built modelling pipeline with classes in Python, have the central class for data processing. For example, the main matrix computation, data augmentations that are common to each of the model products. Other prereferral calculations. For example, getting the moving average or graphing with matplotlib can be placed into an Utils class and be called in the to the main modelling class when needed. The different client configuration file can be stored in a configuration class. In that file you would have the parameters for each client. For example, client 1 only have images at 256 X 256 resolution while client 2 at 128 X128. The saved model weights files can be called after the configuration file has been processed. This design allows the company to take on additional clients and to scale the products efficiently.
