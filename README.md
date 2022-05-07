# Predicting Heart Disease

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Project Overview**

Selected an open-source dataset from Kaggle that that contains data pertaining to a number of indicators of heart disease. We will use the heart disease dataset to determine which indicators are factors for heart disease and further go on to test a machine learning model that can accurately predict which factors lead to heart disease. As part of the project, the heart dataset will be stored in a SQL database using PgAdmin, cleaned used the ETL process through Pandas, and tested through a machine learning model. Due to the complexity of this task, we will design and train a deep learning neural network that will evaluate all types of input data and produce a clear decision-making result, that will allow us to draw conclusions from our data.


## Goals:

- Determine which factors are key indicators of heart disease based on the data provided.
- Create a SQL database to store and load the dataset.
- Clean the data to provide to allow for testing using machine learning.
- Select and develop a machine learning model for testing.
- Using the machine learning model, determine its ability to accurately predict heart disease.


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Resources**

- Data Source: `heart_disease_key_indicators.csv', 'binary_data.csv, 'heart_dummies.csv', 'heart_undersampled.csv', 'heart_clean.csv', 'non-binary_data.csv', 'HeartDisease_ERD.png'.

- Software: `Python 3.7.10`, `Visual Studio Code 1.38.1`, `Jupyter Notebook`, `Anaconda3`, `PgAdmin`. 

- Resources: https://www.kaggle.com/datasets?search=heart+disease, https://scikit-learn.org/stable/modules/cross_validation.html, https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d#:~:text=What%20is%20XGBoost%3F,all%20other%20algorithms%20or%20frameworks, https://help.tableau.com/current/pro/desktop/en-us/formatting_fonts_beta.htm.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Project Phases** 

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Machine Learning**

We update the Logistical Regression Models to include both oversampling and undersampling algorithms as method of diversifying and testing for the best model. Previously, in Segment 2, the models we used for undersampling had an accuracy score of 89% but this was because the data was binary and skewed in a particular direction, which then caused the model to generalize that the majority of samples would fall into this category. This is evident in the confusion matrix (below) in which one category had a precision of 92% while the other category had a precision of 55%; similarly, all other metrics for this model reflected similar discrepancies.

![Segment_2_Undersmapling_Confusion_Matrix](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Segment_2_Undersmapling_Confusion_Matrix.png)

A link to the above image can be found [here](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Segment_2_Undersmapling_Confusion_Matrix.png).

We accounted for this skew in the data by normalizing the samples using the “resample” function. This was done initially for the undersampling algorithm, in which “n_samples” was set to the minority class. Similarly, this was done again for the oversampling algorithm, but in this case “n_samples” was set to the majority class.  

- **Undersampling**

![Segment_3_Logistic_Regression_Resampling_Undersampling](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Segment_3_Logistic_Regression_Resampling_Undersampling.png)

A link to the above image can be found [here](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Segment_3_Logistic_Regression_Resampling_Undersampling.png).

- **Oversampling**

Segment_3_Logistic_Regression_Resampling_Oversampling.png
A link to the above image can be found [here](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Segment_3_Logistic_Regression_Resampling_Oversampling.png).

In addition to the above changes, we also added an additional algorithm to the Random Forest Model for Segment 3. Whereas the original Random Forest Model only tested using an undersampling algorithm, for this segment we added a new algorithm to search with cross validation and search for hyperparameters. This method allowed us to test multiple variables within the data to determine which model achieved the best performance on the dataset.  

- **Random Forest Hyperparameter Optimization**

![Segment_3_Random_Forest_Hyperparameter_Optimization_Accuracy_Score_Confusion_Matrix](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Segment_3_Random_Forest_Hyperparameter_Optimization_Accuracy_Score_Confusion_Matrix.png)

A link to the above image can be found [here](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Segment_3_Random_Forest_Hyperparameter_Optimization_Accuracy_Score_Confusion_Matrix.png).

- **Accuracy Scores**

The accuracy scores for each model are listed here:

- Logistic Regression: 76.4%

- Random Forest Classifier:  76.5%

- XGBoost: 75.8%

- **Confusion Matrices**

Images of the confusion matrix for each model are listed here:

- **Logistic Regression** 

![Linear_Regression_Undersampling_Confusion_Matrix](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Linear_Regression_Undersampling_Confusion_Matrix.png)

A link to the above image can be found [here](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Linear_Regression_Undersampling_Confusion_Matrix.png).

- **Logistic Regression Oversampling**

![Linear_Regression_Oversampling_Confusion_Matrix](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Linear_Regression_Oversampling_Confusion_Matrix.png)

A link to the above image can be found [here](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Linear_Regression_Oversampling_Confusion_Matrix.png).

- **Random Forest Undersampling**

![Random_Forest_Undersampling_Confusion_Matrix](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Random_Forest_Undersampling_Confusion_Matrix.png)

A link to the above image can be found [here](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Random_Forest_Undersampling_Confusion_Matrix.png).

- **Random Forest Hyperparameter Optimization**

![Random_Forest_Hyperparameter_Optimization_Confusion_Matrix](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Random_Forest_Hyperparameter_Optimization_Confusion_Matrix.png)

A link to the above image can be found [here](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/Random_Forest_Hyperparameter_Optimization_Confusion_Matrix.png).

- **XGBoost**

![image](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/Images/XgBoost.png)

**Database**

A link displaying the code in the database of the GitHub repository can be found [here](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/binary_nonbinary_tables.sql) where two tables were created, one with binary data and the second table only containing non-binary data. We were able to join both tables using an inner join on the **id** column, the join can be found [here](https://github.com/apombo90/Predicting_Heart_Disease/blob/main/tables_join.sql).

![image](https://user-images.githubusercontent.com/91766276/161567395-db098bfb-6429-4fd0-a77e-43695c2a18b2.png) ![image](https://user-images.githubusercontent.com/91766276/161568335-0967fd5d-024b-4974-8e41-e16e763810ff.png)


**Dashboard**

A link to the dashboard of the GitHub repository can be found [here](https://public.tableau.com/views/HeartDiseasePrediction_16498175885680/HeartDiseasePrediction?:language=en-US&:display_count=n&:origin=viz_share_link).

Tableau Public was the tool used to create the dashboard. The dashboard tells the story of the data and presents it in a visually appealing way that aims to allow audiences, particularly physicians, and to a lesser extent the lay public as to what the main influencers are of heart disease and to what degree each factor can influence an individual’s ability to develop heart disease. The dashboard was comprised of elements that outline each factors relationship to the overall data. Visually, individual health concerns and factors such gender and smoking habits are delineated by colour and hovering over each parcel of data on a particular graph provides a specific insight for that datum.  


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Results**

**Definitions and Calculations of Scores**

Accuracy: the difference between its predicted values and actual values.

Precision: Precision = TP/(TP + FP) Precision is a measure of how reliable a positive classification is.

Sensitivity = TP/(TP + FN) Sensitivity is a measure of the probability of a positive test, conditioned on truly having the condition.

F1 = 2(Precision * Sensitivity)/(Precision + Sensitivity) A pronounced imbalance between sensitivity and precision will yield a low F1 score.


**Accuracy Score**

- Logistic Regression: 76.4%

- Random Forest Classifier:  76.5%

- XGBoost: 75.8%


**Precision Score**

- Logistic Regression: 76%

- Random Forest Classifier:  77%

- XGBoost: 76%

**Recall (Sensitivity) Score**

- Logistic Regression: 76%

- Random Forest Classifier:  76%

- XGBoost: 76%


**F1 Score** 

- Logistic Regression: 76%

- Random Forest Classifier:  76%

- XGBoost: 76%


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Summary**

**Data Results**

Based on the above accuracy scores, we can see that Logistic Regression machine learning model had the highest rate of accuracy with the ability to predict the correct values 76% of the time. Taken individually, the resampling models had similar accuracy scores falling between 75% and 77%, with the both the oversampling and the undersampling techniques receiving the same accuracy score of 76%. Before normalization of the data, the Random Forest Classifier Model received an accuracy score of only 67% which falls ten percent short of the logistic regression model, however with the data being scaled (normalized), this improved the accuracy to 73%. This score was further enhanced by implementing a hyperparameter optimization model, which raised the accuracy score for the Radom Forest model to 76%. The XGBoost model had the lowest accuracy score of 75.8%; this difference however is not statistically significant, and all models can be considered to have similar accuracy scores.

The precision scores for the two machine learning models effectively yielded the same percentages as compared to the accuracy scores. This means that machine learning models can be relied upon to likely predict a positive classification 76%, 77%, and 76% of the time respectively for logistical regression over and undersampling models, the Random Forest Classifier Undersampling, and the XGBoost model. However, the precision score alone can tell us very little, and it must be coupled with the sensitivity of the score. The sensitivity scores effectively tell us how reliable in our prediction our tests are, that is to say, how fine-tuned or the probability of a positive test, conditioned on truly having the condition. Based on the above scores, it is evident that all models are equally tuned to correctly predict heart disease risk potential, each receiving the exact same sensitivity score of 76%.

The F1 scores of each model effectively tell us is there is a pronounced imbalance between sensitivity and precision; a pronounced imbalanced will yield a low F1 score. Based on this, we can again see that once again, each model is equally capable and effectively balances sensitivity and precision to the same extent as each model received an F1 score of 76%, thereby demonstrating the equal disparity between sensitivity and precision.


**Recommendation**

Based on the results and the subsequent analysis of the data, it is the recommendation that the Random Forest machine learning model, be adopted for use in predicting heart disease risk. While each machine learning model tested had similar scores for each category, the Random Forest model, slightly edged out the other models. It consistently had the highest scores, albeit slightly, particularly in accuracy and precision, thus correctly made the correct predictions compared to the other models. Overall, all models were comparable and could be used to effectively predict the risk of an individual having heart disease.
