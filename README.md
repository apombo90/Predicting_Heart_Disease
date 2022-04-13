# Group-7-Project

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Project Overview**

As part of the final project of the Data Analytics Bootcamp at the University of Toronto, students have been assigned to groups in order to confirm their knowledge of the underlying concepts of data analytics and data science tools including their proficiency in Python, SQL, Pandas, and their ability to extract, transform, and load (ETL) data, and finally perform an analysis on the data to derive conclusions and provide a recommendation to potential employers. 

Group 7, has selected an open-source dataset from Kaggle that that contains data pertaining to a number of indicators of heart disease. The group will use the heart disease dataset to determine which indicators are factors for heart disease and further go on to test a machine learning model that can accurately predict which factors lead to heart disease. As part of the project, the heart dataset will be stored in a SQL database using PgAdmin, cleaned used the ETL process through Pandas, and tested through a machine learning model that will be developed by the group. Due to the complexity of this task, we will design and train a deep learning neural network that will evaluate all types of input data and produce a clear decision-making result, that will allow us to draw conclusions from our data.


## Goals:

Determine which factors are key indicators of heart disease based on the data provided.
Create a SQL database to store and load the dataset.
Clean the data to provide to allow for testing using machine learning.
Select and develop a machine learning model for testing.
Using the machine learning model, determine its ability to accurately predict heart disease.


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Resources**

-Data Source: `heart_disease_key_indicators.csv', 'binary_data.csv, 'heart_dummies.csv', 'heart_undersampled.csv', 'heart_clean.csv', 'non-binary_data.csv', 'HeartDisease_ERD.png'.

-Software: `Python 3.7.10`, `Visual Studio Code 1.38.1`, `Jupyter Notebook`, `Anaconda3`, `PgAdmin`. 

-Resources: https://www.kaggle.com/datasets?search=heart+disease.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Project Deliverables** 

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Segment 1 Deliverables

**Presentation**

**Approach**
We will be using the Agile approach to project management. Based on the fact that team members are not familiar with each other and come from different skill levels, we required a methodology that is be flexible, yet still provide clear goals and tasks. This seemed particularly relevant as each team member can harness their own strengths for deliverables and apply them to weekly tasks. 
Furthermore, this approach prevents members from being solely responsible for one task throughout and encourages a collaborative team. 

**Topic Selection**
We selected a dataset that analyses factors that lead to heart disease. We reviewed several other open-source datasets however, we selected this one based on interest, its relevancy, and the ability of the created machine learning model to be applicable to real world scenarios and provide a foundation from which similar datasets could be analysed.

A link to the dataset can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Resources/heart_disease_key_indicators.csv):

**Questions To Be Answered**

We hope to answer the following questions throughout the project:

1. What are the key indicators that lead to heart disease. 
2. What factors had the greatest influence on a person developing heart disease and in what proportions.
3. Can a machine learning model accurately predict whether a person will develop heart disease based on the data provided.

**GitHub**

A link to the main page of the Group 7 Final Project GitHub repository can be found [here](https://github.com/Rangisal/Group-7-Project):


[Branches](https://github.com/Rangisal/Group-7-Project/blob/mcbride_branch/Images/Segment_1_Image_1_Active_Branches.png) have been created for each member of the group.
![Segment_1_Image_1_Active_Branches](https://user-images.githubusercontent.com/92111396/159023203-74e01501-5741-4d6e-b2c6-e72b47893cda.png)

**Communication Protocols - Standard Operating Procedures (SOPs)**
1. Weekly meeting will be held on Tuesday and Thursday of each week (1900-2100).
2. All group discussion will be conduced on the “Group 7 Chat” on Slack.
3. Consult weekly task list.
4. Provide initial/update code as required.
5. After each major update to code or 1 hour of work, upload to GitHub under individual branch.
6. Once committed to individual branch, tag to a specific member for review. 
7. The reviewing member is to make changes/suggestions as necessary for the specified code to which they have been tagged.
8. The reviewing member will then upload code to their own branch and tag the original member.
9. Original member will review code and upload to main page.
10. Alert the team to all commits to main GitHub page.
11. All code uploaded to the main page should be reviewed by at least 1 other member.

A weekly tracker has also been created and will be uploaded to GitHub outlining each member’s tasks and the priority of work for the week. A link to the PowerPoint presentation and weekly task tracker can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Data%20Analytics%20Group%207%20Workflow%20Procedure.pptx): 


**Machine Learning Model**

A preview of the machine learning model code can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Logistic_Regression_Heart_Disease.ipynb):

![Logistic_Regression_Heart_Disease](https://user-images.githubusercontent.com/42978221/159191274-773cead5-839b-4f86-a682-ac65068c2521.png)


**Database**

A preview of the SQL Database code can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/HeartDisease_DB%20creation.sql):

![pgAdmin Heart_Disease_db creation](https://user-images.githubusercontent.com/42978221/159190443-a7a88b8c-921f-4dda-b5fa-1bded3193aea.png)


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Segment 2 Deliverables

**Presentation**

A link to the Google Slides presentation can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Project%20-%20Group%207.pptx).

The above presentation will outline:
1.	The selected topic.
2.	Reason why the topic was selected.
3.	Questions we hope to answer with the data.
4.	A description of the data exploration phase of the project.

**GitHub**

A link to the main page of the Group 7 Final Project GitHub repository can be found [here](https://github.com/Rangisal/Group-7-Project).

A link to the branches of the Group 7 Final Project GitHub repository can be found [here](https://github.com/Rangisal/Group-7-Project/branches).

**Machine Learning Model**

A link to the machine learning of the Group 7 Final Project GitHub repository can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Random%20Forest%20Classifier%20Model..ipynb).

The Logistic Regression model compares the actual outcome (y-test) from the test set against the model's predicted values (predictions). 
We obtained the accuracy score of the model, which is simply the percentage of predictions that are correct. In our case, the model's accuracy score is 0.916, meaning that the model was correct 91.6% of the time.

![Seg2 - Regression - Model validation Results](https://user-images.githubusercontent.com/42978221/161459329-e99ca650-0438-448e-88c9-38c3c6c3f7d4.png)

Additionally, we implemented a random forest algorithm into our analysis to rank the importance of input variables in our predictions.
It is clear that the most relevant features to impact decisions based on our model are SleepTime, PhysicalHealth and MentalHealth, followed by others.

![Seg2 - Randon Forest - Rank Important Features](https://user-images.githubusercontent.com/42978221/161459364-ce49f15b-710b-4b2e-a1af-ecedd6eb1abd.png)


**Database**

A link displaying the code in the database of the Group 7 Final Project GitHub repository can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/binary_nonbinary_tables.sql) where two tables were created, one with binary data and the second table only containing non-binary data. We were able to join both tables using an inner join on the **id** column, the join can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/tables_join.sql).

![image](https://user-images.githubusercontent.com/91766276/161567395-db098bfb-6429-4fd0-a77e-43695c2a18b2.png) ![image](https://user-images.githubusercontent.com/91766276/161568335-0967fd5d-024b-4974-8e41-e16e763810ff.png)



**Dashboard**

A link to the dashboard of the Group 7 Final Project GitHub repository can be found [here](https://public.tableau.com/views/HeartDiseasePrediction_16498175885680/HeartDiseasePrediction?:language=en-US&:display_count=n&:origin=viz_share_link).

Tableau Public was the tool used to create the dashboard. The dashboard tells the story of the data and presents it in a visually appealing way that aims to allow audiences, particularly physicians, and to a lesser extent the lay public as to what the main influencers are of heart disease and to what degree each factor can influence an individual’s ability to develop heart disease. The dashboard was comprised of elements that outline each factors relationship to the overall data. Visually, individual health concerns and factors such gender and smoking habits are delineated by colour and hovering over each parcel of data on a particular graph provides a specific insight for that datum.  


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Segment 3 Deliverables

**Presentation**
A link to the Google Slides presentation can be found [here](https://docs.google.com/presentation/d/1GtclwW8afOzY0mhaW8DeH-n36IszhfB8L265w8wBHSU/edit#slide=id.p1)

The above presentation will outline with more information than the segment 2:

1. The selected topic.
2. Reason why the topic was selected.
3. Description of source of data.
4. Questions we hope to answer with the data.
5. A description of the data exploration phase of the project.
   - As the data set was too big , broke it in to two data set as categorical and continuous by using dummies data set. To utilize different skill sets for the categorical variable correlation clculated heatmap and the continuous variables correlation have been calulated in excel. The links for the dataset and the correlation can be found here.
Dataset - [here](https://github.com/Rangisal/Group-7-Project/tree/main/Resources)
correlation - [here](https://github.com/Rangisal/Group-7-Project/blob/main/Correlation%20among%20the%20variables.ipynb)
   
7. A description of the analysis phase of the project.
8. Technologies , languages , tools and algorithms used throughout the project.

**Github**

A link to the main page of the Group 7 Final Project GitHub repository can be found [here](https://github.com/Rangisal/Group-7-Project).

A link to the branches of the Group 7 Final Project GitHub repository can be found [here](https://github.com/Rangisal/Group-7-Project/branches).


***Machine Learning Model**

A link to the machine learning model for logistic regression of the Group 7 Final Project GitHub repository can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Logistic_Regression_Heart_Disease.ipynb)

A link to the machine learning model for random forest model of the Group 7 Final Project GitHub repository can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Random%20Forest%20Classifier%20Model..ipynb) and [here](https://github.com/Rangisal/Group-7-Project/blob/main/Improving%20Random%20Forest%20Class.ipynb).

For Segment 3, we update the Logistical Regression Models to include both oversampling and undersampling algorithms as method of diversifying and testing for the best model. Previously, in Segment 2, the models we used for undersampling had an accuracy score of 89% but this was because the data was binary and skewed in a particular direction, which then caused the model to generalize that the majority of samples would fall into this category. This is evident in the confusion matrix (below) in which one category had a precision of 92% while the other category had a precision of 55%; similarly, all other metrics for this model reflected similar discrepancies.

![Segment_2_Undersmapling_Confusion_Matrix](https://user-images.githubusercontent.com/92111396/162645590-66fe9519-6ad9-4c57-8c99-8d03bb1ded33.png)

A link to the above image can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Images/Segment_2_Undersmapling_Confusion_Matrix.png).

In segment 3, we accounted for this skew in the data by normalizing the samples using the “resample” function. This was done initially for the undersampling algorithm, in which “n_samples” was set to the minority class. Similarly, this was done again for the oversampling algorithm, but in this case “n_samples” was set to the majority class.  

**Undersampling**

![Segment_3_Logistic_Regression_Resampling_Undersampling](https://user-images.githubusercontent.com/92111396/162645676-c4186fc3-9581-480d-96ae-f9bbda0aae34.png)

A link to the above image can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Images/Segment_3_Logistic_Regression_Resampling_Undersampling.png).

**Oversampling**

Segment_3_Logistic_Regression_Resampling_Oversampling.png
A link to the above image can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Images/Segment_3_Logistic_Regression_Resampling_Oversampling.png).

In addition to the above changes, we also added an additional algorithm to the Random Forest Model for Segment 3. Whereas the original Random Forest Model only tested using an undersampling algorithm, for this segment we added a new algorithm to search with cross validation and search for hyperparameters. This method allowed us to test multiple variables within the data to determine which model achieved the best performance on the dataset.  

**Random Forest Hyperparameter Optimization**

![Segment_3_Random_Forest_Hyperparameter_Optimization_Accuracy_Score_Confusion_Matrix](https://user-images.githubusercontent.com/92111396/162645981-2c9a90c4-dc0d-4694-9234-65811cd5be0d.png)

A link to the above image can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Images/Segment_3_Random_Forest_Hyperparameter_Optimization_Accuracy_Score_Confusion_Matrix.png).

**Accuracy Scores**

The accuracy scores for each model are listed here:

- Logistic Regression: 76.4%

- Random Forest Classifier:  76.5%

- XGBoost: 75.8%

**Confusion Matrices**

Images of the confusion matrix for each model are listed here:

**Logistic Regression** 

![Linear_Regression_Undersampling_Confusion_Matrix](https://user-images.githubusercontent.com/92111396/162646417-1e22047a-d8c4-4da5-a064-46cdc8c137a1.png)

A link to the above image can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Images/Linear_Regression_Undersampling_Confusion_Matrix.png).

**Linear Regression Oversampling**

![Linear_Regression_Oversampling_Confusion_Matrix](https://user-images.githubusercontent.com/92111396/162646412-d44020d7-a15b-46be-a765-bb55ed74963d.png)

A link to the above image can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Images/Linear_Regression_Oversampling_Confusion_Matrix.png).

**Random Forest Undersampling**

![Random_Forest_Undersampling_Confusion_Matrix](https://user-images.githubusercontent.com/92111396/162646439-9e0b44de-dfa7-4cd0-b972-643db7163629.png)

A link to the above image can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Images/Random_Forest_Undersampling_Confusion_Matrix.png).

**Random Forest Hyperparameter Optimization**

![Random_Forest_Hyperparameter_Optimization_Confusion_Matrix](https://user-images.githubusercontent.com/92111396/162646447-2d65292a-0cf3-4268-ad2c-b7e67befb640.png)

A link to the above image can be found [here](https://github.com/Rangisal/Group-7-Project/blob/main/Images/Random_Forest_Hyperparameter_Optimization_Confusion_Matrix.png).

**XGBoost**

HEREE

**Dashboard**

A link to the dashboard of the Group 7 Final Project GitHub repository can be found [here](https://public.tableau.com/app/profile/ethan.mcbride/viz/HeartDiseaseKeyIndicatorsDashboard/GeneralOverview).

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Segment 4 Deliverables
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Purpose**

The purpose of this assignment was to create a machine learning model that could be used to predict heart disease within individuals, based on seventeen key lifestyle factors including an individual’s body mass index (BMI), physical activity status, age, sex, and various health factors ranging from their smoking status to their sleep time. The machine learning models that were identified to perform this task were a Linear Regression Model and Random Forest Classifier Model; data was oversampled and undersampled within the linear regression model using the “resample” algorithm, while the newly extracted over and undersampled data was used in the Random Forest Classifier model. 


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


A link to images of the results can be found [here](https://github.com/Rangisal/Group-7-Project/tree/main/Images/Results).


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Summary**

**Data Results**

Based on the above accuracy scores, we can see that Linear Regression machine learning model had the highest rate of accuracy with the ability to predict the correct values 76% of the time. Taken individually, the resampling models had similar accuracy scores falling between 75% and 77%, with the both the oversampling and the undersampling techniques receiving the same accuracy score of 76%. Before normalization of the data, the Random Forest Classifier Model received an accuracy score of only 67% which falls ten percent short of the linear regression model, however with the data being scaled (normalized), this improved the accuracy to 73%. This score was further enhanced by implementing a hyperparameter optimization model, which raised the accuracy score for the Radom Forest model to 76%. The XGBoost model had the lowest accuracy score of 75.85; this difference however is not statistically significant, and all models can be considered to have similar accuracy scores.

The precision scores for the two machine learning models effectively yielded the same percentages as compared to the accuracy scores. This means that machine learning models can be relied upon to likely predict a positive classification 76%, 77%, and 76% of the time respectively for logistical regression over and undersampling models, the Random Forest Classifier Undersampling, and the XGBoost model. However, the precision score alone can tell us very little, and it must be coupled with the sensitivity of the score. The sensitivity scores effectively tell us how reliable in our prediction our tests are, that is to say, how fine-tuned or the probability of a positive test, conditioned on truly having the condition. Based on the above scores, it is evident that all models are equally tuned to correctly predict heart disease risk potential, each receiving the exact same sensitivity score of 76%.

The F1 scores of each model effectively tell us is there is a pronounced imbalance between sensitivity and precision; a pronounced imbalanced will yield a low F1 score. Based on this, we can again see that once again, each model is equally capable and effectively balances sensitivity and precision to the same extent as each model received an F1 score of 76%, thereby demonstrating the equal disparity between sensitivity and precision.


**Recommendation**

Based on the results and the subsequent analysis of the data, it is the recommendation that the Random Forest machine learning model, be adopted for use in predicting heart disease risk. While each machine learning model tested had similar scores for each category, the Random Forest model, slightly edged out the other models. It consistently had the highest scores, albeit slightly, particularly in accuracy and precision, thus correctly made the correct predictions compared to the other models. Overall, all models were comparable and could be used to effectively predict the risk of an individual having heart disease.
