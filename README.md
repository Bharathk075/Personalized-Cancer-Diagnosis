# Personalized-Cancer-Diagnosis

* This project is part of a Kaggle Competition hosted by Memorial Sloan Kettering Cancer Center (MSKCC). 
 
#### Problem statement : 
* Classify the given genetic variations/mutations based on evidence from text-based clinical literature.

#### Detailed Description:

* When we break/mutate a gene, it has multiple variants.
###### Work-Flow:
1. A Molecualr Pathologist selects a list of genetic variations of interest that he/she want to analyze
2. The molecular pathologist searches for evidence in the medical literature that somehow are relevant to genetic variations of interest
3. Finally this molecular pathologist spends a huge amount of time analyzing the evidence related to each of the the variations to classify them
4. Our goal is to replace step-3 by a Machine Learning Model

#### Data Overview
Source: https://www.kaggle.com/c/msk-redefining-cancer-treatment/data
* We have two data files: one conatins the information about the genetic mutations and the other contains the clinical evidence (text) that human experts/pathologists use to classify the genetic mutations.
* Both these data files are have a common column called ID
###### Data File Information: 
* training_variants (ID , Gene, Variations, Class)
* training_text (ID, Text)

#### Real-world/Business objectives and constraints.
* No low-latency requirement.
* Interpretability is important.
* Errors can be very costly.
* Probability of a data-point belonging to each class is needed.
##### There are nine different classes a genetic mutation can be classified into => Multi class classification problem

###### Metrics:
* Multi class log-loss
* Confusion matrix
###### Objective: Predict the probability of each data-point belonging to each of the nine classes.

###### Constraints:

* Interpretability 
* Class probabilities are needed.
* Penalize the errors in class probabilites => Metric is Log-loss.
* No Latency constraints

#### FeatureEngineering
From the Given Features, we have created these features
* Frequency of Gene
* Frequency of Variation
* No of words in Text

###### Data Distribution among the 9 Classes
![Data Distb in PCD](https://user-images.githubusercontent.com/42597977/139613640-8a8ccc7d-f07f-4b80-a7d9-6a39c7d53d43.png)


### Modelling:

We have implemeted the Following Models
* Random Probability Model
* Naive Bayes
* K-Nearest Neighbours
* Logistic Regression
* Linear SVM
* Random Forests
* Stacking Model
* Logistic Regression with Feature Engineering

##### Results:

| Mdoel | Log-loss |
| --- | --- |
| Random Probability Model | 2.570 |
| Naive Bayes | 1.265 |
| K-Nearest Neighbours | 1.055 |
| Logistic Regression | 1.008 |
| Linear SVM | 1.101 |
| Random Forest | 1.11 |
| Naive Bayes | 1.265 |
| Stacking | 1.121 |
| Logistic Regression with Feature Engineering | 0.98 |
