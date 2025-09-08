# credit_card_customer_segmentaton_using_unsupervised_learning

## Introduction
This project present a complete clustering workflow to uncover meaningful customer segments from credit card usage data. The target is identify different groups of customers with similar spending behaviour, so that company can define a marketing strategy for the right customer groups.

## Repository Structure
credit_card_customer_segmentaton_using_unsupervised_learning/

│── data/: contain the dataset files used for the project.

│── notebooks/: contain Jupyter notebooks for EDA, data preprocessing, and model building steps.

│── models/: contain the saved trained model

│── src/: contain any additional source code used in the project.


## Project Overview

1. Data Exploration
We begin by examining the structure and distribution of the dataset:
Viewing column summaries and data types
Identifying key behavioral features
Visualizing distributions and relationships

2. Data Cleaning
To ensure data quality and consistency:
Handle missing values and duplicates
Check for outliers and skewed distributions
Prepare the data for modeling

3. Pre-Clustering Analysis
Before applying clustering:
Scale numerical features for fair comparison
Optionally apply PCA for dimensionality reduction

4. Modeling
I apply unsupervised clustering algorithm  K-Means, DBSCAN, Hierarchical Clustering...to segment customers based on behavior
Choosing the optimal number of clusters
Assigning cluster labels
Visualizing cluster separation

5. Post-Clustering Analysis
To interpret and validate the clusters:
Visualize feature distributions per cluster
Analyze demographic and behavioral patterns
Build charts and summaries to support storytelling

# Inference

# Customer Profiling

Cluster 0: Smallest Spenders and Lowest Credit Limit - this is the group with the lowest credit limit but they don't appear to buy much. Unfortunately this appears to be the largest group of customers.

Cluster 1: Medium Spenders with third highest Payments - the second highest Purchases group (after the Big Spenders).

Cluster 2: Big Spenders with large Payments - they make expensive purchases and have a credit limit that is between average and high. This is only a small group of customers.

Cluster 3: Cash Advances with Small Payments - this group likes taking cash advances, but make only small payments.

Cluster 4: Small Spenders and Low Credit Limit - they have the smallest Balances after the Smallest Spenders, their Credit Limit is in the bottom 3 groups

Cluster 5: Cash Advances with large Payments but Highest Credit Limit and Frugal - this group takes the most cash advances. They make large payments, but this appears to be a small group of customers. this group doesn't make a lot of purchases. It looks like the 3rd largest group of customers.

### Dataset
Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioral variables.

CUST_ID : Identification of Credit Card holder (Categorical)
BALANCE : Balance amount left in their account to make purchases (
BALANCE_FREQUENCY : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
PURCHASES : Amount of purchases made from account
ONEOFF_PURCHASES : Maximum purchase amount done in one-go
INSTALLMENTS_PURCHASES : Amount of purchase done in installment
CASH_ADVANCE : Cash in advance given by the user
PURCHASES_FREQUENCY : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
ONEOFFPURCHASESFREQUENCY : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
PURCHASESINSTALLMENTSFREQUENCY : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
CASHADVANCEFREQUENCY : How frequently the cash in advance being paid
CASHADVANCETRX : Number of Transactions made with "Cash in Advanced"
PURCHASES_TRX : Numbe of purchase transactions made
CREDIT_LIMIT : Limit of Credit Card for user
PAYMENTS : Amount of Payment done by user
MINIMUM_PAYMENTS : Minimum amount of payments made by user
PRCFULLPAYMENT : Percent of full payment paid by user
TENURE : Tenure of credit card service for user

	