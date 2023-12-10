# Multi-Class and Multi-Label Classification with SVMs and K-Means Clustering

## Overview

This project focuses on applying Support Vector Machines (SVM) and K-Means Clustering to a multi-class and multi-label dataset: the Anuran Calls (MFCCs) Data Set. The project involves evaluating various machine learning techniques and metrics for classification and clustering.

## Part 1: Multi-class and Multi-Label Classification Using SVMs

### (a) Data Preparation

- Download the Anuran Calls (MFCCs) Data Set from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29).
- Randomly select 70% of the data as the training set.

### (b) Classification Approach

- The dataset contains three labels: Families, Genus, and Species, each with multiple classes.
- Implement a binary relevance approach by training a separate SVM for each label.

#### (i) Evaluation Metrics

- Research and apply exact match and hamming score/loss methods for multi-label classification evaluation.

#### (ii) Gaussian Kernel SVM

- Train a Gaussian kernel SVM for each label.
- Use 10-fold cross-validation to determine the SVM penalty weight and Gaussian kernel width.
- Experiment with both standardized and raw attributes.

#### (iii) L1-Penalized SVMs

- Train L1-penalized SVMs for each label.
- Standardize attributes and use 10-fold cross-validation to determine the SVM penalty weight.

#### (iv) Class Imbalance Treatment

- Apply SMOTE or other methods to address class imbalance.
- Evaluate and compare the performance of classifiers.

#### (v) Classifier Chain Method

- Apply the Classifier Chain method to the problem as an extra practice.

#### (vi) Additional Metrics

- Research and compute confusion matrices, precision, recall, ROC, and AUC for multi-label classification.

## Part 2: K-Means Clustering on Multi-Class and Multi-Label Data

### Monte-Carlo Simulation

- Perform the following procedures 50 times, and report the average and standard deviation of the Hamming Distances.

#### (a) Clustering

- Apply k-means clustering to the entire dataset.
- Determine the optimal number of clusters (k) using CH, Gap Statistics, scree plots, Silhouettes, or other methods.

#### (b) Majority Label Identification

- In each cluster, identify the majority label for family, genus, and species based on true labels.

#### (c) Hamming Distance Calculation

- For each cluster, calculate the average Hamming distance, Hamming score, and Hamming loss between true labels and labels assigned by the clusters.

## Objectives

- To implement and evaluate multi-class and multi-label classification using SVMs.
- To explore the effectiveness of various techniques in handling multi-label datasets.
- To apply k-means clustering to a multi-class and multi-label dataset and evaluate clustering performance using Hamming metrics.
