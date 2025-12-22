# Myresearch
A Comparative analysis of benchmark search based optimizer and feature selection for cross project defect prediction
Context:
Software Defect prediction (SDP) is use to predict defects in software components. Machine Learning techniques (ML) are extensively use to tackle this problem.
Objective:
The objective of software defect prediction (SDP) is to identify defect-prone modules. This is achieve by using datasets obtained by mining software historical repositories. However, data extracted from these repositories are often associated with high dimensionality, class imbalance, and mislabels which deteriorate classification performance and increase model complexity. One possible solution to eliminate those metrics is Feature Selection (FS) using filtering method. Therefore, our research objective to answer whether Filter e.g. Nearest-Neighbor (NN)-Filter can improve prediction accuracy of Software Prediction Model (SPM) through search based algorithm.
Method:
In this paper, FS techniques applied to evaluate the performance of the proposed approach upon 41 real-world software projects from Tera PROMISE repository. To assess the impact of feature sets, we use two sets of features, SCM+OO+LOC (all) and CK+LOC (ckloc) as well as iterative info-gain subsetting (IG) for feature selection.
Result:
Since we have multi-class problem we applied ANN-filter to remove outliers and balance the class. Overall, the performance of ANN-Filter is comparable to that of within project defect prediction (WPDP) benchmarks. In terms of multiple comparisons test, all variants of ANN belong to the top ranking group of approaches. Besides our results will be validate through k-fold validation.
