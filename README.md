# AgeEstimator

Creates a model that can be used to predict a persons age by looking at a their reddit posts. 

It is able to predict ages correctly within a margin of 10 years more than 80% of the time. 

The AgeEstimatorDataScraper script creates a database of reddit users, their age and their reddit comments. 
The AgeEstimatorModelCreation script trains two models (TDIDF and Word2vec) on the data collected in the AgeEstimatorDataScraper script.

 
