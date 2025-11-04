## Project Overview
This project aims to compare four machine learning models for predicting spring onset using meteorological time series data. The source code will be organized into multiple modules and will include sample data files (still being organized).
## Dataset Description
## Code Module Description
### 1. DataProcess
- Loads and preprocesses data.
- Organize features according to the model's format requirements and split the data into training and validation sets.
- Output the model prediction results to the specified path in the designated file format.
### 2. ModelsTunRun
- Perform regression using four machine learning models with K-fold cross-validation.
- Optimizes hyperparameters using Hyperopt to find the best parameter combination.
- Trains the final model using the best parameters, computes evaluation metrics ($R^2$, RMSE, MAE).
- Saves the model, prediction results, and evaluation metrics to specified files, completing the data analysis and model construction process.
### 3. ResultPresent
- Create charts and draw images to facilitate analysis of the results.
