# california-housing-price-prediction
Project Description: California Housing Price Prediction
This project aims to predict housing prices in California using the California Housing dataset. The dataset contains various features related to housing attributes, such as median income, house age, and average number of rooms, along with the target variable, which is the house price.

Key Steps in the Project:
1.Data Import and Preparation:

The project begins by importing necessary libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.
The California Housing dataset is loaded, and a DataFrame is created to facilitate data manipulation and analysis.
2.Exploratory Data Analysis (EDA):

The dataset is explored through summary statistics, data structure information, and visualizations.
A correlation heatmap is generated to identify relationships between features, and a pairplot is created to visualize the relationships among selected features.
3.Data Preprocessing:

The dataset is split into features (X) and the target variable (y).
The data is further divided into training and testing sets to evaluate the model's performance.
4.Model Training:

A Linear Regression model is initialized and trained using the training dataset.
The model's coefficients and intercept are displayed to understand the influence of each feature on the predicted price.
5.Model Evaluation:

Predictions are made on the test dataset, and evaluation metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score are calculated to assess the model's accuracy.
Visualization of Results:

A scatter plot is created to compare actual house prices against predicted prices, providing a visual representation of the model's performance.
Conclusion:
The project demonstrates the application of linear regression for predicting housing prices based on various features. The evaluation metrics indicate the model's effectiveness, and visualizations help in understanding the relationship between actual and predicted values. This project serves as a foundational example of predictive modeling in real estate analytics.


