predict_sales.py
Inaya Siddiqui
10/14/2024

This program predicts missing housing prices from a dataset using machine
learning methods. It reads data from an input file, processes categorical and numerical data, 
and applies data transformations to conduct analysis. The program trains and evaluates
models, including SVM, Random Forest, and Linear Regression, and selects the best model 
based on performance metrics. After predicting the missing sales prices, results are output to 
a new data file with a complete list of housing prices. Additionally, the program produces exploratory
data analysis visualizations inlcuding a linear regression plot, a heatmap to display correlations between variables, a bar plot 
displaying the quantity of types within each categorical value, and a bar plot displaying the distribution
of values across each categorical feature.

To run this program use the command
python predict.py

Input: HousePricePrediction.xlsx

Output: Complete_HousePricePrediction.xlsx

Sources:
Geekforgeeks.org
scikit-learn.org
ibm.com
w3schools.com
