import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# read the dataset
dataset = pd.read_excel("HousePricePrediction.xlsx")

# separate rows with and without known SalePrice
# store houses with known prices in known_price_data
known_price_data = dataset[dataset['SalePrice'].notnull()]  
# store houses with unknown prices in unknown_price_data
unknown_price_data = dataset[dataset['SalePrice'].isnull()] 

# store categorical data in object_cols
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)

# use OneHotEncoding on categorical columns
OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# For known data
OH_cols_known = pd.DataFrame(OH_encoder.fit_transform(known_price_data[object_cols]))
OH_cols_known.index = known_price_data.index
OH_cols_known.columns = OH_encoder.get_feature_names_out()
df_known = known_price_data.drop(object_cols, axis=1)
df_known = pd.concat([df_known, OH_cols_known], axis=1)

# For unknown data (same transformation)
OH_cols_unknown = pd.DataFrame(OH_encoder.transform(unknown_price_data[object_cols]))
OH_cols_unknown.index = unknown_price_data.index
OH_cols_unknown.columns = OH_encoder.get_feature_names_out()
df_unknown = unknown_price_data.drop(object_cols, axis=1)
df_unknown = pd.concat([df_unknown, OH_cols_unknown], axis=1)

# Handle missing values by filling with the median (for numerical columns)
df_known.fillna(df_known.median(), inplace=True)
df_unknown.fillna(df_unknown.median(), inplace=True)

# Split the known dataset into features (X) and target (Y)
X = df_known.drop(['SalePrice'], axis=1)
Y = df_known['SalePrice']

# Split the known data into training and validation sets
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Ensure that X_train, X_valid, and X_unknown have no missing values before training the models
X_train.fillna(X_train.median(), inplace=True)
X_valid.fillna(X_valid.median(), inplace=True)
X_unknown = df_unknown.drop(['SalePrice'], axis=1)
X_unknown.fillna(X_unknown.median(), inplace=True)

# Train different models
# SVM Model
model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
Y_pred_svr = model_SVR.predict(X_valid)
print("SVM MAPE:", mean_absolute_percentage_error(Y_valid, Y_pred_svr))

# Random Forest Regressor Model
model_RFR = RandomForestRegressor(n_estimators=100, random_state=0)
model_RFR.fit(X_train, Y_train)
Y_pred_rfr = model_RFR.predict(X_valid)
print("Random Forest MAPE:", mean_absolute_percentage_error(Y_valid, Y_pred_rfr))

# Linear Regression Model
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred_lr = model_LR.predict(X_valid)
print("Linear Regression MAPE:", mean_absolute_percentage_error(Y_valid, Y_pred_lr))

# Plot predicted vs actual SalePrice for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(Y_valid, Y_pred_lr, color='pink', alpha=0.5)  # Scatter points in pink
plt.plot([Y_valid.min(), Y_valid.max()], [Y_valid.min(), Y_valid.max()], color='#ff1493', lw=2)  
plt.title('Linear Regression: Predicted vs Actual SalePrice')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.grid(True)

# Predict missing SalePrices using the best model (for example, Random Forest)
predicted_prices = model_RFR.predict(X_unknown)

# Add predicted prices back into the unknown data using .loc to avoid SettingWithCopyWarning
unknown_price_data.loc[:, 'SalePrice'] = predicted_prices

# Combine known and unknown data back into one complete dataset
final_dataset = pd.concat([known_price_data, unknown_price_data])

# Save the complete dataset to a file (optional)
final_dataset.to_excel("Complete_HousePricePrediction.xlsx", index=False)

print("Predicted missing SalePrices and saved the complete dataset.")

# Find the most expensive house
most_expensive_house = final_dataset.loc[final_dataset['SalePrice'].idxmax()]

# Print the details of the most expensive house
print(most_expensive_house)

### Exploratory Data Analysis Plots ###
# Correlation heatmap for numerical features
numerical_dataset = dataset.select_dtypes(include=['number'])
plt.figure(figsize=(12,6))
sns.heatmap(numerical_dataset.corr(),
            cmap = 'PuRd',
            fmt = '.2f',
            linewidths = 2,
            annot = True)
plt.title('Correlation Heatmap for Numerical Features')


# Barplot showing unique values in categorical features
unique_values = [dataset[col].nunique() for col in object_cols]
plt.figure(figsize=(10,6))
plt.title('Unique Values in Categorical Features')
plt.xticks(rotation=90)
palette = sns.color_palette(["#ff1493", "#ff69b4", "#ff6eb4", "#ffc0cb"])
sns.barplot(x=object_cols, y=unique_values, palette=palette, dodge=False)


# Barplot distribution of each categorical feature
plt.figure(figsize=(18, 36))
plt.suptitle('Distribution of Categorical Features', fontsize=16, y=1)
index = 1

for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    plt.title(col, fontsize=12)
    palette = sns.blend_palette(["#ff69b4", "#ff1493"], n_colors=len(y))
    sns.barplot(x=y.index, y=y, palette=palette, dodge=False)
    plt.xlabel(col, fontsize=10)
    plt.ylabel('Count', fontsize=10)
    index += 1

plt.tight_layout()
plt.show()
