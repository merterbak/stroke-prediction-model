import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
#from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from catboost import CatBoostClassifier

# Read the train CSV file into a pandas dataframe
real_data = pd.read_csv('./healthcare-dataset-stroke-data.csv')
extra_data = pd.read_csv('./train.csv')
train_df = pd.concat([real_data, extra_data], ignore_index=True)
test_df = pd.read_csv('./test.csv')
test_id = test_df['id']

print(real_data.columns)
# # Compute z-scores for numerical columns
# z_scores = np.abs(stats.zscore(train_df[['age', 'avg_glucose_level', 'bmi']]))
#
# # Remove rows with z-score > 3
# train_df = train_df[(z_scores < 3).all(axis=1)]

train_df = train_df.dropna(subset=['bmi'])
train_df['smoking_status'].fillna('never smoked', inplace=True)
train_df.loc[train_df['gender'] == 'Other', 'gender'] = 'Male'
# Add additional features

train_df['age/bmi'] = train_df.age / train_df.bmi
train_df['age*bmi'] = train_df.age * train_df.bmi
train_df['bmi/prime'] = train_df.bmi / 25
train_df['obesity'] = train_df.avg_glucose_level * train_df.bmi / 1000
train_df['blood_heart']= train_df.hypertension * train_df.heart_disease

for col in train_df.select_dtypes(include='object'):
    unique_perc = train_df[col].nunique() / len(train_df) * 100
    print(f'{col}: {unique_perc:.2f}% unique values')

# Fill unknown category form smoking status as never smoked
test_df['smoking_status'].fillna('never smoked', inplace=True)

# Fill other class from gender as male
test_df.loc[test_df['gender'] == 'Other', 'gender'] = 'Male'

# Add additional features
test_df['age/bmi'] = test_df.age / test_df.bmi
test_df['age*bmi'] = test_df.age * test_df.bmi
test_df['bmi/prime'] = test_df.bmi / 25
test_df['obesity'] = test_df.avg_glucose_level * test_df.bmi / 1000
test_df['blood_heart']= test_df.hypertension * test_df.heart_disease

numeric_features = ['bmi','age','avg_glucose_level','age/bmi','age*bmi','bmi/prime','obesity','blood_heart'] # Select the numeric features

#Logistic Regression
X = train_df.drop(['id', 'stroke'], axis=1)
y = train_df['stroke']
X_train_before_oversample, X_val, y_train_before_oversample, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample to deal with imbalanced data
#oversampler = RandomOverSampler(random_state=42)
# Fit and transform the training data
#X_train, y_train = oversampler.fit_resample(X_train_before_oversample, y_train_before_oversample)

# Define the undersampling object
undersampler = RandomUnderSampler(random_state=42)
# Fit and transform the training data
X_train, y_train = undersampler.fit_resample(X_train_before_oversample, y_train_before_oversample)

# Define the column transformer for one-hot encoding the categorical features
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)], remainder='passthrough')

# Fit the column transformer on the training data and transform the data
X_train = transformer.fit_transform(X_train)
X_val = transformer.transform(X_val)
test_df = transformer.transform(test_df)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_df = scaler.transform(test_df)

# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

smote_enn = SMOTEENN(random_state=42)
X_train, y_train = smote_enn.fit_resample(X_train, y_train)

# Train a Random Forest Classifier, to select feature
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Get the feature importances
importances = rf_model.feature_importances_

# Get the feature names after the column transformer
feature_names = transformer.get_feature_names_out()

# Combine the feature names and their importances into a dataframe and sort by importance
importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importances_df = importances_df.sort_values(by='importance', ascending=False)

# Display the feature importances
print(importances_df)
# Select the top k features based on their importances
k = 13
selected_features = importances_df.head(k)['feature'].tolist()

feature_names = importances_df['feature']
importance_scores = importances_df['importance']

# Create a bar plot
plt.figure(figsize=(10, 6))  # Adjust the figure size as per your preference
plt.bar(range(len(feature_names)), importance_scores)
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.title('Feature Importance')
plt.xticks(range(len(feature_names)), feature_names, rotation='vertical', fontsize=8)  # Adjust the rotation angle and font size
plt.tight_layout()  # Ensure the labels are properly spaced
plt.show()

# Subset the training and validation data to include only the selected features
X_train_selected = X_train[:, importances_df.head(k).index]
X_val_selected = X_val[:, importances_df.head(k).index]
test_df_selected=test_df[:, importances_df.head(k).index]

print()# Logistic Regression Model 1

# Define the logistic regression model with L1 penalty
model = LogisticRegression(penalty='l2', C=0.1, solver='liblinear')
# Train the model on the selected features
model.fit(X_train_selected, y_train)

# Evaluate the model's performance on the selected features
y_pred = model.predict(X_val_selected)
y_pred_proba = model.predict_proba(X_val_selected)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred_proba)
pr_auc = average_precision_score(y_val, y_pred_proba)
y_pred = (y_pred_proba > 0.7).astype(int)
print("LL:\n")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred, average='weighted'))
print("Recall:", recall_score(y_val, y_pred))
print("F1-score:", f1_score(y_val, y_pred,average='weighted'))
print("ROC AUC score:", roc_auc)
print("PR AUC score:", pr_auc)

# Calculate the confusion matrix
cm = confusion_matrix(y_val, y_pred)

# Create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Add labels, title, and axis ticks
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.xticks([0, 1], ['No Stroke', 'Stroke'])
plt.yticks([0, 1], ['No Stroke', 'Stroke'])

# Display the plot
plt.show()

# y_pred = model.predict(test_df_selected)
# results_df = pd.DataFrame({'id': test_id, 'stroke': y_pred})
# results_df.to_csv('results_a.csv', index=False)

print()# XGBoost with optimal parameters
# GRIDSEARCH TO FIND OPTIMAL PARAMETERS
# # Define the hyperparameters to search over
# param_grid = {
#     'max_depth': [3, 4, 5, 6],
#     'learning_rate': [0.01, 0.1, 0.3],
#     'n_estimators': [50, 100, 200],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'reg_alpha': [0, 0.1, 0.5],
#     'reg_lambda': [0, 0.1, 1.0]
# }
#
# xgb_model = XGBClassifier(random_state=42)
#
# # Create a grid search object
# grid_search = GridSearchCV(xgb_model, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
#
# # Fit the grid search object to the data
# grid_search.fit(X_train_selected, y_train)
#
# # Print the best parameters and the corresponding score
# print("Best parameters: ", grid_search.best_params_)
# print("Best ROC AUC score: ", grid_search.best_score_)

# XBG
xgb_clf = xgb.XGBClassifier(objective='binary:logistic',
                            colsample_bytree=1.0,
                            learning_rate=0.3,
                            max_depth=6,
                            n_estimators=100,
                            subsample=1.0,
                            random_state=42,
                            reg_alpha=0.1,
                            reg_lambda=0.1)

# Fit the classifier to the training data
xgb_clf.fit(X_train_selected, y_train)

# Predict the probabilities of the positive class for the validation data
y_pred_proba_selected_xg = xgb_clf.predict_proba(X_val_selected)[:, 1]

# Find the threshold value with the highest F1-score on the validation data
f1_scores = []
for threshold in np.arange(0.1, 1.0, 0.1):
    y_pred_thresh = (y_pred_proba_selected_xg > threshold).astype(int)
    f1 = f1_score(y_val, y_pred_thresh)
    f1_scores.append(f1)
best_threshold = np.arange(0.1, 1.0, 0.1)[np.argmax(f1_scores)]

# Apply the threshold to the predicted probabilities to obtain binary predictions
y_pred_val_xg = (y_pred_proba_selected_xg > best_threshold).astype(int)

# Evaluate the performance of the model on the validation data
accuracy = accuracy_score(y_val, y_pred_val_xg)
precision = precision_score(y_val, y_pred_val_xg)
recall = recall_score(y_val, y_pred_val_xg)
f1 = f1_score(y_val, y_pred_val_xg)
roc_auc = roc_auc_score(y_val, y_pred_proba_selected_xg)
pr_auc = average_precision_score(y_val, y_pred_proba_selected_xg)
print("XGB:\n")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC AUC score:", roc_auc)
print("PR AUC score:", pr_auc)

# Test with actual test set
# Make predictions using the XGBoost model with optimized hyperparameters
y_test_pred_proba = xgb_clf.predict_proba(test_df_selected)[:, 1]
y_pred_test_xg = (y_test_pred_proba > best_threshold).astype(int)
#print(y_pred_test_xg)
# results_df = pd.DataFrame({'id': test_id, 'stroke': y_pred_test_xg})
# results_df.to_csv('results_xg_noOutliers.csv', index=False)

print()# LGBM
#lgb_clf = lgb.LGBMClassifier(random_state=42)
# Define the hyperparameter grid to search over
# param_grid = {
#     'learning_rate': [0.1, 0.3, 0.5],
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'reg_alpha': [0.0, 0.1, 0.5],
#     'reg_lambda': [0.0, 0.1, 0.5],
# }
# Perform a grid search with 5-fold cross-validation
# grid_search = GridSearchCV(estimator=lgb_clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
# grid_search.fit(X_train_selected, y_train)
# Print the best hyperparameters and the corresponding ROC AUC score
# print("Best parameters: ", grid_search.best_params_)
# print("Best ROC AUC score: ", grid_search.best_score_)

# LGBM
params = {'colsample_bytree': 0.8,
          'learning_rate': 0.3,
          'max_depth': 7,
          'n_estimators': 200,
          'reg_alpha': 0.0,
          'reg_lambda': 0.0,
          'subsample': 0.6}

# Train the LGBM classifier with the best hyperparameters
lgb_clf_best = lgb.LGBMClassifier(**params, random_state=42)
lgb_clf_best.fit(X_train_selected, y_train)

# Evaluate the performance on the training data
y_pred_lgb = lgb_clf_best.predict(X_val_selected)
y_pred_proba_lgb = lgb_clf_best.predict_proba(X_val_selected)[:, 1]
print("LGBM:\n")
# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_val, y_pred_lgb))
print("Precision:", precision_score(y_val, y_pred_lgb, zero_division=0))
print("Recall:", recall_score(y_val, y_pred_lgb))
print("F1-score:", f1_score(y_val, y_pred_lgb))

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_val, y_pred_proba_lgb)
print("ROC AUC score:", roc_auc)

# Calculate PR AUC score
pr_auc = average_precision_score(y_val, y_pred_proba_lgb)
print("PR AUC score:", pr_auc)
# Evaluate the performance on the test data
#y_pred_test = lgb_clf_best.predict(test_df_selected)
#y_pred_proba_test = lgb_clf_best.predict_proba(test_df_selected)[:, 1]
# Define the voting classifier
voting_model = VotingClassifier(
    estimators=[('logreg', model), ('xgb', xgb_clf), ('lgbm', lgb_clf_best)],
    voting='soft'  # Use soft voting for probabilistic outputs
)

# Fit the voting classifier on the training data
voting_model.fit(X_train_selected, y_train)

# Make predictions on the validation data
y_pred_ensemble = voting_model.predict(X_val_selected)
print("\nLL/LGM/XG (voting classifier):\n")
# Evaluate the ensemble model's performance
print("Accuracy:", accuracy_score(y_val, y_pred_ensemble))
print("Precision:", precision_score(y_val, y_pred_ensemble))
print("Recall:", recall_score(y_val, y_pred_ensemble))
print("F1-score:", f1_score(y_val, y_pred_ensemble))
# Calculate the confusion matrix
cm = confusion_matrix(y_val, y_pred_ensemble)

# Create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Add labels, title, and axis ticks
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.xticks([0, 1], ['No Stroke', 'Stroke'])
plt.yticks([0, 1], ['No Stroke', 'Stroke'])

# Display the plot
plt.show()
# Define the CatBoost Classifier
catboost_clf = CatBoostClassifier(iterations=100, learning_rate=0.1, random_state=42)

# Fit the model on the training data
catboost_clf.fit(X_train_selected, y_train)

# Predict on the validation data
y_pred_catboost = catboost_clf.predict(X_val_selected)
print("Catboost:\n")
# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_val, y_pred_catboost))
print("Precision:", precision_score(y_val, y_pred_catboost))
print("Recall:", recall_score(y_val, y_pred_catboost))
print("F1-score:", f1_score(y_val, y_pred_catboost))

print() # Display1
# # Display the first few rows of the dataframe
# print(train_df.head())
#
# # Display summary statistics for numerical variables
# print(train_df.describe())

# Display the distribution of numerical variables
# sns.displot(train_df, x='age')
# sns.displot(train_df, x='bmi')
# sns.displot(train_df, x='avg_glucose_level')

# Display the distribution of categorical variables
# sns.countplot(data=train_df, x='gender')
# sns.countplot(data=train_df, x='hypertension')
# sns.countplot(data=train_df, x='heart_disease')
# sns.countplot(data=train_df, x='ever_married')
# sns.countplot(data=train_df, x='work_type')
# sns.countplot(data=train_df, x='Residence_type')
# sns.countplot(data=train_df, x='smoking_status')

#Replace rows with missing BMI values with mean of the values
#mean_bmi = train_df['bmi'].mean()
#train_df['bmi'] = train_df['bmi'].fillna(mean_bmi)

# Drop rows with missing BMI values
print() # Display2
# sns.histplot(train_df, x='bmi', kde=True)
# plt.show()
# sns.histplot(train_df, x='avg_glucose_level', kde=True)
# plt.show()
# sns.histplot(train_df, x='work_type', kde=True)
# plt.show()
# sns.histplot(train_df, x='age', kde=True)
# plt.show()
# sns.histplot(train_df, x='hypertension', kde=True)
# plt.show()
# sns.histplot(train_df, x='heart_disease', kde=True)
# plt.show()
# sns.histplot(train_df, x='ever_married', kde=True)
# plt.show()
# sns.histplot(train_df, x='work_type', kde=True)
# plt.show()
# sns.histplot(train_df, x='Residence_type', kde=True)
# plt.show()
# sns.histplot(train_df, x='smoking_status', kde=True)
# plt.show()


# Plot the relationship between age and stroke

# CORRELATION BETWEEN STROKE AND CATEGORICAL FEATURES
# sns.histplot(data=train_df, x="age", hue="stroke", kde=True, multiple="stack")
# plt.show()
#
# # Plot the relationship between hypertension and stroke
# sns.countplot(data=train_df, x="hypertension", hue="stroke")
# plt.show()
#
# # Plot the relationship between heart disease and stroke
# sns.countplot(data=train_df, x="heart_disease", hue="stroke")
# plt.show()
#
# # Plot the relationship between smoking status and stroke
# sns.countplot(data=train_df, x="smoking_status", hue="stroke")
# plt.show()

#BMI RELATIONS VISUALITIONS

# Plot the distribution of BMI
# sns.histplot(train_df, x='bmi', kde=True)
#
# # Plot the relationship between BMI and age
# sns.scatterplot(data=train_df, x='age', y='bmi')
#
# # Plot the relationship between BMI and hypertension
# sns.catplot(data=train_df, x='hypertension', y='bmi')
# plt.show()
#
# # Plot the relationship between BMI and heart disease
# sns.boxplot(data=train_df, x='heart_disease', y='bmi')
#
# # Plot the relationship between BMI and smoking status
# sns.violinplot(data=train_df, x='smoking_status', y='bmi')
#
# # Plot the relationship between BMI and work type
# sns.boxplot(data=train_df, x='work_type', y='bmi')
#
# # Plot the relationship between BMI and residence type
# sns.boxplot(data=train_df, x='Residence_type', y='bmi')
#
# # Plot the relationship between BMI and stroke status
# sns.boxplot(data=train_df, x='stroke', y='bmi')
#
# # Display the correlation matrix of numerical variables
# corr_matrix = train_df[['age', 'bmi', 'avg_glucose_level', 'stroke']].corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
#
# # Display summary statistics for BMI by gender
# sns.catplot(data=train_df, x='gender', y='bmi', kind='box')
#
# # Display summary statistics for BMI by marital status
# sns.catplot(data=train_df, x='ever_married', y='bmi', kind='box')
#
# sns.catplot(data=train_df, x='stroke', y='bmi', kind='box')
#
# #BMI RELATIONS VISUALITIONS
print() # Display3
# Display the correlation matrix of numerical variables
# corr_matrix = train_df[['age', 'bmi', 'avg_glucose_level', 'stroke','hypertension','heart_disease']].corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Display the percentage of missing values for each variable
# missing_perc = train_df.isnull().mean() * 100
# print(missing_perc)

# Display the percentage of unique values for each categorical variable
# Fill unknown category form smoking status as never smoked



