import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set display options for Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.3f}'.format)

# Load the dataset
shop = pd.read_csv("online_shoppers_intention.csv")

# Function to explore and summarize the DataFrame
def explore_dataframe(dataframe, head=5):
    """Provide a general overview of a DataFrame."""
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe())

# Explore the loaded DataFrame
explore_dataframe(shop, head=2)

# Visualizing the distribution of stress levels
sns.set(style="whitegrid")
palette = sns.color_palette("Blues", n_colors=len(shop['stress level'].unique()))
sns.countplot(x='stress level', data=shop, palette=palette)
plt.title('Stress Level Distribution')
plt.show()

# Visualizing relationship between features and stress levels
for feature in shop.columns[:-1]:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=shop, x=feature, y='stress level', hue='stress level', palette='rocket')
    plt.title(f"{feature} vs. Stress Level")
    plt.xlabel(feature)
    plt.ylabel("Stress Level")
    plt.show()

# Visualizing the distribution of features by stress level
for feature in shop.columns[:-1]:
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=shop, x='stress level', y=feature, palette=palette)
    plt.title(f"{feature} Distribution by Stress Level")
    plt.xlabel("Stress Level")
    plt.ylabel(feature)
    plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = shop.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title("Correlation Matrix Heatmap")
plt.show()

# Preparing data for modeling
X = shop.drop(['stress level'], axis=1)
y = shop['stress level']

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying Power Transformation
power_transformer = PowerTransformer(method='yeo-johnson')
X_train_transformed = power_transformer.fit_transform(X_train_scaled)
X_test_transformed = power_transformer.transform(X_test_scaled)

# Feature importance with RandomForest
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_

# Printing feature importances
print("Feature Importances:")
for i, feature in enumerate(X_train.columns):
    print(f"{i + 1}) {feature:30} {importances[i]:.4f}")

# Visualizing feature importances
plt.figure(figsize=(10, 6))
colors = [sns.color_palette("Greens", as_cmap=True)(x) for x in importances/max(importances)]
plt.bar(X_train.columns, importances, align='center', color=colors)
plt.xticks(rotation=90)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

# Modelling and Evaluation
def evaluate_classifier(clf, clf_name):
    """Evaluate classifier performance."""
    clf.fit(X_train_transformed, y_train)
    y_pred = clf.predict(X_test_transformed)
    score = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"{clf_name} Accuracy: {score}\n")
    print(f"{clf_name} Confusion Matrix:\n{matrix}\n")
    print(f"{clf_name} Classification Report:\n{report}")
    print("-" * 60)

# Evaluating classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=13),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}
for name, clf in classifiers.items():
    evaluate_classifier(clf, name)

# Prediction on New Data
new_data = pd.DataFrame([[90.0, 23.0, 92.0, 15.0, 90.0, 95.0, 2.0, 70.0]], columns=X.columns)
log_reg = LogisticRegression(max_iter=1000, C=0.1)
log_reg.fit(X_train_transformed, y_train)
predicted_stress_level = log_reg.predict(new_data)
predicted_probabilities = log_reg.predict_proba(new_data)[0]
stress_level_labels = ["Low/Normal", "Medium Low", "Medium", "Medium High", "High"]

# Plotting predicted probabilities
plt.figure(figsize=(10, 6))
plt.bar(stress_level_labels, predicted_probabilities, color='skyblue')
plt.xlabel('Stress Levels')
plt.ylabel('Probability')
plt.title('Predicted Stress Level Probabilities')
predicted_label = stress_level_labels[predicted_stress_level[0]]
plt.annotate(f'Predicted Stress Level:\n{predicted_label}',
             xy=(predicted_label, predicted_probabilities[predicted_stress_level[0]]),
             xytext=(predicted_label, predicted_probabilities[predicted_stress_level[0]] + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center')
plt.tight_layout()
plt.show()
