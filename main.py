import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Set style for better visualizations
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv("diabetes_dataset.csv")

# 1. Data Exploration
print("Dataset Overview:")
print("-" * 50)
print(df.info())
print("\nBasic Statistics:")
print("-" * 50)
print(df.describe())

# 2. Visualizations
def create_visualizations():
    # Distribution of diabetes diagnosis
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='diabetes_stage')
    plt.title('Distribution of Diabetes Stages')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('diabetes_distribution.png')
    plt.close()

    # Age distribution by diabetes diagnosis
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='diabetes_stage', y='age')
    plt.title('Age Distribution by Diabetes Stage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('age_distribution.png')
    plt.close()

    # Risk factors analysis
    risk_factors = ['bmi', 'glucose_fasting', 'hba1c']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, factor in enumerate(risk_factors):
        sns.boxplot(data=df, x='diagnosed_diabetes', y=factor, ax=axes[i])
        axes[i].set_title(f'{factor} by Diabetes Diagnosis')
    plt.tight_layout()
    plt.savefig('risk_factors.png')
    plt.close()

    # Correlation matrix
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

# 3. Feature Engineering
def prepare_features(df):
    # Create BMI categories
    df['bmi_category'] = pd.cut(df['bmi'], 
                               bins=[0, 18.5, 24.9, 29.9, 100],
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Create age groups
    df['age_group'] = pd.cut(df['age'],
                            bins=[0, 20, 40, 60, 80, 100],
                            labels=['0-20', '21-40', '41-60', '61-80', '80+'])
    
    # Combine medical history features
    df['medical_history_score'] = df['hypertension_history'] + \
                                 df['cardiovascular_history'] + \
                                 df['family_history_diabetes']
    
    return df

# 4. Model Training
def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print("-" * 50)
    print(feature_importance.head(10))
    
    return model, X_test, y_test, y_pred

def main():
    # Create visualizations
    create_visualizations()
    
    # Prepare features
    processed_df = prepare_features(df.copy())
    
    # Prepare data for modeling
    categorical_cols = ['gender', 'ethnicity', 'education_level', 'employment_status', 
                       'smoking_status', 'bmi_category', 'age_group']
    
    # Create dummy variables
    X = pd.get_dummies(processed_df.drop(['diagnosed_diabetes', 'diabetes_stage'], axis=1),
                      columns=categorical_cols)
    y = processed_df['diagnosed_diabetes']
    
    # Train and evaluate model
    model, X_test, y_test, y_pred = train_model(X, y)
    
    print("\nAnalysis completed! Visualization files have been saved.")

if __name__ == "__main__":
    main()