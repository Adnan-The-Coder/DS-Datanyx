import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample dataset provided (usually, this would be read from a CSV file or other sources)
data = {
    'Animal Type': ['Giraffe', 'Giraffe', 'Panda', 'Lion', 'Lion', 'Tiger', 'Panda', 'Lion', 'Tiger', 'Elephant', 'Elephant', 'Tiger', 'Panda', 'Elephant', 'Lion', 'Lion', 'Elephant'],
    'Age (Years)': [28, 8, 41, 39, 1, 13, 28, 49, 25, 33, 38, 6, 45, 45, 21, 16, 47],
    'Health Condition': ['Arthritis', 'Healthy', 'Digestive Issues', 'Obesity', 'Healthy', 'Respiratory Problems', 'Healthy', 'Heart Disease', 'Joint Issues', 'Skin Issues', 'Obesity', 'Skin Infection', 'Healthy', 'Arthritis', 'Heart Disease', 'Skin Infection', 'Obesity'],
    'Symptoms': ['Stiffness, Pain', 'None', 'Nausea, Vomiting', 'Weight Gain', 'None', 'Breathing Issues', 'None', 'Chest Pain', 'Joint Pain', 'Rashes, Itching', 'Weight Gain', 'Swelling', 'None', 'None', 'Chest Pain', 'Rashes, Fever', 'Weight Loss'],
    'Treatment Type': ['Anti-inflammatory', 'None', 'Dietary Change', 'Diet Regulation', 'None', 'Medication', 'None', 'Cardiac Care', 'Physical Therapy', 'Topical Treatment', 'Diet Regulation', 'Topical Treatment', 'None', 'Pain Management', 'Cardiac Care', 'Antibiotics', 'Diet Regulation']
}

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Define expected lifespans for the species (in years)
lifespan = {
    'Giraffe': 25,  # Approximate average lifespan in the wild
    'Panda': 20,
    'Lion': 15,
    'Tiger': 15,
    'Elephant': 60
}

# Function to check if the animal's age is within a valid range
def check_age_validity(row):
    expected_lifespan = lifespan.get(row['Animal Type'], 20)  # Default to 20 if unknown
    return row['Age (Years)'] <= expected_lifespan * 1.5

# Apply the age validity check
df['Age Valid'] = df.apply(check_age_validity, axis=1)

# Function to check if the treatment type is correct based on the health condition
def check_treatment_validity(row):
    treatment_conditions = {
        'Arthritis': ['Anti-inflammatory', 'Pain Management'],
        'Digestive Issues': ['Dietary Change'],
        'Obesity': ['Diet Regulation'],
        'Respiratory Problems': ['Medication'],
        'Skin Issues': ['Topical Treatment'],
        'Skin Infection': ['Antibiotics', 'Topical Treatment'],
        'Heart Disease': ['Cardiac Care'],
        'Joint Issues': ['Physical Therapy']
    }
    return row['Treatment Type'] in treatment_conditions.get(row['Health Condition'], [])

# Apply the treatment validity check
df['Incorrect Treatment'] = df.apply(check_treatment_validity, axis=1)

# Create the feature 'Immediate Care Required' based on invalid age or treatment
df['Immediate Care Required'] = ~df['Age Valid'] | ~df['Incorrect Treatment']

# Preprocessing the features
X = df[['Age (Years)', 'Animal Type', 'Health Condition', 'Symptoms', 'Treatment Type', 'Age Valid', 'Incorrect Treatment']]
y = df['Immediate Care Required']

# Convert categorical features to numeric
# Preprocessing the features with handling unknown categories in OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('animal_type', OneHotEncoder(handle_unknown='ignore'), ['Animal Type']),
        ('health_condition', OneHotEncoder(handle_unknown='ignore'), ['Health Condition']),
        ('treatment_type', OneHotEncoder(handle_unknown='ignore'), ['Treatment Type']),
        ('symptoms', OneHotEncoder(handle_unknown='ignore'), ['Symptoms']),
        ('age_valid', 'passthrough', ['Age Valid']),
        ('incorrect_treatment', 'passthrough', ['Incorrect Treatment'])
    ])


# Create the pipeline with Logistic Regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
