# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# Load data
PATH = 'input/'
train_data = pd.read_csv(PATH + 'train.csv')
test_data = pd.read_csv(PATH + 'test.csv')

# Function to preprocess title, family size, and other columns
def preprocess_data(df):
    # Create FamilySize feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Map Sex to numeric values
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Extract and map Title from Name, then drop Name column
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others'
    )
    df['Title'] = df['Title'].replace(['Ms', 'Mlle', 'Mme'], 'Miss')
    df['Title'] = df['Title'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Others': 4})
    
    # Drop unnecessary columns
    df = df.drop(columns=['Ticket', 'PassengerId', 'Cabin', 'Name'])
    return df

# Apply preprocessing to both train and test data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Fill missing values in Embarked with the majority class (assuming 2 is the majority class code)
train_data['Embarked'] = train_data['Embarked'].fillna(2)
test_data['Embarked'] = test_data['Embarked'].fillna(2)

# Fill missing values in Age based on similar rows
def fill_age_by_similar_passengers(df, reference_df):
    NaN_indexes = df['Age'][df['Age'].isnull()].index
    for i in NaN_indexes:
        # Find median Age for similar passengers in reference_df
        pred_age = reference_df['Age'][((reference_df.SibSp == df.iloc[i]["SibSp"]) & 
                                        (reference_df.Parch == df.iloc[i]["Parch"]) & 
                                        (reference_df.Pclass == df.iloc[i]["Pclass"]))].median()
        # Use calculated median if available; otherwise, use the overall median
        df.at[i, 'Age'] = pred_age if not np.isnan(pred_age) else reference_df['Age'].median()
    return df

# Apply the age-filling strategy to both train and test sets
train_data = fill_age_by_similar_passengers(train_data, train_data)
test_data = fill_age_by_similar_passengers(test_data, train_data)

# Fill missing values in Fare in test set with median Fare from the train set
test_data['Fare'] = test_data['Fare'].fillna(train_data['Fare'].mean())

# Separate features and target for training
X_train = train_data.drop(columns='Survived')
y_train = train_data['Survived']
X_test = test_data.copy()

# Combine train and test data for consistent encoding
combined_data = pd.concat([X_train, X_test], keys=['train', 'test'])

# One-Hot Encoding for categorical features
combined_data = pd.get_dummies(combined_data, columns=['Embarked', 'Title'], drop_first=True)

# Split back into X_train and X_test
X_train = combined_data.loc['train'].reset_index(drop=True)
X_test = combined_data.loc['test'].reset_index(drop=True)

# Z-score Scaling for all features
# scaler = StandardScaler()
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save processed data
X_train.to_csv('X_train_processed.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_test.to_csv('X_test_processed.csv', index=False)

# Check processed data
print("X_train sample:")
print(X_train.head())
print("\nX_test sample:")
print(X_test.head())

