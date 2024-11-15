# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data
X_train = pd.read_csv('./X_train_processed.csv')
Y_train = pd.read_csv('Y_train.csv')
X_test = pd.read_csv('./X_test_processed.csv')

Y_real_test = pd.read_csv('./test/gender_submission.csv')

Y_real_test = Y_real_test.drop("PassengerId", axis=1)



# X_test = X_test[X_train.columns]

# If Y_train has a single column with the target, extract it as a series
Y_train = Y_train.squeeze()

# Initialize logistic regression model
model = LogisticRegression(max_iter=10000)

# Train the model
model.fit(X_train, Y_train)

# Make predictions
predictions = model.predict(X_test)

# (Optional) Evaluate the model on the training set
# train_predictions = model.predict(X_train)
# accuracy = accuracy_score(Y_train, train_predictions)
# conf_matrix = confusion_matrix(Y_train, train_predictions)


# (Optional) Evaluate the model on the training set
train_predictions = model.predict(X_test)
accuracy = accuracy_score(Y_real_test, train_predictions)
conf_matrix = confusion_matrix(Y_real_test, train_predictions)

print(f"Training Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)

# Save predictions to a CSV file
pd.DataFrame(predictions, columns=['Survived']).to_csv('predictions_test.csv', index=False)

