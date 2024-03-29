import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn

# Load the digits dataset
digits = load_digits()

# Display grayscale images
plt.gray()

# Display the first 5 digits
for i in range(5):
    plt.matshow(digits.images[i])  # Display digit image
    plt.show()

# Extract target values for the first 5 samples
target = digits.target[0:5]
print(target)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
print(len(x_train))
print(len(x_test))

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Evaluate the model accuracy
accuracy = model.score(x_test, y_test)
print("Model accuracy:", accuracy)

# Display a specific digit (e.g., the 70th digit)
target = digits.target[69]
print("Target for the 70th digit:", target)

plt.matshow(digits.images[69])  # Display the 70th digit image
plt.show()

# Make predictions for a specific digit (e.g., the 70th digit and the first 5 digits)
test_single = model.predict([digits.data[69]])
test_multiple = model.predict(digits.data[0:5])

print("Prediction for the 70th digit:", test_single)
print("Predictions for the first 5 digits:", test_multiple)


y_predicted = model.predict(x_test)

cm = confusion_matrix(y_test,y_predicted)

print(cm)

plt.figure(figsize =(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
