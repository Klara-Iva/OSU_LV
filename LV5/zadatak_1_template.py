import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                           random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# a)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
            cmap='PiYG')

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
            cmap='bwr', marker='x')
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# b)
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# c)
# Retrieve the coefficients and intercept from the trained model
coef = lr_model.coef_[0]
intercept = lr_model.intercept_

# Define the decision boundary as a function of x1


def decision_boundary(x1):

    return (-coef[0]*x1 - intercept) / coef[1]


# Plot the decision boundary along with the training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='Spectral')
plt.plot(X_train[:, 0], decision_boundary(X_train[:, 0]))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Logistic Regression Decision Boundary')
plt.show()

# d)
# Use the trained model to classify the test data
y_pred = lr_model.predict(X_test)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot()
plt.show()
# Compute the accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print the accuracy, precision, and recall
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)

# e)
# Plot the test dataset in the x1-x2 plane
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)

# Color the correctly classified examples in green and incorrectly classified examples in black
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        plt.scatter(X_test[i, 0], X_test[i, 1], c='g')
    else:
        plt.scatter(X_test[i, 0], X_test[i, 1], c='k',)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Test Dataset')
plt.show()
