from sklearn.datasets import load_breast_cancer
import numpy as np 
import matplotlib.pyplot as plt 

# command to activate the virtual environment: source neuralnet/sklearn-env/bin/activate
# Command to run the code in the venv: python neuralnet/main.py 

# Import dataset 
dataset = load_breast_cancer()
X = dataset.data
Y = dataset.target 

data = np.column_stack((X, Y)) 
m, n = data.shape 
np.random.shuffle(data) # randomly reorder the elements of the array 

# Create test set 
test_data = data[455:] # remaining 114 samples for testing 

X_test = test_data[:, :-1] 
Y_test = test_data[:, -1:]  

training_data = data[:455] # First 455 samples for training

X_train = training_data[:, :-1]
Y_train = training_data[:, -1:] 

# Standardize data by calculating the mean and average of the training data 
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# --------Neural Network Calculations and Implementation--------

# Initialize all the parameters 
# np.random generates random numbers between 0 and 1, subtracting 0.5 shifts the range to -0.5 to 0.5 so it produces both positive and negative weights. 
def init_params(): 
    W1 = np.random.rand(30, 8) - 0.5
    b1 = np.random.rand(1, 8) - 0.5

    W2 = np.random.rand(8, 8) - 0.5
    b2 = np.random.rand(1, 8) - 0.5

    W3 = np.random.rand(8, 1) - 0.5
    b3 = np.random.rand(1, 1) - 0.5 
    return W1, b1, W2, b2, W3, b3 

# Define the ReLU activation function
def ReLU(Z):
    return np.maximum(Z, 0) 

# Define the Sigmoid activation function
# This activation function is used to map the output of the neural network to a probability value between 0 and 1. 
def sigmoid(Z): 
    return 1 / (1 + np.exp(-Z)) 

def forward_prop(W1, b1, W2, b2,W3, b3, X): 
    Z1 = X.dot(W1) + b1 
    A1 = ReLU(Z1)

    Z2 = A1.dot(W2) + b2 
    A2 = ReLU(Z2)

    Z3 = A2.dot(W3) + b3
    A3 = sigmoid(Z3) 
    return Z1, A1, Z2, A2, Z3, A3

# Define the derivative of the ReLU activation function 
def ReLU_deriv(Z):
    return Z > 0 

# Define the backward propagation function to compute the gradients of the loss function with respect to the weights and biases. 
def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y): 
    dZ3 = A3 - Y
    dW3 = 1 / m * A2.T.dot(dZ3)
    db3 = 1 / m * np.sum(dZ3)

    dZ2 = dZ3.dot(W3.T) * ReLU_deriv(Z2)
    dW2 = 1 / m * A1.T.dot(dZ2) 
    db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)

    dZ1  = dZ2.dot(W2.T) *  ReLU_deriv(Z1) 
    dW1 = 1 / m * X.T.dot(dZ1) 
    db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)  # (axis=0) means sum down all the training examples while keeping the 8 neurons columns separate. (keepdims=True) keeps the result shaped as (1,8) instead of (8,) 
    return dW1, db1, dW2, db2, dW3, db3 

# Update the parameter (weights and biases) using the computed gradients and a learning rate. 
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha): 
    W1 = W1 - alpha * dW1 
    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dW2 
    b2 = b2 - alpha * db2

    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3 

# Define method to get the inital predictions made by the neural network before back propagation learning 
def get_predictions(A3):
    return (A3 >= 0.5).astype(int) 

# Define the method to calculate the accuracy of the predictions made by the neural network 
def get_accuracy(predictions, Y): 
    print(predictions, Y)
    return np.mean(predictions == Y ) 

# Define the loss function (gradient descent) to calculate the loss between the predicted and actual values 
def gradient_descent(X, Y, alpha, iterations): 
    W1, b1, W2, b2, W3, b3 = init_params() 
    for i in range(iterations): 
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X) 
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y) 
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 10 == 0:
            print( "Iteration: ", i)
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y)) 
    return W1, b1, W2, b2, W3, b3

# Start training the neural network using the training data, learning rate, and number of iterations
W1, b1, W2, b2, W3, b3 = gradient_descent( X_train, Y_train, 0.10, 500) 

# --------Visualization of the predictions--------

# Make predictions using the test data
Z1_test, A1_test, Z2_test, A2_test, Z3_test, A3_test = forward_prop(
    W1, b1, W2, b2, W3, b3, X_test
)

test_predictions = get_predictions(A3_test)

# Calculate test accuracy
test_accuracy = np.mean(test_predictions == Y_test)

# Count malignant and benign tumors in the full dataset
malignant_count = np.sum(dataset.target == 0)
benign_count = np.sum(dataset.target == 1)

# Create bar graph
tumor_types = ["Malignant", "Benign"]
tumor_counts = [malignant_count, benign_count]

plt.figure(figsize=(8, 6))

bars = plt.bar(tumor_types, tumor_counts)

plt.title(f"Breast Cancer Tumor Distribution\nNeural Network Test Accuracy: {test_accuracy:.2%}")
plt.xlabel("Tumor Classification")
plt.ylabel("Number of Tumors")

# Display the number above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 5,
        str(int(height)),
        ha="center"
    )

plt.tight_layout()
plt.show()
