import numpy as np

print('Loading data...')
train_x = np.loadtxt('train_images.csv', delimiter=',').T
train_y = np.loadtxt('train_labels.csv', delimiter=',', skiprows=1, usecols=0).reshape(7000,1).T
test_x = np.loadtxt('test_images.csv', delimiter=',').T
test_y = np.loadtxt('test_labels.csv', delimiter=',', skiprows=1, usecols=0).reshape(2734,1).T
print('Training Set Features:',train_x.shape)
print('Training Set Labels:', train_y.shape)
print('Testing Set Features:', test_x.shape)
print('Testing Set Labels:', test_y.shape)

# Standardize the datasets
train_x = train_x/255.0
test_x = test_x/255.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def propagate(w, b, X, Y):
    m = X.shape[1]
    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b) # Compute the activation
    cost = (-1 / m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) # Compute the cost
    cost = np.squeeze(cost)
    # Backward propagation
    dw = (1/m)*np.dot(X, (A-Y).T)
    db = (1/m)*np.sum(A-Y)
    gradients = {'dw': dw, 'db': db}
    return gradients, cost

def optimize(w, b, X, Y, iterations, learning_rate, print_cost=False):
    for i in range(iterations):
        # Calculate cost and gradients
        gradients, cost = propagate(w, b, X, Y)
        # Retrieve derivatives
        dw = gradients['dw']
        db = gradients['db']
        # Update weights and b
        w = w-learning_rate*dw
        b = b-learning_rate*db
        # Print cost
        if print_cost and i % 100 == 0:
            print('Cost after iteration %s: %s' % (i, cost))
    # Package final results
    parameters = {'w': w, 'b': b}
    gradients = {'dw': dw, 'db': db}
    return parameters, gradients

def predict(w, b, X):
    m = X.shape[1]
    Y_predictions = np.zeros((1,m))
    # Calculate probabilities
    A = sigmoid(np.dot(w.T, X) + b)
    # Convert probabilities to predictions
    for i in range(m):
        Y_predictions[0,i] = 1 if A[0,i] > 0.5 else 0
    return Y_predictions

def model(train_x, train_y, test_x, test_y, iterations=2000, learning_rate=0.5, print_cost=False):
    # Initialize parameters
    w = np.zeros((train_x.shape[0],1))
    b = 0
    # Gradient descent
    parameters, gradients = optimize(w, b, train_x, train_y, iterations, learning_rate, print_cost)
    # Retrieve parameters
    w = parameters['w']
    b = parameters['b']
    # Make predictions
    training_set_predictions = predict(w, b, train_x)
    testing_set_predictions = predict(w, b, test_x)
    # Print errors
    print('Training Accuracy: ',(100 - np.mean(np.abs(training_set_predictions - train_y))*100),'%')
    print('Testing Accuracy: ',(100 - np.mean(np.abs(testing_set_predictions - test_y))*100),'%')

print('Training model...')
model(train_x, train_y, test_x, test_y, iterations=50000, learning_rate=0.01, print_cost=True)