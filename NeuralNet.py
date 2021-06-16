import numpy as np
X = np.array([[3, 5],
             [5, 1],
             [10, 2]])
y = np.array([[75], [82], [93]])

def sigmoid(z):
    return 1/(1 + np.exp(-z))

class Neural_Network(object):
    def __init__(self):
        self.inputLayersSize = 2
        self.outputLayersSize = 1
        self.hiddenLayersSize = 3

        self.W1 = np.random.randn(self.inputLayersSize, self.hiddenLayersSize)
        self.W2 = np.random.randn(self.hiddenLayersSize, self.outputLayersSize)

    def forward(self, X):
        self.Z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.a2, self.W2)

        y_hat = self.sigmoid(self.Z3)

        return y_hat

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

# Magic adapted from https://mgarod.medium.com/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6
# Makes it more convenient to add as I go in the same notebook
def add_method(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator

@add_method(Neural_Network)
def sigmoidPrime(self, z):
  return np.exp(-z) / ((1 + np.exp(-z))**2)

@add_method(Neural_Network)
def costFunction(self, X, y):
    self.y_hat = self.forward(X)
    J = sum(0.5 * (y - self.y_hat)**0.5)
    return J

@add_method(Neural_Network)
def costFunctionPrime(self, X, y):
    self.y_hat = self.forward(X)

    self.sigPrimeZ3 = self.sigmoidPrime(self.Z3)
    self.wrongness = y - self.y_hat

    self.delta_3 = np.multiply(-self.wrongness, self.sigPrimeZ3) # element-wise

    dJdW2 = np.dot(self.a2.T, self.delta_3)

    np.dot(X.T, self.delta_3)

    self.sigPrimeZ2 = self.sigmoidPrime(self.Z2)
    self.delta_2 = np.dot(self.delta_3, self.W2.T) * self.sigPrimeZ2

    dJdW1 = np.dot(X.T, self.delta_2)

    return dJdW1, dJdW2

# this is a helper function to get the weights in a simple format
@add_method(Neural_Network)
def get_params(self):
    params = np.concatenate(( self.W1.ravel(), self.W2.ravel() ))
    return params

# this is a helper function to set the weights
# it works with the getter to roundtrip the weights
# it assumes the weights come in as one giant 1D array
# then we need to chop up that 1d array at the correct places
# np.reshape can run over a 1d array and make rows and cols out of it
@add_method(Neural_Network)
def setParams(self, params):
    W1_start = 0
    W1_end = self.hiddenLayersSize * self.inputLayersSize
    self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayersSize, self.hiddenLayersSize))
    W2_end = W1_end + self.hiddenLayersSize * self.outputLayersSize
    self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayersSize, self.outputLayersSize))

# a helper function to get our gradients, then flatten them into a 1d array
@add_method(Neural_Network)
def computeGradients(self, X, y):
    dJdW1, dJdW2 = self.costFunctionPrime(X, y)
    return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

# This does the hard work of getting the params as a 1d array
# then iterating across them all to add and subtract our epsilon
@add_method(Neural_Network)
def computeNumericalGradient(self, X, y):
    paramsInitial = self.get_params()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for p in range(len(paramsInitial)):
        #Set perturbation vector: all values in there are now e
        perturb[p] = e
        self.setParams(paramsInitial + perturb)  # matrix addition, element-wise
        loss2 = self.costFunction(X, y)

        self.setParams(paramsInitial - perturb)
        loss1 = self.costFunction(X, y)

        #Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)

        #Return the value we changed to zero:
        perturb[p] = 0  # this seems unnecessary though.
        
        #Return Params to original value:
        self.setParams(paramsInitial)

    return numgrad

@add_method(Neural_Network)
def checkAccuracy(self):
    grad = self.computeGradients(X,y)
    numgrad = self.computeNumericalGradient(X, y)
    return np.linalg.norm(grad - numgrad) / np.linalg.norm(grad + numgrad)
