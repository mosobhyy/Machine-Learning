# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data
path = 'E:\\Projects\\Python\\Machine Learning\\Multivariate Linear Regression\\data.txt'
data = pd.read_csv(path, header=None, names=['Space', 'Rooms', 'Price'])

# Show data details
print('All Data =\n', data.to_string())
print('\n**************************************\n')
print('Data head =\n', data.head())
print('\n**************************************\n')
print('Data Description =\n', data.describe())
print('\n**************************************\n')

# rescaling data
data = (data - data.mean()) / data.std()
print('data after normalization = \n', data.head(10))

# Draw data
data.plot(kind='scatter', x='Space', y='Price', figsize=(5, 5))
data.plot(kind='scatter', x='Rooms', y='Price', figsize=(5, 5))
# As the data is linearly separable, then we will use "linear regression" model using straight line equation


# =========================================================================

#
# Adding a new column called "ones" before the data to satisfy "straight-line equation"
data.insert(0, 'Ones', 1)
print('New Data =\n', data.head())
print('\n**************************************\n')

#  Separate X (training data) from Y (target variable)
rows = data.shape[0]
cols = data.shape[1]
X = data.iloc[:, :cols - 1]
Y = data.iloc[:, cols - 1:]
print('X =\n', X.head())
print('\n**************************************\n')
print('Y =\n', Y.head())
print('\n**************************************\n')

# Convert from data frames to numpy matrices
X = np.matrix(X.values)
Y = np.matrix(Y.values)
print('X Matrix =\n', X)
print('\n**************************************\n')
print('X Shape =', X.shape)
print('\n**************************************\n')
print('Y Matrix =\n', Y)
print('\n**************************************\n')
print('Y Shape =', Y.shape)
print('\n**************************************\n')

# Initialize Theta values by '0'
Theta = np.matrix(np.array([0, 0, 0]))


print('Theta Matrix =\n', Theta)
print('\n**************************************\n')
print('Theta Shape =', Theta.shape)
print('\n**************************************\n')


# =========================================================================


# Compute Cost Function
def cost(x, y, theta):
    z = np.power(((x * theta.T) - y), 2)
    return np.sum(z) / (2 * len(x))


print('Cost Function = ', cost(X, Y, Theta))
print('\n**************************************\n')


# Gradient Descent function
def gradientDescent(x, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    loss = np.zeros(iters)

    for i in range(iters):
        error = (x * theta.T) - y

        for j in range(theta.shape[1]):
            term = np.multiply(error, x[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(x)) * np.sum(term))

        theta = temp
        loss[i] = cost(x, y, theta)
    return theta, loss


# initialize variables for learning rate and iterations
alpha = 0.1
iters = 100

# perform gradient descent to "fit" the model parameters
gd, loss = gradientDescent(X, Y, Theta, alpha, iters)

print('GD = ', gd)
print('\n**************************************\n')
print('Loss  = ', loss[0:50])
print('\n**************************************\n')
print('Cost = ', cost(X, Y, gd))
print('**************************************')

# =========================================================================

# get best fit line for "Space vs Price"
x = np.linspace(data.Space.min(), data.Space.max(), 100)
f = gd[0, 0] + (gd[0, 1] * x)

# draw the line for "Space vs Price"
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Space, data.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Space')
ax.set_ylabel('Price')
ax.set_title('Space vs. Price')

# # get best fit line for "Rooms vs Price"
# x = np.linspace(data.Rooms.min(), data.Rooms.max(), 100)
# f = gd[0, 0] + (gd[0, 1] * x)
#
# # draw the line for "Rooms vs Price"
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.plot(x, f, 'r', label='Prediction')
# ax.scatter(data.Rooms, data.Price, label='Training Data')
# ax.legend(loc=2)
# ax.set_xlabel('Rooms')
# ax.set_ylabel('Price')
# ax.set_title('Rooms vs. Price')

# draw error graph
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(np.arange(iters), loss, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

plt.show()
