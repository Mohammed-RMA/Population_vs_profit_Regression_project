import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# path of data
path='data.txt'
data=pd.read_csv(path,header=None,names=['Population','Profit'])
plt.scatter(data.Population,data.Profit)

# dividing the data
data.insert(0,'ones',1)
cols=data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]

# convert to matrices
X=np.matrix(X)
y=np.matrix(y)
theta=np.matrix(np.array([0,0]))

# cost function
def Copmutecost(X,y,theta):
    cost=np.pow((X*theta.T-y),2)
    return (np.sum(cost)/(2*len(X)))

# Gradientdescent function
def gradient_descent(X,y,theta,alpha,iterations):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.shape[1])
    cost=np.zeros(iterations)
    for i in range(iterations):
        error=X*theta.T-y
        for j in range(parameters):
            term=np.multiply(error,X[:,j])
            temp[0,j]=theta[0,j]-alpha*np.sum(term)/len(X)
        theta =temp
        cost[i]=Copmutecost(X,y,theta)
    return theta,cost





alpha=.01
iterations=1800
theta, cost = gradient_descent(X, y, theta, alpha, iterations)

print('theta = ' , theta)
print('cost  = ' , cost[0:50] )
print('computeCost = ' , Copmutecost(X, y, theta))

# best fit line
x=np.linspace(data.Population.min(),data.Population.max(),100)
f=theta[0,0]+theta[0,1]*x

# draw the best fit line
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

# draw error graph
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iterations), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

plt.show()
