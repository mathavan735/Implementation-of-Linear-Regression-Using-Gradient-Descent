# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph. 
````
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Mathavan S
RegisterNumber:  212221220031
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) 

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) 

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")





`````

## Output:

# Profit prediction:

![WhatsApp Image 2024-03-06 at 05 32 56_e650b96c](https://github.com/23013743/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161271714/41fe0253-d6ca-4bfc-ad94-b389ee23baf5)

# Function:

![WhatsApp Image 2024-03-06 at 05 34 37_b00a01a9](https://github.com/23013743/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161271714/4a599df4-185f-4597-8a62-b37c094033e9)

# Gradient descent:

![WhatsApp Image 2024-03-06 at 05 35 14_6d7e38a0](https://github.com/23013743/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161271714/329cfd58-1889-4168-a6df-9e483e7cc642)

# COST FUNCTION USING GRADIENT DESCENT:

![WhatsApp Image 2024-03-06 at 05 36 56_d1f4b9c9](https://github.com/23013743/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161271714/b74bc8ad-fbc8-4ee8-922b-6a91293f1406)

# LINEAR REGRESSION USING PROFIT PREDICTION:


![WhatsApp Image 2024-03-06 at 05 38 15_2c8b56e8](https://github.com/23013743/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161271714/395a06b3-ce6d-4d1e-a359-b325214c1eb8)

# PROFIT PREDICTION FOR A POPULATION OF 35000:

![WhatsApp Image 2024-03-06 at 05 39 16_a9d47f42](https://github.com/23013743/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161271714/0f69dc99-8d01-45fc-9ff7-055f1d7005f3)

# PROFIT PREDICTION FOR A POPULATION OF 70000:


![WhatsApp Image 2024-03-06 at 05 40 05_a86c44f7](https://github.com/23013743/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161271714/74fa93c8-d0fd-43ed-aec5-e0c7cbd0be15)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
