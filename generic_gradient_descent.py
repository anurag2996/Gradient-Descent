import numpy as np

def step_gradient(X, Y, learning_rate, slope):
    #As there are n features, we will initialize n slopes as 0
    new_slope = np.zeros(X.shape[1])
    #new_slope = np.copy(slope)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i][j]
            y = Y[i]
            new_slope[j] = -(2/X.shape[0]) * (y - (slope * X[i]).sum()) * x
        #Update slope after every data point    
        slope = slope - (learning_rate * new_slope) 
        
    return slope

def gd(X, Y, learning_rate, num_iterations):
    #Let the initial slopes be 0
    slope = np.zeros(X.shape[1])
    print("Cost at starting : ", cost(X, Y, slope))
    for i in range(num_iterations):
        slope = step_gradient(X, Y, learning_rate, slope)
#         if i%100 == 0:
        print(" Cost at ",i, "iteration : ", cost(X, Y, slope))
    
#     print("Cost at end : ", cost(X, Y, slope))
    return slope

#Function to calculate cost
def cost(X, Y, slope):
    total_cost = 0
    for i in range(X.shape[0]):
        total_cost += (1/X.shape[0])*((Y[i] - (slope * X[i]).sum())**2)
        
    return total_cost
    

def run(X, Y, learning_rate, num_iterations):
    slope = gd(X, Y, learning_rate, num_iterations)
    return slope

learning_rate = 0.0730
num_iterations = 1000
test_data = np.loadtxt('/Users/Anurag/Desktop/test_boston_gd.csv',delimiter=",")
train_data = np.loadtxt('/Users/Anurag/Desktop/test_boston_gd.csv',delimiter=",")
X_test_l = data[:,0:12]
Y_test_l = data[:,12]
X_train_l = data[:,0:12]
Y_train_l = data[:,12]
y_ = run(X_train_l, Y_train_l, learning_rate, num_iterations)
y_pred = run(X_test_l, Y_test_l, learning_rate, num_iterations)