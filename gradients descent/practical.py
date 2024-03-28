import numpy as np

def gradient_descent(x,y):
    m_curr_slope = b_curr_intercept = 0
    iteration = 1000
    n = len(x)
    learnin_rate = 0.08
    for i in range(iteration):
        y_predicted = m_curr_slope * x + b_curr_intercept # y = mx +b
        cost = (1/n) * sum( [val **2 for val in (y - y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted)) #derivative of m
        bd = -(2/n)*sum(y-y_predicted)#derivative of m
        m_curr_slope = m_curr_slope - learnin_rate * md #m = m-learning rate * derivative of [m] artial derivative
        b_curr_intercept = b_curr_intercept - learnin_rate * bd  #m = b-learning rate * derivative of [bd] partial derivative
        print(" m {}, b {}, cost {}, iteration {}".format(m_curr_slope,b_curr_intercept,cost,i))

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)