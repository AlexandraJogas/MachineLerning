import numpy as np

# 1
def get_sigmoid(x):
    '''
    input: x: array - an array of float numbers
    output: y: array - the same array after passing through a sigmoid function: 1/(1+e^(-x))
    '''
    return 1/(1+np.exp(-x))
# 2
def get_sigmoid_derivative(x):  
    '''
    input: x: array - an array of float numbers
    output: s: array - an array of sigmoid derivateves [s'(x) = s(x)*(1-s(x))].
    '''
    return x*(1-x)

# 3
np.random.seed(1)                  
w1 = 2*np.random.random((3,3))-1  
w2 = 2*np.random.random((3,1))-1

# 4
def calc_error(Y,l):
    ''' input : Y - array - an array of float numbers (the label array)
                l - array - an array of float numbers (the hidden layer array)
        output: error - array - the difference between the two inputs.'''
    error = Y-l
    return error

def run_prog(l0,w,l2):
    '''input: l0 - array - the input of the network.
               w - array - the weights of the network
              l2 - array - the output of the network ( the label)
        output: l1 - array - the hidden layer after learning
                 w - array - the weights after the learning process
                error - array - the loss function of the network per neuron'''
    for i in range(10000):
        l1 = get_sigmoid((np.dot(l0,w)))
        error  = calc_error(l2,l1)
        error  = error*get_sigmoid_derivative(l1)
        w += np.dot(l0.T, error)
    return w,l1, error

#5
x = np.array([[0,0,1], [0,1,1,], [1,0,1], [1,1,1]])
y = np.array([[0],[0],[1],[1]])

final_w, output, error = run_prog(x,w2,y)

print('The final output with 1 hidden layer:\n',output)
print('The final weights with 1 hidden layer:\n',final_w)
print('The final loss_function with 1 hidden layer:\n',error)


# 6
def run_prog2(X,w1,w2,Y):
    '''
    input: l0 - the input of the network.
              w1 - the weights of the network for the 1st hidden layer
              w - the weights of the network for the 2nd hidden layer
              l2 - the output of the network ( the label)
        output: l2: the hidden layer after learning
                l2_error: The loss function of the network
                '''
    for i in range(10000):
        l1 = get_sigmoid(np.dot(X,w1))
        l2 = get_sigmoid(np.dot(l1,w2))
        l2_error = calc_error(Y,l2)
        l2_error = l2_error * get_sigmoid_derivative(l2)
        l1_error = np.dot(l2_error,w2.T)
        l1_error = l1_error * get_sigmoid_derivative(l1)
        w1 += np.dot(X.T, l1_error)
        w2 += np.dot(l1.T,l2_error)
        if(i%1000==0):
            print(l2_error)
    return l2, l2_error

output2, error2 = run_prog2(x, w1, w2, y)
print('\nThe final output with 2 hidden layer:\n',output2)
print('\nThe final los_function with 2 hidden layer:\n',error2)

# 7
print('For the new X and Y:')
x = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[ 0 ],[ 1 ],[ 1 ],[ 0 ]])

np.random.seed(1)
w = np.random.random((3,1))
final_w, output, error = run_prog(x,w,y)
print('The final output with 1 hidden layer:\n',output)
print('The final weights with 1 hidden layer:\n',final_w)
print('The final loss_function with 1 hidden layer:\n',error)

np.random.seed(1)
w1 = np.random.random((3,3)) 
w2 = np.random.random((3,1))
output2, error2 = run_prog2(x, w1, w2, y)
print('\nThe final output with 2 hidden layer:\n',output2)
print('\nThe final los_function with 2 hidden layer:\n',error2)
