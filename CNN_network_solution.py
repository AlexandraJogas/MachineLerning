#Today we will write the forward pass of a very simple convolutional neural network (one
#convolutional layer and then one fully connected layer) from scratch using only NumPy.

import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt 
from tqdm import tqdm  # loads bar= titkadem be lulaa

#1. Load MNIST data using scikit-learn: http://scikit-learn.org/stable/datasets/index.html
digits = load_digits()  

print(digits.data.shape)  
plt.gray()   
for i in range(10):  
    plt.matshow(digits.images[i])   
plt.show() 

data = digits.data/255   
X=digits.data
X-= int(np.mean(X))     
X/= int(np.std(X))       
n_samples, n_features = data.shape   
n_digits = len(np.unique(digits.target))  
labels = digits.target                    

# training data
trainIdx = np.random.rand(len(labels)) < 0.8  
features_train = data[trainIdx]
labels_train   = labels[trainIdx]
features_test  = data[~trainIdx]     
labels_test    = labels[~trainIdx]  
print ("Number of training samples: ",   features_train.shape[0])   
print ("Number of test samples: ",       features_test.shape[0])    
print ("Feature vector dimensionality: ",features_train.shape[1])  

y_train=labels_train.reshape(labels_train.shape[0],1)   
y_test =labels_test.reshape( labels_test.shape[0], 1)
train_data = np.hstack((features_train, y_train))

num_epochs=2
batch_size = 32

num_classes = 10   
lr = 0.01
beta1 = 0.95
beta2 = 0.99
img_dim = 8        
img_depth = 1     

## Initializing all the parameters
def initializeFilter(size, scale = 1.0):  
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)  
                                                                  
def initializeWeight(size):                             
    return np.random.standard_normal(size=size) * 0.01   

    
def convolution(image, filt, bias, s=1):    
    '''
    Confolves `filt` over `image` using stride `s`
    '''
    (n_f, n_c_f, f, _) = filt.shape  
    n_c, in_dim, _     = image.shape
    
    out_dim = int((in_dim - f)/s)+1  
    
    # ensure that the filter dimensions match the dimensions of the input image
    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
    
    out = np.zeros((n_f,out_dim,out_dim)) # create the matrix to hold the values of the convolution operation, leathel convolution
    
  
    for curr_f in range(n_f):        
        curr_y = out_y = 0            
        # move filter vertically across the image
        while curr_y + f <= in_dim:  
            curr_x = out_x = 0        
            # move filter horizontally across the image 
            while curr_x + f <= in_dim: 
                # perform the convolution operation and add the bias
                # ze agdara convolution: out
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f] 
                curr_x += s            
                out_x += 1
            curr_y += s      
            out_y += 1
        
    return out   

def maxpool(image, f=2, s=1):  
    '''
    Downsample input `image` using a kernel size of `f` and a stride of `s`
    '''
    n_c, h_prev, w_prev = image.shape
    
    # calculate output dimensions after the maxpooling operation.
    h = int((h_prev - f)/s)+1 
    w = int((w_prev - f)/s)+1
    
    # create a matrix to hold the values of the maxpooling operation.
    downsampled = np.zeros((n_c, h, w)) 
    
    # slide the window over every part of the image using stride s. Take the maximum value at each step.
    for i in range(n_c):
        curr_y = out_y = 0
        # slide the max pooling window vertically across the image
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            # slide the max pooling window horizontally across the image
            while curr_x + f <= w_prev:
                # choose the maximum value within the window at each step and store it to the output matrix
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled

def softmax(raw_preds):
    '''
    pass raw predictions through softmax activation function
    '''
    out = np.exp(raw_preds) # exponentiate vector of raw predictions
    return out/np.sum(out) # divide the exponentiated vector by its sum. All values in the output sum to 1.

f = 4          # filter be godel 4X4
num_filt1 = 4  # 4islo filterov
num_filt2 = 4                # mispar channels
f1, f2, w3, w4 = (num_filt1 ,img_depth, f, f), (num_filt2 ,num_filt1, f, f), (128,4), (10, 128)
               # (output=num filtr1, input=num chanels, godel filtr), (output=num filtr2, input=output num filtr1, godel filtr),
# esh 2 shhavot, ve be kol shihva 2 filters zazim
# leapes filters, mishkolot               
f1 = initializeFilter(f1)
f2 = initializeFilter(f2)
w3 = initializeWeight(w3)
w4 = initializeWeight(w4)
# bias samim afasim be athala
b1 = np.zeros((f1.shape[0],1))
b2 = np.zeros((f2.shape[0],1))
b3 = np.zeros((w3.shape[0],1))
b4 = np.zeros((w4.shape[0],1))

params = [f1, f2, w3, w4, b1, b2, b3, b4]

cost = []
#2. Write a loop over M the number of epochs and in it build a loop over n the number of
#samples in every mini-batch.
for epoch in range(num_epochs): 
    np.random.shuffle(train_data)    
    batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]  
                                                 
    t = tqdm(batches)
    for x,batch in enumerate(t):
        X = batch[:,0:-1] # get batch inputs
        X = X.reshape(len(batch), img_depth, img_dim, img_dim)
        Y = batch[:,-1] # get batch labels
    
        cost_ = 0
        batch_size = len(batch)
        for i in range(batch_size): 
            image = X[i]                    
            y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)
            # Collect Gradients for training example
            #grads, loss = conv(x, y, params, 1, 2, 2)
            conv_s =1  # step=stride 
            pool_f =2  # size filtr
            pool_s =2  # step=stride
            [f1, f2, w3, w4, b1, b2, b3, b4] = params 
    
#3. Build the first forward pass. The first convolutional layer will have two layers in its output.
#The convolution layer is comprised of four for loops, multiplications and additions.

            conv1 = convolution(image, f1, b1, conv_s) # convolution operation
            conv1[conv1<=0] = 0 # pass through ReLU non-linearity
            
            conv2 = convolution(conv1, f2, b2, conv_s) # second convolution operation
            conv2[conv2<=0] = 0 # pass through ReLU non-linearity
            
            pooled = maxpool(conv2, pool_f, pool_s) # maxpooling operation,  pool_f= move window size,  poop_s=stride
            
            (nf2, dim2, _) = pooled.shape
#4. Flatten the output of the previous convolutional network to a one dimentional vector
            fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer
            
            z = w3.dot(fc) + b3 # first dense layer w*filtr = fully connected
            z[z<=0] = 0 # pass through ReLU non-linearity
            
#5. Write the forward pass of a fully-connected layer. The input should be the size of the
#previous flatted layer. The output should have 10 neurons in order to predict the one-hot
#probability of the 10 possible digit classes.          
            out = w4.dot(z) + b4 # second dense layer
#6. Use a softmax in order to convert the numbers to probabilities. Write the softmax
#function as a separte function using exponent, multiplication and division.     
            probs = softmax(out) # predict class probabilities with the softmax activation function
  
            loss = -np.sum(y * np.log(probs)) # categorical cross-entropy loss
        

#7. Thats it for now. Begin thinking about how would you implement the backward pass in
#the next class.
#plt.close('all')           
