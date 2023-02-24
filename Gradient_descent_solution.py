""""Normal Equations
""""

# 1.  A new research tries to find a relation between ages of two siblings and 
#     the the number of times they talk to each other in an average week.

import numpy as np
import matplotlib.pyplot as plt

X=np.array([[31,22],[22,21],[40,37],[26,25]])
Y=np.array([2,3,8,12])


def The_best_linear_regQ(X,Y):
    #h=(X^t*X)^-1*X^t*Y
    XT=X.transpose()
    XTY=np.dot(XT,Y)
    XTX_inv=np.linalg.inv(np.dot(XT,X))
    h=np.dot(XTX_inv,XTY)
    return h      #h(Q1,Q2)

    #return np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose()),Y)

def Chek_reg(X,Y):
    h_reg= The_best_linear_regQ(X,Y)  #--->[Q1,Q2]
    sum_QX =   np.dot(X,h_reg)        #--->sum(X*Q)
    loss   = ((sum_QX-Y)**2).mean()
    print('h_Regression:', h_reg)            #--> ze kav atama le nekudot (X1,X2)   parametrim h[Q1,Q2]
    print('predicted  y=sum_QX:', sum_QX)
    print('loss:', loss)
    return h_reg
    
# A. Assuming the research hypothesis is that there is a linear connection i.e. 
#    h(x1,x2)= θ1 x1 + θ2 x2 , find the best θ1 θ2.    
    
h_reg=Chek_reg(X,Y)   #h_Regression: [-0.58363913  0.89545547]
                      #predicted   y=sum_QX: [1.60720726 5.96450397 9.78628713 7.21176933]
                      #loss: 8.765136153462954

"""
x1=np.array([31,22,40,25])
x2=np.array([22,21,37,25])
plt.figure(num=1)                  # plt.figure(1); plt.gcf().number => 1
plt.scatter(x1, Y, color='red')    # scatter is graph of points (x,y) from training data, mezaer nekudat
plt.scatter(x2, Y, color='green')
plt.plot(x1,x2, h_reg[0]*x1+h_reg[1]*x2)
plt.show()
"""

# Add to matrix constant feature [1,1,1,1]- this low the loss
"""
X1=np.ones((4,3))
X1[:,:-1]=X      
X1
"""
# matrix with constant bias 4*3
x_ones=np.ones((4,1))
X1=np.column_stack((X, x_ones))  
h_reg=Chek_reg(X1,Y)             #h_Regression: h[Q1,Q2,Q3]=[-0.64009062  0.8781789 2.24049985]
                                 #predicted y=sum_QX: [1.71762643 6.60026308 9.1294943  7.55261619]
                                 #loss:  8.524152294650692                         
                                
# B. One research assistant suggested to another feature: 
#    The age difference between the siblings i.e. x3 = x1 −x2
#    Can this improve the fitting? (atama linyarit)
#    If it can calculate the new  θ1 θ2 θ3. if it will not, explain why.

X2=X1.copy()            #copy all matrix X1 to X2
X2[:,2]=X[:,0]-X[:,1]   #osafat amuda 2: amuda0-amuda1
h_reg=Chek_reg(X2,Y)    #h_Regression: Q=[ 0.09375  -0.015625 -0.859375]
                        #predicted   y=sum_QX: [-5.171875  0.875     0.59375   1.1875  ]
                        #loss: 56.92852783203125  gavoa biglyal tlut linyarit bein features x1,x2

# C. Another research assistant suggested to add the square of the age difference
#    as a feature i.e. x3= (x1−x2)^2.  Can this improve the fitting?
#    If it can calculate the new  θ1 θ2 θ3. if it will not, explain why.
                        
X3=X1.copy()
X3[:,2]=(X[:,0]-X[:,1])**2   #add column 2: (col0-col1)**2 
h_reg=Chek_reg(X3,Y)    #h_Regression: [-8.65644199  9.36076005  0.79598912]
                        #predicted   y=sum_QX: [2.06213797 6.93022643 7.25434439 9.74749868]
                        #loss: 5.2700763559768244  ahi yarad ki ein tlut linyarit bein features x1, x2
         
                        
 # D.  A third research assistant suggested to add a strange feature: a vector of ones, i.e. x3i=1 Vi
 #     Can this improve the fitting? if it will not, explain why. If it will,
 #     calculate the new If it will, calculate the newθ1 θ2 θ3. and also explain 
 #     what is the meaning of this new feature. (hint: start by writing what is the new h(x1,x2,x3)


x_ones=np.ones((4,1))
X4=np.column_stack((X3,x_ones)) #mehaber amudot: noten matriza amudot shel data ve amuda ehadim
h_reg=Chek_reg(X4,Y)            #h_Regression: Q[-25.5625    27.8125     2.515625 -21.203125]
                                #predicted   y=sum_QX: [ 2.  3.  8. 12.]
                                #loss: 1.3108106600516218e-21  -hosafat feature x3 le linyari
                                # vegam vektor constant bias noten loss ahi namuh





#############################################################################################
""""Gradient descent
""""                               

x=np.array([0,1,2], dtype=np.float32)
y=np.array([1,3,7], dtype=np.float32)
x_ones=np.ones((3,1))
X=np.column_stack((x_ones,x,x**2))  # X= np.hstack((x_ones,x,x**2)) mehaber amudot: noten matriza amudot shel data ve amuda ehadim
                                    # X-yazarnu matriza X(3x3) bishvil Q0+Q1X+Q2X^2
h_reg=Chek_reg(X,y)  #h_Regression: [1. 1. 1.] ze [Q1,Q2,Q3] ze kav atama le nekuda X   
                     # predicted  y=sum_QX: [1. 3. 7.]
                     # loss: 3.828604926672644e-28

#X=np.c_[np.ones_like(x),x,x**2]  --> c_ mehaber lefi column, ones_like(x)- ose matrix 1 lefi shape x

start_Q=np.array([2,2,0], dtype=np.float32)

def gradient_iter(X,y,start_Q, alfa, iterr):
    Q=start_Q.copy()
    for i in range(iterr):
        sum_QX= np.dot(X,Q)       #All matrixX(3x3)*vectorQ(1x3)=array([2., 4., 6.]) (1x3)
        Ls    = 0.5*((sum_QX-y)**2).mean()
        Lsgrad= np.dot(sum_QX-y,X)/len(X)   #nigzeret shel Ls
        Q     = Q-alfa*Lsgrad
        print('iter:{}; loss:{}; Q:{}'.format(iterr,Ls,Q))
        
        
gradient_iter(X,y,start_Q,1,1)    #iter:1;   loss:0.5;                    Q:[1.66666667       2.33333333      1.        ]
gradient_iter(X,y,start_Q,1,100)  #iter:100; loss:7.374485599080004e+163; Q:[-8.10807089e+81 -1.35348987e+82 -2.50793783e+82]
gradient_iter(X,y,start_Q,0.1,1)  #iter:1;   loss:0.5;                    Q:[1.96666667       2.03333333      0.1
gradient_iter(X,y,start_Q,0.1,100)#iter:100; loss:0.010707293346528925;   Q:[0.85701293       1.64278695      0.6993264 ]
gradient_iter(X,y,start_Q,0.01,1) #iter:1;   loss:0.5;                    Q:[1.99666667       2.00333333      0.01      ]
gradient_iter(X,y,start_Q,0.01,100)#iter:100; loss:0.18917195809609136;   Q:[1.56810916       1.98098507      0.28687238]
gradient_iter(X,y,start_Q,0.09,100)#iter:100; loss:0.011652564916106624;  Q:[0.85724894       1.66984904      0.68464517]

##########################################################################################

def h_model(Q):
    sum_QX=np.dot(X,Q)
    return sum_QX


def loss_mse(sum_QX,y):
    Ls=0.5*((sum_QX-y)**2).mean()
    return Ls
 
def loss_mse_grad(sum_QX,y):
    Lsgrad=(np.dot(sum_QX-y,X))/len(X)
    return Lsgrad


def gradient_iter_momentum(X,y,start_Q, alfa, moment_masa, iterr):
    Q=start_Q.copy()
    v=np.array([0,0,0], dtype=np.float32)   #v=np.zeros_like(start_Q)
    for i in range(iterr):
        sum_QX = h_model(Q) # np.dot(X,Q)
        loss   = loss_mse(sum_QX,y)
        grad   = loss_mse_grad(sum_QX,y)
        v      = moment_masa*v -alfa*grad
        Q      = Q+v
        print('iter:{}; loss:{}; Q:{}'.format(iterr,loss,Q))

gradient_iter_momentum(X,y,start_Q, 0.01,0.9,100)  
#iter:100; loss:0.011232499032582144; Q:[0.83554253 1.6576578  0.6989489 ]




def gradient_iter_momentum_nesterov(X,y,start_Q, alfa, moment_masa, iterr):
    Q=start_Q.copy()
    v=np.array([0,0,0], dtype=np.float32)   #v=np.zeros_like(start_Q)
    for i in range(iterr):
        sum_QX    = h_model(Q)   # np.dot(X,Q)
        sum_Qmv_X = h_model(Q+moment_masa*v) # np.dot(X, Q+moment_masa*v) grad benukada Q aherei impuls shela m*v
        loss      = loss_mse(sum_QX,y)
        grad      = loss_mse_grad(sum_Qmv_X,y)
        v         = moment_masa*v -alfa*grad
        Q         = Q+v
        print('iter:{}, loss:{}, Q:{}'.format(iterr,loss,Q))
        
gradient_iter_momentum_nesterov(X,y,start_Q, 0.01,0.9,100)      
#iter:100, loss:0.011260068015421251, Q:[0.83509172 1.65825141 0.69807773]












