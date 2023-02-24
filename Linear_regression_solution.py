# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 07:40:06 2019

@author: Dom
"""

import numpy as np
import matplotlib.pyplot as plt


 #בנו וקטור של עשרה מספרים שלמים רנדומליי  a.#
# b. floats בנו וקטור של עשרה #
# בנו וקטור של חמישה מספרים אקראיים שהם כפולה של שלו c.#
 # **שאלת אתגר :  בחרו באקראיות מספר ששייך לעשרת המספרים הראשונים של סדר פיבונאצי d.#

 
def ex1():
    random_vector =np.random.randint(0,10,size=10)
    random_floats =np.random.rand(10)
    random_vector_by3 =np.random.randint(0,10,size=5)*3
    
    def fibonachi(n):
        if n==0:
            return 0
        elif n==1:
            return 1
        else:
            return fibonachi(n-1)+fibonachi(n-2)
    print([fibonachi(i) for i in range(10)])       #[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    fib10=[fibonachi(i) for i in range(10)]
    
    rand_index=np.random.randint(0,10)   #--> output 1 random number is 2 --> random_choice od funkzia
    rand_number=fib10[rand_index]        #--> 1
    print('fib10_random: {}; random_index: {}; random_number: {}'.format(fib10, rand_index, rand_number))
    
    return  random_floats



# בנו סט עשר נקודות אקראיות הנמצאות על קו ישר אחד העובר דרך ראשית הצירים#
#רמז: בחרו  בשיפוע 2 והכפילו בסט של 10 נקודות אקראיות על ציר הx לדוגמה בנקודות ם 1b  #
# שמרו את התוצאה במשתנה בשם  first_array
# הוסיפו רעש גאוסיאני לכל אחד מהנקודות#
    
def ex2(random_floats):
    #y=m*x   --> over dereh (0,0)
    m=2
    first_array= random_floats * m
    print('y=2*x: ', first_array )
    
    #Gausian noise --->Gaussian distribution:
    # mu, sigma = 0, 0.1   ---> mean and standard deviation
    # s = np.random.normal(mu, sigma, size=1000)
    gaussian_noise=np.random.normal(0,0.1, size=10)  #--> bahrnu tvah misparim katan raash she le yashpia 
    first_array += gaussian_noise
    print('first_array: ', first_array)
    
    
      # בנו וקטור אחר של עשר נקודות אקראיות על ישר אחד, שלא עובר דרך ראשית הצירים #
   #  בחרו קבוע אחר שהוא השיפוע והכפילו בו ועוד קבוע והוסיפו אותו לכל הנקודות #
  #  שימרו תוצאה והוסיפו רעש גאוסיאני second_array #
    #y=m*x+b   --> le over dereh (0,0)
    m=2
    b=3
    second_array= random_floats * m + b
    print('y=2*x+3: ', second_array )
    
    #Gausian noise --->Gaussian distribution:
    gaussian_noise=np.random.normal(0,0.1,size=10)
    second_array += gaussian_noise
    print('second_array: ', second_array)
    
    
#  third_array  בנו וקטור של עשר נקודות שנמצאות על פרבולה. שימרו התוצאה ב  
    # y=ax^2+bx+c
    a=2
    b=1
    c=3
    third_array=(random_floats**2)*a + random_floats*b+c
    print('y=ax^2+bx+c: ', third_array )
    
    #Gausian noise --->Gaussian distribution:
    gaussian_noise=np.random.normal(0,0.1,size=10)
    third_array += gaussian_noise
    print('third_array: ', third_array)
    
    return first_array, second_array, third_array

 # תרגול מטריצות  4*4  #
 # בנו שתי מטריצות     #
 # הכפילו אותם זו בזו  #
 #  מצאו  של התוצא  transpose   inverse  #
    
def ex3():
    m1=np.random.rand(4,4)
    m2=np.random.rand(4,4)
    m_multy=np.dot(m1,m2) 
    print(m_multy)
    
    m_multyT=m_multy.transpose()
    print(m_multyT)
    
    m_multy_inv=np.linalg.inv(m_multy)
    print(m_multy_inv)



    
 # חישוב רגרסיה #
 # מצאו התאמה לינארית לנקודות של y=first_array  ע"י הצבה במשוואת h הרגרסיה הלינארית שלמדנו  #
# reshape  הוא ווקטור חד מימדי, צרו ממנו ווקטור דו מימדי בעזרת פקודת x=random_floats שימו לב,  
# h=(Xt*X)^-1 *(Xt*Y)   ​  הוא הפטרון שמגדיר את קבועים של התאמה לינארית h  # 
 
def regresia(xx,yy):      #mekabel XX-matriza, Y-vektor
    assert len(xx.shape) == 2   #pkuda assert im ze le mitkaem ze iten shgia! im matriza mi meymad 2X2
    Xt=xx.transpose()
    XtX_inv=np.linalg.inv(np.dot(Xt,xx))   # h=(Xt*X)^-1 *(Xt*Y)  -ze mishvaot normaliet
    XtY=np.dot(Xt,yy)
    h=np.dot(XtX_inv, XtY)     #tozaa ze Q shel mishvaot normaliet
    return h
 
 
def ex4(random_floats, first_array):   
    # y=m*x   --> over dereh (0,0), no have bias for h
    # m=2
    # x=random_floats =np.random.rand(10)
    # y=first_array= random_floats * m
    # y=first_array += gaussian_noise
    #------------------------------------------------------------------------#
    m=2
    x=np.random.rand(10,2)  #---> new random_floats --> size (10*2)
    print(x)
    y=m*x
    print(y)
    
    gaussian_noise=np.random.normal(0,0.1, size=(10,2))
    y += gaussian_noise
    print('y: ','\n', y)
    
       
    h_reg= regresia( x, y )
    print('h_Regression:', '\n', h_reg)  #--> ze kav atama le nekudot (x,y)   parametrim h[Q1,Q2]
    #------------------------------------------------------------------------#
    
    
    x1 = random_floats.reshape(10, 1)     #--->  size (10*1)
    h_reg1= regresia(x1, first_array )
    print('h_Regression1:', '\n', h_reg1)  #--> ze kav atama le nekudot (x,y)  
                                          # parametrim h[Q1]=[2.03352429]
    
    # Plot-  plt.plot(x, y), plt.scatter(x,y)
    plt.figure(num=1)               # plt.figure(1); plt.gcf().number => 1
    plt.scatter(x1, first_array)    # scatter is graph of points (x,y) from training data, mezaer nekudat
    plt.title('Regression: random_floats by first_array')   
    plt.xlabel('X = random_floats')
    plt.ylabel('Y = first_array')  
    plt.text(0.1, 1.75, r'y=m*x')    #coordinate on graph (0.1, 1.75)=(x,y)
     
    new_x = np.linspace(0, 1, 11)  # to build new points (x,y) from new data test, nivne vektor hadash im 11 nekudot
    # ---> new_x=rray([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
    plt.plot(new_x, h_reg1[0]*new_x, color='red') #livnot yashar beezrat parametrim H she kvar hishavnu lemala ve lasim x hadashim
    plt.show()    
        
        
    
def ex5(random_floats, second_array):   
    # y=m*x+b   --> le over dereh (0,0)
    # m=2
    # b=3
    # x=random_floats =np.random.rand(10)
    # y=second_array =random_floats * m + b
    # y=second_array += gaussian_noise    
    x1 = random_floats.reshape(10, 1)   #--->  size (10*1)
    x1_ones= np.ones((10,1))            #---> vector for bias into h
    x=np.column_stack((x1_ones, x1))    #x = np.hstack((ones_vec, x1)) mehaber amudot: noten matriza amuda 1 ve amuda x1 shel data
   
    h_reg2= regresia(x, second_array )
    print('h_Regression2:', '\n', h_reg2)  #--> ze kav atama le nekudot (x,y)  
                                          # parametrim h[Q1,Q2]=[2.98332033 2.02553309]
     
        
    # Plot-  plt.plot(x, y), plt.scatter(x,y)
    plt.figure(num=2)              # plt.figure(2); plt.gcf().number => 2
    plt.scatter(x1, second_array)  # scatter is graph of points (x,y) from training data
    plt.title('Regression: random_floats by second_array')   
    plt.xlabel('X = random_floats')
    plt.ylabel('Y = second_array') 
    plt.text(0.1, 5, r'y=m*x+b')     #coordinate on graph (0.1, 5)=(x,y)
     
    new_x = np.linspace(0, 1, 11)  # to build new points (x,y) from new data test
    # ---> new_x=rray([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
    plt.plot(new_x, h_reg2[1]*new_x + h_reg2[0], color='green') #livnot yashar beezrat parametrim H ve lasim x hadashim
    plt.show()     
        
        
    
def ex6(random_floats, third_array):   
    # y= ax^2+bx+c   -->  parabola
    # a=2
    # b=1
    # c=3
    # x=random_floats =np.random.rand(10)
    # y=third_array =( random_floats**2 )*a + random_floats*b + c
    # y=third_array += gaussian_noise    
    
    x1 = random_floats.reshape(10, 1)   #--->  size (10*1)   
    x1_ones= np.ones((10,1))            #---> vector for bias into h
    x=np.column_stack((x1_ones, x1, x1**2))   #x3 = np.hstack((x1_ones, x1, x1 ** 2)) lasim 3 amudot
   
    h_reg3= regresia(x, third_array )   # x matriza hadasha im 3 amudot
    print('h_Regression3:', '\n', h_reg3)  #--> ze kav atama le nekudot (x,y)  
                                          # parametrim h[Q1,Q2,Q3]=[3.04055948 0.85626819 2.02162823]   
 
    # Plot-  plt.plot(x, y), plt.scatter(x,y)
    # plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.figure(num=3)  # plt.figure(3); plt.gcf().number => 3 
    plt.title('Regression: random_floats by third_array')   
    plt.xlabel('X = random_floats')
    plt.ylabel('Y = third_array')  
    plt.text(0.1, 5.5 , r'y=ax^2+bx+c')  #coordinate on graph (0.1, 5.5)=(x,y)
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')  coordinate on graph (60,0.025)=(x,y)
    #plt.axis([40, 160, 0, 0.03])
    plt.scatter(x1, third_array)      # scatter is graph of points (x,y) from training data
    
    new_x=np.linspace(0,1,11) # to build new points (x,y) from new data test, nivne vektor hadash 11 nekudot
    # ---> new_x=rray([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
    plt.plot(new_x, h_reg3[2]*(new_x**2) + h_reg3[1]*new_x + h_reg3[0], color='pink' )  # nasim Q she kvar hishavnu lemala h[Q1,Q2,Q3]
    plt.show()
                    
    
    
    

if __name__ == "__main__":
    random_floats = ex1()
    firsta, seconda, thirda = ex2(random_floats)
    ex3()
    ex4(random_floats, firsta)
    ex5(random_floats, seconda)
    ex6(random_floats, thirda)
    











    
