import numpy as np
import os
import sys
#####import matplotlib.pyplot as plt 
from scipy.stats import norm
from sklearn import linear_model
####%matplotlib inline 
from sklearn.metrics import mean_squared_error, r2_score

###args = sys.argv
###ver = np.int(args[1])
####option = np.int(args[2])
alpha = 0.15
montemonte = 100
optimzation_step = 3
trajectory =200
ver = 0 
step_size = 10
x_list = np.zeros((trajectory,step_size))
a_list = np.zeros((trajectory,step_size))
y_list = np.zeros((trajectory,step_size))
g_list = np.zeros((trajectory,step_size))
gsum_list = np.zeros((trajectory,step_size))
x = 0.0 
theta = 1.0
b1 = 1.0
b2 = -1.0 
sigma = np.power(12.0/trajectory ,0.2)
### Collection of Data 

###args = sys.argv
##ver = np.int(args[1])
###option = np.int(args[2])


x = 0.0 
sigma2 = 1.0
sigma3 = 0.03
### Collection of Data 
theta_b = 0.8

theta_mse = np.zeros([7,montemonte])

option = 0
theta_list = np.zeros((montemonte,optimzation_step))

for monte in range(montemonte):
    print monte
    x_list = np.zeros((trajectory,step_size))
    a_list = np.zeros((trajectory,step_size))
    y_list = np.zeros((trajectory,step_size))

   
    #### Data collection
    for i in range(trajectory):
        x = 0.0 
        for t in range(step_size):
            x_list[i,t] = x
            #### Behavior policy 
            a = theta_b*x+np.random.normal(0,sigma2)
            a_list[i,t] = a
            #### Reward 
            if t==step_size-1:
                y = -x*x
            else:
                y =-x*x
            ### Transition 
            y_list[i,t] = y
            x = b1*a+b2*x+np.random.normal(0,sigma3)
        
    theta = 1.0
    g_list = np.zeros((trajectory,step_size))
    g2_list = np.zeros((trajectory,step_size))
    gsum_list = np.zeros((trajectory,step_size))
    g2sum_list = np.zeros((trajectory,step_size))
    ratio_list = np.zeros((trajectory,step_size))
    cumulative_raito_list= np.zeros((trajectory,step_size))
    print theta
    ### IPW estimator
    for i in range(trajectory):
        x = 0.0 
        cute = 0.0 
        for t in range(step_size):
            x = x_list[i,t]
            a = a_list[i,t]
            ratio_list[i,t] = norm.pdf(a,theta*x, sigma)/norm.pdf(a,theta_b*x, sigma2)
            g_list[i,t] = (a-theta*x)*x*1.0/(sigma*sigma)
            gsum_list[i,t] = np.sum(g_list[i,:])
            ###print ratio_list[i,t]
            cute = cute + np.log(norm.pdf(a,theta*x, sigma)/norm.pdf(a,theta_b*x, sigma2))
            cumulative_raito_list[i,t] = cute### + np.log(norm.pdf(a,theta*x,sigma)/norm.pdf(a,x,sigma))


    ### Marginal Ratio 
    marginal_raito_list = np.zeros((trajectory,step_size))
    for t in range(step_size):
        x1 = x_list[:,t] 
        x2 = x_list[:,t]*x_list[:,t]
        x3 = x_list[:,t]*x_list[:,t]*x_list[:,t]
        x4 = x_list[:,t]*x_list[:,t]*x_list[:,t]*x_list[:,t]
        x = np.transpose(np.array([x1,x2,x3,x4]))
        if t==0:
            y = np.ones(trajectory)
        else:
            y = np.exp(cumulative_raito_list[:,t-1])
        reg = linear_model.LinearRegression()
        reg.fit(x,y)
        marginal_raito_list[:,t]= reg.predict(x)*ratio_list[:,t]

    ### Marginal Derivative 
    marginal_derivative_list = np.zeros((trajectory,step_size))
    for t in range(step_size):
        x1 = x_list[:,t] 
        x2 = x_list[:,t]*x_list[:,t]
        x3 = x_list[:,t]*x_list[:,t]*x_list[:,t]
        x4 = x_list[:,t]*x_list[:,t]*x_list[:,t]*x_list[:,t]
        x = np.transpose(np.array([x1,x2,x3,x4]))
        y = np.exp(cumulative_raito_list[:,t-1])*gsum_list[:,t-1]
        reg = linear_model.LinearRegression()
        reg.fit(x,y)
        marginal_derivative_list[:,t]= reg.predict(x)*ratio_list[:,t]+marginal_raito_list[:,t]*g_list[:,t]*ratio_list[:,t]

    marginal_derivative_list = np.zeros((trajectory,step_size))
    for t in range(step_size):
        x1 = x_list[:,t] 
        x2 = x_list[:,t]*x_list[:,t]
        x3 = x_list[:,t]*x_list[:,t]*x_list[:,t]
        x4 = x_list[:,t]*x_list[:,t]*x_list[:,t]*x_list[:,t]
        a1 = a_list[:,t]
        a2 = a_list[:,t]*a_list[:,t]
        x = np.transpose(np.array([x1,x2,x3,x4,a1,a2]))
        y = np.exp(cumulative_raito_list[:,t])*gsum_list[:,t]
        reg = linear_model.LinearRegression()
        reg.fit(x,y)
        marginal_derivative_list[:,t]= reg.predict(x)


    ### Q Monte Carlo
    Q_estimator_list = np.zeros((trajectory,step_size))
    V_estimator_list = np.zeros((trajectory,step_size))
    reg_list = []
    for t_ in range(step_size):
        t = step_size-t_-1
        x1 = x_list[:,t]
        x2 = x_list[:,t]*x_list[:,t]
        a1 = a_list[:,t]
        a2 = a_list[:,t]*a_list[:,t]
        if t_==0:
            y = y_list[:,t]
        else:
            y = y_list[:,t]+y*ratio_list[:,t+1]
        x = np.transpose(np.array([x1,x2,a1,a2]))
        reg = linear_model.LinearRegression()
        reg.fit(x,y)
        reg_list.append(np.append(reg.intercept_, reg.coef_))
        ####print r2_score(y,reg.predict(x))
        Q_estimator_list[:,t] = reg.predict(x)
        mumu = theta*x_list[:,t]
        a1 = mumu
        a2 = mumu*mumu+sigma*sigma
        x = np.transpose(np.array([x1,x2,a1,a2]))
        V_estimator_list[:,t] = reg.predict(x)
        
    ### Q Monte Carlo Derivative 

    Q_estimator_derivative_list = np.zeros((trajectory,step_size))
    V_estimator_derivative_list = np.zeros((trajectory,step_size))
    for t_ in range(step_size):
        t = step_size-t_-1
        x1 = x_list[:,t]
        x2 = x_list[:,t]*x_list[:,t]
        a1 = a_list[:,t]
        a2 = a_list[:,t]*a_list[:,t]
        y  = np.zeros(trajectory)
        if t_!=0:
            for j in range(t+1,step_size):
                y = y+y_list[:,j]*np.exp(np.sum(np.log(ratio_list[:,t+1:j+1]),1))*np.sum(g_list[:,t+1:j+1],1)
            ###y  = np.zeros(trajectory)
        x = np.transpose(np.array([x1,x2,a1,a2]))
        reg = linear_model.LinearRegression()
        reg.fit(x,y)
        ####print r2_score(y,reg.predict(x) )
        Q_estimator_derivative_list[:,t] = reg.predict(x)

        mumu = theta*x_list[:,t]
        ##### Integration 
        a1 = mumu ### a|x
        a2 = mumu*mumu +sigma*sigma ### a^2|x
        x = np.transpose(np.array([x1,x2,a1,a2]))
        V_estimator_derivative_list[:,t] += reg.predict(x)
        V_estimator_derivative_list[:,t] -= x_list[:,t]*x_list[:,t]*Q_estimator_list[:,t]

        xx =   mumu ##### a|x
        x1 =  mumu*x_list[:,t] ### ax|x
        x2 =   mumu*x_list[:,t]*x_list[:,t]   ### ax^2|x
        a1 = mumu*mumu + sigma*sigma  ###a^2|x
        a2 =  mumu*mumu*mumu +3*mumu*sigma*sigma  ###a^3|x
        coefficient = reg_list[t_]
        V_estimator_derivative_list[:,t] +=   x_list[:,t]*np.sum(np.array([xx,x1,x2,a1,a2])*coefficient[:,np.newaxis],0) 


     ### REINFORCE  

    ipw_list = []
    for i in range(trajectory):
        ipw_list.append(np.sum(np.exp(cumulative_raito_list[i,:])*y_list[i,:]*gsum_list[i,:]))
    ipw_list = np.array(ipw_list)
    gradient = np.mean(ipw_list)
    print(gradient)
    theta_mse[0,monte] =  gradient
    ### PG1


    pg_list = []
    for i in range(trajectory):
        pg_list.append(np.sum(np.exp(cumulative_raito_list[i,:])*g_list[i,:]*Q_estimator_list[i,:]))
    pg_list = np.array(pg_list)
    gradient = np.mean(pg_list)
    print(gradient)
    #####theta_mse[0,monte] =  gradient
    theta_mse[1,monte] =  gradient
    ###  Cumulative PG 
    
    ####elif option==3:

    auxi =  np.zeros((trajectory,step_size))
    for t in range(step_size):
        if t !=0:
            auxi[:,t] +=  np.exp(cumulative_raito_list[:,t])*gsum_list[:,t]*(y_list[:,t]-Q_estimator_list[:,t])
            auxi[:,t] +=  -np.exp(cumulative_raito_list[:,t])*Q_estimator_derivative_list[:,t]
            auxi[:,t] +=  np.exp(cumulative_raito_list[:,t-1])*V_estimator_derivative_list[:,t]
            auxi[:,t] +=  np.exp(cumulative_raito_list[:,t-1])*gsum_list[:,t-1]*V_estimator_list[:,t]
        else:
            auxi[:,t] +=  np.exp(cumulative_raito_list[:,t])*gsum_list[:,t]*(y_list[:,t]-Q_estimator_list[:,t])
            auxi[:,t] +=  -np.exp(cumulative_raito_list[:,t])*Q_estimator_derivative_list[:,t]                                                                                                                          
            auxi[:,t] +=  V_estimator_derivative_list[:,t]
    db_estimator = np.sum(auxi,1)
    gradient = np.mean(db_estimator)
    print(gradient)
    theta_mse[2,monte] =  gradient
    
    ###  Marginal PG    
    ####elif option==4: 
    auxi =  np.zeros((trajectory,step_size))
    for t in range(step_size):
        if t !=0:
            auxi[:,t] +=  marginal_derivative_list[:,t]*(y_list[:,t]-Q_estimator_list[:,t])
            auxi[:,t] +=  -marginal_raito_list[:,t]*Q_estimator_derivative_list[:,t]
            auxi[:,t] +=  marginal_raito_list[:,t-1]*V_estimator_derivative_list[:,t]
            auxi[:,t] +=  marginal_derivative_list[:,t-1]*V_estimator_list[:,t]
        else:
            auxi[:,t] +=  marginal_derivative_list[:,t]*(y_list[:,t]-Q_estimator_list[:,t])
            auxi[:,t] +=  -marginal_raito_list[:,t]*Q_estimator_derivative_list[:,t]                                                                                                                          
            auxi[:,t] +=  V_estimator_derivative_list[:,t]
    db_estimator = np.sum(auxi,1)
    gradient = np.mean(db_estimator)
    print(gradient)
    ####theta = theta + alpha*gradient
    ###gradient = np.mean(db_estimator)
    theta_mse[3,monte] =  gradient
    
    ####elif option==5:
    auxi =  np.zeros((trajectory,step_size))
    for t in range(step_size):
        auxi[:,t] +=  marginal_derivative_list[:,t]*y_list[:,t]   
    db_estimator = np.sum(auxi,1)
    gradient = np.mean(db_estimator)
    print(gradient)
    ####theta = theta + alpha*gradient
    theta_mse[4,monte] =  gradient
    
    ####elif option==6:
    auxi =  np.zeros((trajectory,step_size))
    gradient = np.mean(V_estimator_derivative_list[:,0])
    print(gradient)
    ###theta = theta + alpha*gradient
    theta_mse[5,monte] =  gradient
    
    ####elif option ==7:
    auxi =  np.zeros((trajectory,step_size))
    for t in range(step_size):
        auxi[:,t] +=  marginal_raito_list[:,t]*Q_estimator_list[:,t]*g_list[:,t]
    ##db_ehttp://localhost:9006/notebooks/PG_simulation/Untitled1.ipynb?kernel_name=python2#stimator = np.sum(auxi,1)
    db_estimator = np.sum(auxi,1)
    gradient = np.mean(db_estimator)
    print(gradient)
    ####theta = theta + alpha*gradient
    theta_mse[6,monte] =  gradient
        

        
    np.savez("result/MSE_list4",x=theta_mse)
    
    