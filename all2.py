import numpy as np
from scipy.stats import norm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

option = 0
montemonte = 2
trajectory = 1000
step_size = 50
x_list = np.zeros((trajectory,step_size))
a_list = np.zeros((trajectory,step_size))
y_list = np.zeros((trajectory,step_size))
g_list = np.zeros((trajectory,step_size))
gsum_list = np.zeros((trajectory,step_size))
x = 0.0 
theta = 1.0
b1 = 1.0
b2 = -1.0 
sigma = 0.2
### Collection of Data 


##### OPE 

trajectory = 1000
step_size = 50
x_list = np.zeros((trajectory,step_size))
a_list = np.zeros((trajectory,step_size))
y_list = np.zeros((trajectory,step_size))
x = 0.0 
sigma = 0.2
### Collection of Data 
theta_b = 1.2

        
g_list = np.zeros((trajectory,step_size))
g2_list = np.zeros((trajectory,step_size))
gsum_list = np.zeros((trajectory,step_size))
g2sum_list = np.zeros((trajectory,step_size))
 

ratio_list = np.zeros((trajectory,step_size))
cumulative_raito_list = np.zeros((trajectory,step_size))

theta_list = np.zeros(montemonte)
convergence_list = np.zeros(montemonte)

for monte in range(montemonte):
    theta = 1.0+np.random.uniform(0,0.2)
    alpha = 0.1
    for i in range(trajectory):
        x = 0.0 
        for t in range(step_size):
            x_list[i,t] = x
            #### Behavior policy 
            a = theta_b*x+np.random.normal(0,sigma)
            a_list[i,t] = a
            #### Reward 
            y =-x*x
            ### Transition 
            y_list[i,t] = y
            x = b1*a+b2*x

   
    for rep in range(30):
        print theta
        ### IPW estimator
        g_list = np.zeros((trajectory,step_size))
        g2_list = np.zeros((trajectory,step_size))
        gsum_list = np.zeros((trajectory,step_size))
        g2sum_list = np.zeros((trajectory,step_size))
        ratio_list = np.zeros((trajectory,step_size))
        cumulative_raito_list = np.zeros((trajectory,step_size))
        for i in range(trajectory):
            x = 0.0 
            cute = 0.0 
            for t in range(step_size):
                x = x_list[i,t]
                a = a_list[i,t]
                ratio_list[i,t] = norm.pdf(a,theta*x, sigma)/norm.pdf(a,theta_b*x, sigma)
                g_list[i,t] = (a-theta*x)*x
                gsum_list[i,t] = np.sum(g_list[i,:])
                ###print ratio_list[i,t]
                cute = cute + np.log(norm.pdf(a,theta*x, sigma)/norm.pdf(a,theta_b*x, sigma))
                cumulative_raito_list[i,t] = cute### + np.log(norm.pdf(a,theta*x,sigma)/norm.pdf(a,x,sigma))


        ### Marginal IPW estimator 
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
                y = y_list[:,t]+y*ratio_list[:,t]
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

        ### Q Monte Carlo

        Q_estimator_derivative_list = np.zeros((trajectory,step_size))
        V_estimator_derivative_list = np.zeros((trajectory,step_size))
        for t_ in range(step_size):
            t = step_size-t_-1
            x1 = x_list[:,t]
            x2 = x_list[:,t]*x_list[:,t]
            a1 = a_list[:,t]
            a2 = a_list[:,t]*a_list[:,t]
            y  = np.zeros(trajectory)
            for j in range(t,step_size):
                y = y+y_list[:,j]*np.exp(np.sum(np.log(ratio_list[:,t:j]),1))*np.sum(g_list[:,t:j],1)
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
            ####print (np.array([xx,x1,x2,a1,a2])*coefficient[:,np.newaxis]).shape
            V_estimator_derivative_list[:,t] +=   x_list[:,t]*np.sum(np.array([xx,x1,x2,a1,a2])*coefficient[:,np.newaxis],0) 

        ##### DB estimator 

        ###Q_estimator_list = np.zeros((trajectory,step_size))
        ###V_estimator_list  = np.zeros((trajectory,step_size))
        ##Q_estimator_derivative_list = np.zeros((trajectory,step_size))
        ###V_estimator_derivative_list = np.zeros((trajectory,step_size))

        if option==0: 
            ### IPW estimator 
            ipw_list = []
            for i in range(trajectory):
                ipw_list.append(np.sum(np.exp(cumulative_raito_list[i,:])*y_list[i,:]*gsum_list[i,:]))
            ipw_list = np.array(ipw_list)
            print np.mean(ipw_list)
            gradient = np.mean(ipw_list)
            theta = theta + alpha*gradient
            
        elif option==1: 
            #### Marginal PG Gradient 
            pg_m_list = []
            auxi_list =  np.zeros((trajectory,step_size))
            for t in range(step_size):
                    auxi_list[:,t] = marginal_raito_list[:,t]*g_list[:,t]*Q_estimator_list[:,t]
            pg_m_list = np.sum(auxi_list,1)
            print np.mean(pg_m_list)

        elif option==2:
            pg_list = []
            auxi_list =  np.zeros((trajectory,step_size))
            for t in range(step_size):
                auxi_list[:,t] =np.exp(cumulative_raito_list[:,t])*g_list[:,t]*Q_estimator_list[:,t]
            pg_list = np.sum(auxi_list,1)
            print np.mean(pg_list)
            gradient = np.mean(pg_list)
            theta = theta + alpha*gradient
            
        elif option==3:
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
            print np.mean(db_estimator)
            gradient = np.mean(db_estimator)
            theta = theta + alpha*gradient 
        elif option==4: 
            auxi =  np.zeros((trajectory,step_size))
            ###marginal_derivative_list = np.zeros((trajectory,step_size))
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
            print np.mean(db_estimator)
            gradient = np.mean(db_estimator)
            ####print np.mean((db_estimator-true)*(db_estimator-true))
            theta = theta + alpha*gradient
        if np.abs(gradient)<0.01:
            break
    
    theta_list[monte] = theta
    if rep==9:
        convergence_list[monte] =1
    np.savez("option_0", x=theta_list, y= convergence_list)
    
