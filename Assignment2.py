#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from datetime import datetime
from sympy import solve, symbols, Eq
from math import sin, cos, pi, acos, fabs
from numpy.linalg import norm
import scipy.optimize as optimize

l=symbols("l")
k=symbols("k")
j=symbols("j")


# In[2]:


def get_times(data):
    data[0]=data[0]+400
    second=[]
    for i in range(len(data)):
        date=data.iloc[i][[0,1,2,3,4]]
        a=datetime(date[0],date[1],date[2],date[3],date[4])
        second.append(a.timestamp())
    second=np.array(second)-second[0]
    days=(second/86400)
    return days


# In[3]:


def get_oppositions(data):
    Angles=[]
    for i in range(len(data)):
        ZodiacIndex=data.iloc[i][5]
        Degree=data.iloc[i][6]
        Minute=data.iloc[i][7]
        Second=data.iloc[i][8]
        angle=ZodiacIndex*30 + Degree + Minute/60 + Second/3600
        Angles.append(angle)
    return Angles


# In[4]:


def MarsEquantModel(variables,r,s,days,oppositions):
    c=variables[0]*(pi/180)
    e1=variables[1]
    e2=variables[2]*(pi/180)
    z=variables[3]
    angle_e=(s*days)%360
    center=np.array([cos(c),sin(c)])
    equant=np.array([e1*cos(e2),e1*sin(e2)])
    ce=equant-center
    Error=[]
    for i in range (len(oppositions)):
        theta=pi*(z+angle_e[i])/180
        l_dist=solve(l**2+2*l*(ce[0]*cos(theta)+ce[1]*sin(theta))+(norm(ce)**2-r**2),l)
        estimated_pos=equant+np.array([l_dist[1]*cos(theta),l_dist[1]*sin(theta)])
        estimated_pos=np.array(estimated_pos, dtype=float)
        #print(estimated_pos)
        alpha=acos(estimated_pos[0]/norm(estimated_pos))*180/pi
        if(estimated_pos[1]<0):
            alpha=360-alpha
        #print(alpha)
        error=fabs(oppositions[i]-alpha)
        Error.append(error)
    #print("max error is",max(Error))
    return Error, max(Error)


# In[5]:


def maxerror(variables,r,s,days,oppositions):
    e,m=MarsEquantModel(variables,r,s,days,oppositions)
    return m


# In[6]:


def bestOrbitInnerParams(variables,r,s,days,oppositions):
    resbrute = optimize.minimize(maxerror,(variables),args=(r,s,days,oppositions),method='Nelder-Mead')
    return resbrute.x


# In[7]:


def bestS(variables,r,days,oppositions):
    s_min=(360/688)
    s_max=(360/686)
    old_error=maxerror(variables,r,(360/687),days,oppositions)
    s_opt=360/687
    for n in range(3):
        Ei=[]
        si=np.linspace(s_min,s_max,100)
        for i in range(100):
            ei=maxerror(variables,r,si[i],days,oppositions)
            Ei.append(ei)
        new_error=np.min(Ei)
        if(new_error<old_error):
            min_index=np.argmin(Ei)
            s_opt=si[min_index]
            s_max=si[min_index+2]
            s_min=si[min_index-2]
            old_error=new_error
    return s_opt


# In[8]:


def bestR(variables,s,days,oppositions):
    c=variables[0]*(pi/180)
    e1=variables[1]
    e2=variables[2]*(pi/180)
    z=variables[3]
    angle_e=(s*days)%360
    center=np.array([cos(c),sin(c)])
    equant=np.array([e1*cos(e2),e1*sin(e2)])
    all_r=[]
    for i in range(len(oppositions)):
        theta=(z+angle_e[i])
        #print(theta)
        eq1=Eq(k*cos(oppositions[i]*(pi/180))-j*cos(theta*(pi/180)),equant[0])
        eq2=Eq(k*sin(oppositions[i]*(pi/180))-j*sin(theta*(pi/180)),equant[1])
        #print(eq1,eq2)
        k_solved=solve((eq1, eq2), (k, j))[k]
        k_solved=float(k_solved)
        r=norm(k_solved*np.array([cos(oppositions[i]*(pi/180)),sin(oppositions[i]*(pi/180))])-center)
        #print(r)
        all_r.append(r)
    r_min=min(all_r)
    r_max=max(all_r)
    print("min and max r is",r_min,r_max)
    r_opt= np.mean(all_r)
    old_error=maxerror(variables,r_opt,s,days,oppositions)
    for n in range(3):
        Ei=[]
        ri=np.linspace(r_min,r_max,100)
        for i in range(100):
            ei=maxerror(variables,ri[i],s,days,oppositions)
            Ei.append(ei)
        new_error=np.min(Ei)
        if(new_error<old_error):
            min_index=np.argmin(Ei)
            r_opt=ri[min_index]
            r_max=ri[min_index+2]
            r_min=ri[min_index-2]
            old_error=new_error
    return r_opt


# In[9]:


'''runed brute search for variables that is [c,e1,e2,z] and then comed with initial guess value for variables. keeping s fixed and 
searched for r can give u the intution that r lies around 10 these guess values are used to generate parameters values randomly.
'''
'''
r=9
s=360/687
variables=[155, 1.5, 150, 55]
'''
def bestMarsOrbitParams(days, oppositions):
    #rranges = (slice(0,180,5), slice(1,2,0.5), slice(0,180,10), slice(50,70,5))
    #resbrute = optimize.brute(maxerror, rranges,args=(r,s,days,oppositions),full_output=True,finish=None)
    #variables=resbrute[0]
    variables=[155,1.5,150,55]
    print("initial guess of variables is",variables)
    ci=variables[0]
    e1i=variables[1]
    e2i=variables[2]
    zi=variables[3]
    Errors=[]
    parameters=[]
    for i in range(50):
        print("this is trial",i)
        r=np.random.uniform(8,10)
        print("r is",r)
        s=np.random.uniform((360/688),(360/686))
        print("s is",s)
        c=np.random.uniform(ci-5,ci+5)
        e1=np.random.uniform(e1i-0.1,e1i+0.1)
        e2=np.random.uniform(e2i-2,e2i+2)
        z=np.random.uniform(zi-2,zi+2)
        variables=[c,e1,e2,z]
        print("variables are",variables)
        for n in range(2):
            variables=bestOrbitInnerParams(variables,r,s,days,oppositions)
            print("modified variables are",variables)
            s=bestS(variables,r,days,oppositions)
            print("modified s is",s)
            r=bestR(variables,s,days,oppositions)
            print("modified r is",r)
            
        error=maxerror(variables,r,s,days,oppositions)
        print("error after 1 iteration is",error)
        Errors.append(error)
        parameters.append([variables,r,s])
        if(error<0.07):
            break
    best_param=parameters[np.argmin(Errors)]
    variables=best_param[0]
    r=best_param[1]
    s=best_param[2]
    error,max_error=MarsEquantModel(variables,r,s,days,oppositions)
    
    return r,s,variables[0],variables[1],variables[2],variables[3],error,max_error


# In[10]:


if __name__ == "__main__":

    # Import oppositions data from the CSV file provided
    data = np.genfromtxt(
        "../data/01_data_mars_opposition_updated.csv",
        delimiter=",",
        skip_header=True,
        dtype="int",
    )

    data = pd.DataFrame(data)
    # Extract times from the data in terms of number of days.
    # "times" is a numpy array of length 12. The first time is the reference
    # time and is taken to be "zero". That is times[0] = 0.0
    times = get_times(data)
    assert len(times) == 12, "times array is not of length 12"

    # Extract angles from the data in degrees. "oppositions" is
    # a numpy array of length 12.
    oppositions = get_oppositions(data)
    assert len(oppositions) == 12, "oppositions array is not of length 12"

    # Call the top level function for optimization
    # The angles are all in degrees
    r, s, c, e1, e2, z, errors, maxError = bestMarsOrbitParams(
        times, oppositions
    )

    assert max(list(map(abs, errors))) == maxError, "maxError is not computed properly!"
    print(
        "Fit parameters: r = {:.4f}, s = {:.4f}, c = {:.4f}, e1 = {:.4f}, e2 = {:.4f}, z = {:.4f}".format(
            r, s, c, e1, e2, z
        )
    )
    print("The maximum angular error = {:2.4f}".format(maxError))


# In[ ]:




