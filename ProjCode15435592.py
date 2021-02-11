# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 18:32:00 2018

@author: ryanr
"""

import pandas as pd 
import numpy as np 
import numpy.polynomial.polynomial as nppp 
import scipy.stats as sps 
from pylab import plot,show 
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans,vq 
from pylab import plot,show

from sklearn.model_selection import train_test_split

#%% pseudo1
DataFrame   = pd.read_csv('file:///C:/Users/ryanr/OneDrive/Desktop/College/PythonModule/Project/ProjectDataCSV.csv',header=None) 

y=DataFrame.iloc[:,13]

DataFrame=DataFrame.iloc[:,:6]      #pseudo3
DataFrame=DataFrame.join(y)

DataMatrix  = DataFrame.as_matrix()

x=DataMatrix[1:,:5]
y=DataMatrix[1:,6]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)       #pseudo4
#train=X_train.join(y_train)
#test=X_train.join(y_test)
X_train=X_train.astype(np.float)
y_train=y_train.astype(np.float)

#pseudo5
X_train=skp.MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(X_train)

y_train=skp.MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(y_train.reshape(-1, 1))



X_test=skp.MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(X_test)

y_test=skp.MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(y_test.reshape(-1, 1))


#%% Define f(x)

def f(x):
    #x=float(x)
    y=1/(1+np.exp(-x))
    return y


#%% pseudo2
N=10
P=5     #pseudo6 (PN=50)


#%% Create Wpns pseudo7

Npop = np.random.uniform(-1,1, (500, P, N))


#%% Define EQ1 
#               X=Xdata,
#               W= originally 500 matrices (5 rows 10 columns) of Wpn values between -1 and 1
#               output= desired name for output of Yhat values

def eq1(X,W):
    output=np.array([],dtype=float)
    for i in range(0,(len(W))):
        exw=np.matmul(X,W[i])
        nexw=np.sum(f(exw),axis=1)
        output=np.append(output,nexw)
    return output;

NpopYhat=eq1(X_train,Npop)


#%% Fitness define              pseudo8
def fitness(NpopYhat,y_train):
    output=np.array([],dtype=float)
    if (len(NpopYhat)/189)!=int(len(NpopYhat)/189):
        print("Invalid length")
        return;
    else:
        nrows=int(len(NpopYhat)/189)
        np.shape(NpopYhat)
        NpopYhat=NpopYhat.reshape((nrows,189))
        for i in range (0,(nrows)):
            numerator=(y_train-(NpopYhat[i,:]))
            numeratorsqr=np.square(numerator)
            numsum=np.sum(numeratorsqr)
            output=np.append(output,( (1-(numsum/189))*100 )   )
        return output;

#nrows=int(len(NpopYhat)/189)
#np.shape(NpopYhat)
#NpopYhat=NpopYhat.reshape((nrows,189))

#Kev=(y_train-(NpopYhat[0,:]))
#np.sum(Kev)
#NpopYhat[0]
fit=fitness(NpopYhat,y_train)

#%%SELECT BEST FIT (STUD)      fit into next function
#   pseudo9 another part to get the string later!
studindex=np.argmax(fit) #returns the first max value, if another chromosome produces a fitness value equal to it's parent's fitness value, I give preference to the parent chromosome and continue to use it as the parent for the next iteration
#studindex
ListOfFitnessValues=list()

ListOfFitnessValues.append(fit[studindex])

#
#
#
#

#%% Binarize function           pseudo10 and pseudo11
def binarize(rm):
   rm=rm.astype(np.float)
   rm=((rm+1)/2)
   
   rm=1000*rm
   rm=np.around(rm)
   rm=rm.astype(int)
   
   rm=rm.flatten()
   rm=rm.astype(int)
   
   rm2=np.array([],dtype=int)
   for x in rm:
       x = np.binary_repr(x).zfill(10)
       rm2=np.append(rm2,x)
   chromo = ''.join(map(str, rm2))                  #pseudo11
   return chromo;


#%% Binarise population and find "stud" - Best fitness chromosome which will become parent
binNpop=np.array([],dtype=int)
for x in Npop:
    binNpop=np.append(binNpop,binarize(x))
#pseudo9
stud=binNpop[studindex]

ListOfStuds=list() #this will be used later

ListOfStuds.append(stud)

#%%
#
#  binNpop to next function
#
#   STUD TO NEXT FUNCTION, CHOOSE STUD?
#
#%%crossover  pseudo12

def babies(stud,studindex,binNpop):     #returns children with parent repeated as first element (is repeated in children twice more)
    #cross=np.random.randint(2,(len(stud)-1))
    
    children=np.array([],dtype=str)
    children=np.append(children,stud)
    
    for x in binNpop:
        cross=np.random.randint(2,(len(stud)-1))
        child1=stud[:cross]+(x[cross:])
        child2=(x[:cross])+stud[cross:]
        children=np.append(children,child1)
        children=np.append(children,child2)
    return children;
        

fam=babies(stud,studindex,binNpop)


#%%
#
#  FAM TO NEXT FUNCTION MUTATE
#
#
#%%Mutate function
# pseudo13
def mutate(chromosome):
    mutateindex=np.random.randint(0,500,25)
    mutateindex=np.sort(mutateindex.astype(int))
    
    chromosome=str(chromosome)
    chromosome=np.array(list(chromosome))
    #chromosome=chromosome[1:]
    #chromosome=chromosome[:-1]
    
    for x in mutateindex:
        if int(chromosome[x])==1:
            chromosome[x]=0
        elif int(chromosome[x])==0:
            chromosome[x]=1
        else:
            print("chromosome format error")
    
    new=""
    for x in chromosome:
        new=new+(x)
    new=str(new)
        
    return new;



mutated = [mutate(chromo) for chromo in fam[1:]] #The first entry is the sire. sire is repeated in the fam data again. I preserve one of the original versions of sire by using fam[1:] as this function then operates on all elements after the first. I want to preserve one of the original copies in case mutating sire makes it less accurate



#%%
#
# MUTATED TO NEXT FUNCTION
#
#

#%%Chromosomes to wpns

#both functions help complete pseudo14
   
def ChroToDeBinarisedArray(chromosome):
    chromosome=str(chromosome)
    chromosome=np.array(list(chromosome))
    #Make array with 10 bit rows to allow de-binarisation
    chromosome=chromosome.reshape((1,500))
    chromosome=chromosome.reshape((50,10))
    
    ##########################
    blank=list("")

    nrows=len(chromosome)
    ncols=len(chromosome[0,])

    for i in range(0,(nrows)):
        blank.append("")
        for j in range(0,(ncols)):
            blank[i]=blank[i]+str(chromosome[i,j])
        blank[i]=((int(blank[i],2))/1000)
    
    return blank;



def PopnToNormalisedMatrices(fam): #fam is a list of chromosomes (Will probably be the mutated children chromosomes)
    listed=list()  
    for i in range(0,(len(fam))):
        listed.append(ChroToDeBinarisedArray(fam[i]))
    #listed=np.array(listed)  
    
    finmatrices=list()
    for k in range(0,(int(len(fam)))):
        matrix=np.reshape(listed[k],(5,10))
        finmatrices.append(matrix)
    return finmatrices;
    

# De-normalise population
newPop=PopnToNormalisedMatrices(fam)

orig=list()
for x in newPop:
    x=((2*x)-1)
    orig.append(x)

newPop=np.array(orig)

#pseudo14 done

#%% start an iteration
for u in range(0,30):
    #at the start of the next iteration pseudo15 resolved
    newPopYhat=eq1(X_train,newPop)
    newfit=fitness(newPopYhat,y_train)

    outsidestudindex=np.argmax(newfit)
    ListOfFitnessValues.append(newfit[np.argmax(newfit)])


    inds = newfit.argsort()
    inds=list(reversed(inds))
    inds=np.array(inds)
    sorty=list()
    for x in inds:
        val=int(x)
        ex=(newPop[val])
        sorty.append(ex)
    
    

    newSortedYhat=eq1(X_train,sorty)
    sortedfit=fitness(newSortedYhat,y_train)

    newfit=sortedfit        #pseudo15
    newPopYhat=newSortedYhat
    newPop=np.array(sorty)
    
    len(newSortedYhat)
    len(sorty)
    len(newPop)    
    newstudindex=np.argmax(sortedfit) #returns the first max value, if another chromosome produces a fitness value equal to it's parent's fitness value, I give preference to the parent chromosome and continue to use it as the parent for the next iteration
    

    # Binarise population and find "stud" - Best fitness chromosome which will become parent
    binNewPop=np.array([],dtype=int)
    for x in newPop[:500]:              #pseudo16
        binNewPop=np.append(binNewPop,binarize(x))

    newstud=binNewPop[newstudindex]         #pseudo17  last parent is still in BinNewPop, so If its fitness score is better it will remain as the parent for the next iteration
    
    ListOfStuds.append(newstud)
    

    newfam=babies(newstud,newstudindex,binNewPop)
    
    

    newmutated = [mutate(chromo) for chromo in newfam[1:]] #The first entry is the sire. sire is repeated in the fam data again. I preserve one of the original versions of sire by using fam[1:] as this function then operates on all elements after the first. I want to preserve one of the original copies in case mutating sire makes it less accurate



    nextPop=PopnToNormalisedMatrices(newmutated)
    #de-normalise
    orig=list()
    for x in nextPop:
        x=((2*x)-1)
        orig.append(x)

    nextPop=np.array(orig)
    newPop=np.array([])
    newPop=nextPop





#Here I test my best result for Wpn against the y_test data and find the associated error

Wpn=np.reshape(ChroToDeBinarisedArray(newstud),(5,10))

    
exw=np.matmul(X_test,Wpn)    
FinalYhat=(np.sum(f(exw),axis=1))


y_test=list(y_test)
FinalYhat=list(FinalYhat)
err=[0]*63
for i in range(0,63):
    err[i]=y_test[i]-FinalYhat[i]
sumsqr=0
for x in err:
    sumsqr=sumsqr+np.square(x)
print("final error for test data is:",(sumsqr/63))
    


amb_plot = plt.scatter(range(0,len(ListOfFitnessValues)),ListOfFitnessValues)
plt.title('Plot of Highest Fitness Values For Each Iteration')
plt.xlabel('Iteration')
plt.ylabel('Fitness Value')



from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax1=fig.add_subplot(111, projection='3d')

x=range(0,len(y_test))
NpopYhat=list(NpopYhat[0:63])
y_test=list(y_test)
ax1.set_xlabel("x range")
ax1.set_ylabel("Yhat values")
ax1.set_zlabel("Real y values")


ax1.scatter(x,NpopYhat,y_test)
#This is my 3d plot (at the end of project guide)
