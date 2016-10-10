
# coding: utf-8

# In[63]:

import json
import numpy as np
from collections import defaultdict
import scipy.misc
from copy import deepcopy
import math


# In[4]:

get_ipython().magic('pylab inline')


# In[5]:

selection = ['Pittsburgh']
cutoff = 9


# In[6]:

# sel_businesses存了所有匹兹堡餐馆的客户评分，少于10个评分的去掉
businesses = []
star = dict()
for line in open('E:/2016Spring/NW/Project/yelp_academic_dataset_business.json',encoding='utf-8'):
    businesses.append(json.loads(line))
sel_businesses = dict()
for bus in businesses:
    if (bus['city'] in selection) and (bus['review_count']>cutoff) and ('Restaurants' in bus['categories']):
        sel_businesses[bus['business_id']] = bus['review_count']
        star[bus['business_id']] = bus['stars']


# In[7]:

#business_star star for each restaurant
business_star = np.zeros((len(sel_businesses)))
for ind,business in enumerate(star.keys()):
    business_star[ind] = star[business]


# In[8]:

# user_reviews存了所有匹兹堡客户对餐馆评分，少于10个评分的去掉得到users
user_reviews = defaultdict(dict)
for line in open('E:/2016Spring/NW/Project/yelp_academic_dataset_review.json',encoding='utf-8'):
    l = json.loads(line)
    if l['business_id'] in sel_businesses:
        user_reviews[l['user_id']][l['business_id']] = l['stars']
users = defaultdict(dict)
for ind, user in enumerate(user_reviews.keys()):
    if len(user_reviews[user].keys())>cutoff:
        users[user] = user_reviews[user]


# In[9]:

user_friend = []
for line in open('E:/2016Spring/NW/Project/yelp_academic_dataset_user.json',encoding='utf-8'):
    user_friend.append(json.loads(line))
sel_user_friend = dict()
for temp in user_friend:
    if(temp['user_id'] in users.keys()):
        sel_user_friend[temp['user_id']] = temp['friends'] 


# In[10]:

sel_user_friend2 = dict()
for i,user in enumerate(users.keys()):
    sel_user_friend2[i] = sel_user_friend[user] 


# In[11]:

user_num = list(enumerate(users.keys()))


# In[12]:

#the social network
friend = defaultdict(list)
for i in range(len(users)):
    for user in sel_user_friend2[i]:
        for j in range(len(users)):
            if user == user_num[j][1]:
                friend[i].append(j)


# In[13]:

# 看看原来有多少评分
sum_r = 0
for ind,user in enumerate(user_reviews.keys()):
    sum_r = sum_r+len(user_reviews[user].keys())
sum_r


# In[14]:

# users中的评分总数
sum_r = 0
for ind,user in enumerate(users.keys()):
    sum_r = sum_r+len(users[user].keys())
sum_r


# In[15]:

# 所有评分分布
rate_sum = np.zeros(5)
for ind,user in enumerate(users.keys()):
    for i in users[user]:
        rate_sum[users[user][i]-1] = rate_sum[users[user][i]-1]+1
plt.bar(range(1,np.size(rate_sum)+1),rate_sum/np.sum(rate_sum),0.5)
#standard gaussian
gaussian = np.around(np.random.normal(3,1,1000))
count = np.zeros(5)
for i in range(1000):
    if gaussian[i]>=1 and gaussian[i]<=5:
        count[gaussian[i]-1] +=1
plt.bar(range(1,6),count/np.sum(count),0.2,color='red')

plt.title('the distribution of reviews')
plt.xlabel('stars')
plt.ylabel('percentage')
plt.legend(['rated dist.','gaussian'],loc=2)


# In[16]:

# rating表示 客户-餐馆 评分矩阵
rating = np.zeros((len(users),len(sel_businesses)))
business_num = defaultdict(dict)
for i1,business in enumerate(sel_businesses.keys()):
    business_num[business] = i1
for i2,user in enumerate(users.keys()):
    for i3 in users[user]:
        rating[i2,business_num[i3]] = users[user][i3]


# In[17]:

# r_cate表示所有categories
restaurant = []
restaurant_count = 0
restaurant_id = []
with open('E:/2016Spring/NW/Project/yelp_academic_dataset_business.json',encoding='utf-8') as fin:
    for line in fin:
        line_contents = json.loads(line)
        temp = line_contents['categories']
        if line_contents['review_count']>4:
            for x in temp:
                if x == 'Restaurants':
                    restaurant_id.append(line_contents['business_id'])
                    restaurant=restaurant+line_contents['categories']
                    restaurant_count = restaurant_count+1
                #restaurant.append({'business_id':line_contents['business_id'],'categories':line_contents['categories']})
                    break
r_cate = np.unique(restaurant)
r_cate = [x for x in r_cate if x!='Restaurants'] 
r_cate = np.asarray(r_cate)
s_r_cate = size(r_cate)


# In[18]:

r_exist = np.ceil(rating/10)
#split into training set and test set with ratio 0.9:0.1
split1 = np.random.choice(2,size=np.shape(r_exist),p=[0.1,0.9])


# In[19]:

w_friend = dict()
for i in range(len(users)):
    w_friend[i] = np.zeros(len(friend[i])+1)
    w_friend[i][0] = 1


# In[20]:

# 原版
x = np.ones((len(users),s_r_cate))#初始化x
lambda0 = 0.1
eta = 0.1
r_training =  r_exist*split1
#初始化theta
theta = np.zeros((len(sel_businesses),s_r_cate))
for bus in businesses:
    if (bus['city'] in selection) and (bus['review_count']>cutoff) and ('Restaurants' in bus['categories']):
        #pdb.set_trace()
        cate_num = len(bus['categories'])-1
        if cate_num==0:
            theta[business_num[bus['business_id']],:] = 1/s_r_cate*bus['stars']
        else:
            for i in bus['categories']:
                if i != 'Restaurants':
                    theta[business_num[bus['business_id']],find(r_cate==i)[0]] = 1/cate_num*bus['stars']
w_friend = np.zeros((len(users),len(users)))
for i in range(len(users)):
    w_friend[i,i] = 1

#pdb.set_trace()
for i0 in range(80):
    r_esti2 = np.dot(x,np.transpose(theta))
    r_esti2[r_esti2>5] = 5
    r_esti2[r_esti2<1] = 1
    
    for i in range(len(users)):
        w_friend[i,i] = np.dot(x[i,:],x[i,:])
        for j in range(len(friend[i])):
            w_friend[i,friend[i][j]] = np.dot(x[i,:],x[friend[i][j],:])
        w_friend[i,:] = w_friend[i,:]/np.sum(w_friend[i,:])
    
    for i1,user in enumerate(users.keys()):
        temp1 = 2*(r_esti2[i1,:]-rating[i1,:])*r_training[i1,:]/np.max([np.sum(r_training[i1,:]),1])#/np.sum(r_exist[i1,:])
        x[i1,:] = x[i1,:]-eta*(np.dot(temp1,theta)+lambda0*x[i1,:])
    for i2,bus in enumerate(sel_businesses.keys()):
        #pdb.set_trace()
        temp2 = 2*(r_esti2[:,i2]-rating[:,i2])*r_training[:,i2]/np.max([np.sum(r_training[:,i2]),1])#/np.sum(r_exist[:,i2])
        theta[i2,:] = theta[i2,:]-eta*(np.dot(temp2,x)+lambda0*theta[i2,:])


# In[21]:

loc=np.where(np.logical_and(rating*(1-split1)*r_exist<3, rating*(1-split1)*r_exist>0))
err2 = r_esti2[loc[0],loc[1]]-rating[loc[0],loc[1]]
[np.sum(np.abs(err2))/len(err2),size(np.where(np.abs(err2)<=0.5))/len(err2)]


# In[24]:

# 原版加好友
x = np.ones((len(users),s_r_cate))#初始化x
lambda0 = 0.1
eta = 0.1
r_training =  r_exist*split1
#初始化theta
theta = np.zeros((len(sel_businesses),s_r_cate))
for bus in businesses:
    if (bus['city'] in selection) and (bus['review_count']>cutoff) and ('Restaurants' in bus['categories']):
        #pdb.set_trace()
        cate_num = len(bus['categories'])-1
        if cate_num==0:
            theta[business_num[bus['business_id']],:] = 1/s_r_cate*bus['stars']
        else:
            for i in bus['categories']:
                if i != 'Restaurants':
                    theta[business_num[bus['business_id']],find(r_cate==i)[0]] = 1/cate_num*bus['stars']
w_friend = np.zeros((len(users),len(users)))
for i in range(len(users)):
    w_friend[i,i] = 1

#pdb.set_trace()
for i0 in range(80):
    r_esti3 = np.dot(np.dot(w_friend,x),np.transpose(theta))#np.transpose(w_friend)
    r_esti3[r_esti3>5] = 5
    r_esti3[r_esti3<1] = 1
    
    for i in range(len(users)):
        w_friend[i,i] = np.dot(x[i,:],x[i,:])
        for j in range(len(friend[i])):
            w_friend[i,friend[i][j]] = np.dot(x[i,:],x[friend[i][j],:])
        w_friend[i,:] = w_friend[i,:]/np.sum(w_friend[i,:])
    
    for i1,user in enumerate(users.keys()):
        temp1 = 2*(r_esti3[i1,:]-rating[i1,:])*r_training[i1,:]/np.max([np.sum(r_training[i1,:]),1])
        x[i1,:] = x[i1,:]-eta*(np.dot(np.sum(w_friend[:,i])*temp1,theta)+lambda0*x[i1,:])
    for i2,bus in enumerate(sel_businesses.keys()):
        temp2 = 2*(r_esti3[:,i2]-rating[:,i2])*r_training[:,i2]/np.max([np.sum(r_training[:,i2]),1])
        theta[i2,:] = theta[i2,:]-eta*(np.dot(temp2,x)+lambda0*theta[i2,:])


# In[171]:

loc=np.where(np.logical_and(rating*(1-split1)*r_exist<3, rating*(1-split1)*r_exist>0))
err2 = r_esti3[loc[0],loc[1]]-rating[loc[0],loc[1]]
[np.sum(np.abs(err2))/len(err2),size(np.where(np.abs(err2)<=0.5))/len(err2)]


# In[22]:

# 修改版
x = np.ones((len(users),s_r_cate))#初始化x
lambda0 = 0.1
eta = 0.1
rating_prb = 4
rating_offset = 1
r_training =  r_exist*split1
#初始化theta
theta = np.zeros((len(sel_businesses),s_r_cate))
for bus in businesses:
    if (bus['city'] in selection) and (bus['review_count']>cutoff) and ('Restaurants' in bus['categories']):
        #pdb.set_trace()
        cate_num = len(bus['categories'])-1
        if cate_num==0:
            theta[business_num[bus['business_id']],:] = 1/s_r_cate*bus['stars']
        else:
            for i in bus['categories']:
                if i != 'Restaurants':
                    theta[business_num[bus['business_id']],find(r_cate==i)[0]] = 1/cate_num*bus['stars']
#pdb.set_trace()
for i0 in range(80):
    r_esti = np.dot(x,np.transpose(theta))
    r_esti[r_esti>5] = 5
    r_esti[r_esti<1] = 1
    
    
    for i1,user in enumerate(users.keys()):
        temp1 = 2*(1-1/(2*rating_prb)*(r_esti[i1,:]-rating[i1,:]))*(r_esti[i1,:]-rating[i1,:])*np.exp(-(r_esti[i1,:]-rating_offset)/rating_prb)*r_training[i1,:]
        #temp1 = temp1/np.sum(r_training)
        temp1 = temp1/np.max([np.sum(r_training[i1,:]*np.exp(-(r_esti[i1,:]-rating_offset)/rating_prb)),1])
        x[i1,:] = x[i1,:]-eta*(np.dot(temp1,theta)+lambda0*x[i1,:])
    for i2,bus in enumerate(sel_businesses.keys()):
        #pdb.set_trace()
        temp2 = 2*(1-1/(2*rating_prb)*(r_esti[:,i2]-rating[:,i2]))*(r_esti[:,i2]-rating[:,i2])*np.exp(-(r_esti[:,i2]-rating_offset)/rating_prb)*r_training[:,i2]
        #temp2 = temp2/np.sum(r_training)
        temp2 = temp2/np.max([np.sum(r_training[:,i2]*np.exp(-(r_esti[:,i2]-rating_offset)/rating_prb)),1])
        theta[i2,:] = theta[i2,:]-eta*(np.dot(temp2,x)+lambda0*theta[i2,:])


# In[23]:

loc=np.where(np.logical_and(rating*(1-split1)*r_exist<3, rating*(1-split1)*r_exist>0))
err1 = r_esti[loc[0],loc[1]]-rating[loc[0],loc[1]]
[np.sum(np.abs(err1))/len(err1),size(np.where(np.abs(err1)<=0.5))/len(err1)]


# In[28]:

# 修改版加好友
x = np.ones((len(users),s_r_cate))#初始化x
lambda0 = 0.1
eta = 0.1
rating_prb = 4
rating_offset = 1
r_training =  r_exist*split1
#初始化theta
theta = np.zeros((len(sel_businesses),s_r_cate))
for bus in businesses:
    if (bus['city'] in selection) and (bus['review_count']>cutoff) and ('Restaurants' in bus['categories']):
        #pdb.set_trace()
        cate_num = len(bus['categories'])-1
        if cate_num==0:
            theta[business_num[bus['business_id']],:] = 1/s_r_cate*bus['stars']
        else:
            for i in bus['categories']:
                if i != 'Restaurants':
                    theta[business_num[bus['business_id']],find(r_cate==i)[0]] = 1/cate_num*bus['stars']
w_friend = np.zeros((len(users),len(users)))
for i in range(len(users)):
    w_friend[i,i] = 1
#pdb.set_trace()
for i0 in range(80):
    r_esti = np.dot(x,np.transpose(theta))
    r_esti[r_esti>5] = 5
    r_esti[r_esti<1] = 1
    
    for i in range(len(users)):
        w_friend[i,i] = np.dot(x[i,:],x[i,:])
        for j in range(len(friend[i])):
            w_friend[i,friend[i][j]] = np.dot(x[i,:],x[friend[i][j],:])
        w_friend[i,:] = w_friend[i,:]/np.sum(w_friend[i,:])
    
    for i1,user in enumerate(users.keys()):
        temp1 = 2*(1-1/(2*rating_prb)*(r_esti[i1,:]-rating[i1,:]))*(r_esti[i1,:]-rating[i1,:])*np.exp(-(r_esti[i1,:]-rating_offset)/rating_prb)*r_training[i1,:]
        temp1 = temp1/np.max([np.sum(r_training[i1,:]*np.exp((r_esti[i1,:]-rating_offset)/rating_prb)),1])
        x[i1,:] = x[i1,:]-eta*(np.dot(np.sum(w_friend[:,i])*temp1,theta)+lambda0*x[i1,:])
    for i2,bus in enumerate(sel_businesses.keys()):
        #pdb.set_trace()
        temp2 = 2*(1-1/(2*rating_prb)*(r_esti[:,i2]-rating[:,i2]))*(r_esti[:,i2]-rating[:,i2])*np.exp(-(r_esti[:,i2]-rating_offset)/rating_prb)*r_training[:,i2]
        temp2 = temp2/np.max([np.sum(r_training[:,i2]*np.exp((r_esti[:,i2]-rating_offset)/rating_prb)),1])
        theta[i2,:] = theta[i2,:]-eta*(np.dot(temp2,x)+lambda0*theta[i2,:])


# In[128]:

loc=np.where(np.logical_and(rating*(1-split1)*r_exist<3, rating*(1-split1)*r_exist>0))
err1 = r_esti[loc[0],loc[1]]-rating[loc[0],loc[1]]
[np.sum(np.abs(err1))/len(err1),size(np.where(np.abs(err1)<=1))/len(err1)]


# In[94]:

def sigmoid(t,c):
    return 1/(1+np.exp(-12*(c+t)))


# In[95]:

# 修改版2
x = np.ones((len(users),s_r_cate))#初始化x
lambda0 = 0.1
eta = 0.1
c = np.linspace(-4.5,-1.5,4)
r_exist = np.ceil(rating/10)
rating_prb = 4
rating_offset = 1
r_training =  r_exist*split1
#初始化theta
theta = np.zeros((len(sel_businesses),s_r_cate))
for bus in businesses:
    if (bus['city'] in selection) and (bus['review_count']>cutoff) and ('Restaurants' in bus['categories']):
        #pdb.set_trace()
        cate_num = len(bus['categories'])-1
        if cate_num==0:
            theta[business_num[bus['business_id']],:] = 1/s_r_cate*bus['stars']
        else:
            for i in bus['categories']:
                if i != 'Restaurants':
                    theta[business_num[bus['business_id']],find(r_cate==i)[0]] = 1/cate_num*bus['stars']
#pdb.set_trace()
for i0 in range(80):
    r_esti_relax = np.dot(x,np.transpose(theta))
    ti = np.asarray([sigmoid(r_esti_relax,ci) for ci in c])
    r_esti1 = 1+np.sum(ti,0)
    
    for i1,user in enumerate(users.keys()):
        temp1 = 2*(1-1/(2*rating_prb)*(r_esti1[i1,:]-rating[i1,:]))*(r_esti1[i1,:]-rating[i1,:])*np.exp(-(r_esti1[i1,:]-rating_offset)/rating_prb)*r_training[i1,:]
        temp1 = temp1/np.sum(r_training)
        #temp1 = temp1/np.max([np.sum(r_training[i1,:]*np.exp((r_esti[i1,:]-rating_offset)/rating_prb)),1])
        x[i1,:] = x[i1,:]-eta*(np.dot(np.sum(16*ti[:,i1,:]*(1-ti[:,i1,:]),0)*temp1,theta)+lambda0*x[i1,:])
    for i2,bus in enumerate(sel_businesses.keys()):
        #pdb.set_trace()
        temp2 = 2*(1-1/(2*rating_prb)*(r_esti1[:,i2]-rating[:,i2]))*(r_esti1[:,i2]-rating[:,i2])*np.exp(-(r_esti1[:,i2]-rating_offset)/rating_prb)*r_training[:,i2]
        temp2 = temp2/np.sum(r_training)#np.max([np.sum(r_training[:,i2]),1])#np.sum(r_training)
        #temp2 = temp2/np.max([np.sum(r_training[:,i2]*np.exp((r_esti[:,i2]-rating_offset)/rating_prb)),1])
        theta[i2,:] = theta[i2,:]-eta*(np.dot(16*np.sum(ti[:,:,i2]*(1-ti[:,:,i2]),0)*temp2,x)+lambda0*theta[i2,:])
    for i3 in range(4):
        temp3 = 2*(1-1/(2*rating_prb)*(r_esti1-rating))*(r_esti1-rating)*np.exp(-(r_esti1-rating_offset)/rating_prb)*r_training/np.sum(r_training)
        c[i3] = c[i3]-eta*(np.sum(temp3*16*ti[i3,:,:]*(1-ti[i3,:,:]))+lambda0*c[i3])
    #theta = theta
#r_esti


# In[27]:

temp = r_esti1 - r_esti
temp[np.logical_and(r_esti<=2,r_esti1<=3.5)] = 0
r_esti0 = r_esti+temp


# In[101]:

loc=np.where(np.logical_and(r_esti1*(1-split1)*r_exist<6, rating*(1-split1)*r_exist>0))
err1 = r_esti1[loc[0],loc[1]]-rating[loc[0],loc[1]]
[np.sum(np.abs(err1))/len(err1),size(np.where(np.abs(err1)<=0.5))/len(err1)]


# In[29]:

average_rating=np.sum(rating*split1,axis=0)/np.sum(r_training,axis=0)


# In[102]:

r_estimate = np.copy(r_esti1)
loc=np.where(rating*(1-split1)>0)
#loc=np.where(np.logical_and(rating*(1-split1)>=3, rating*(1-split1)>0))
for i,j in zip(loc[0],loc[1]):
    if average_rating[j]>4:#check if the estimate has too much difference with the overall rating
        r_estimate[i,j] = average_rating[j]
err = r_estimate[loc[0],loc[1]]-rating[loc[0],loc[1]]


# In[103]:

[np.sum(np.abs(err))/len(err),size(np.where(np.abs(err)<=0.5))/len(err)]


# In[32]:

temp = r_esti1-floor(r_esti)
temp[floor(r_esti)<2] = 0
r_esti0 = floor(r_esti)+temp


# In[104]:

loc=np.where(np.logical_and(rating*(1-split1)*r_exist==1, rating*(1-split1)*r_exist>0))
err1 = r_estimate[loc[0],loc[1]]-rating[loc[0],loc[1]]
[np.sum(np.abs(err1))/len(err1),size(np.where(np.abs(err1)<=0.5))/len(err1)]


# In[105]:

count = np.array([2.31,1.51,0.65,0.36,1.04])
plt.bar(range(1,6),count,0.4)
plt.title('the MAE of reviews')
plt.xlabel('stars')
plt.ylabel('MAE')


# In[96]:

x = np.linspace(0,5,1000)
c1 = np.linspace(-4.5,-1.5,4)
a = np.ones(1000)
for i in range(1000):
    for j in range(4):
        a[i] += sigmoid(x[i],c1[j])
b = np.around(x)
b[np.where(b==0)]=1


# In[100]:

plt.plot(x,x)
plt.plot(x,b)
plt.plot(x,a)
plt.legend(['predicted','oringinal','rounding'],loc='best')
plt.xlabel('predicted rating')
plt.ylabel('rounding rating')


# In[34]:

c


# In[35]:

temp = r_esti2 - r_esti
temp[np.logical_and(r_esti<=2.5,r_esti2>=3.5)] = 0
r_esti0 = r_esti+temp
loc=np.where(np.logical_and(rating*(1-split1)*r_exist<6, rating*(1-split1)*r_exist>0))
err1 = r_esti0[loc[0],loc[1]]-rating[loc[0],loc[1]]
[np.sum(np.abs(err1))/len(err1),size(np.where(np.abs(err1)<=1))/len(err1)]


# In[ ]:



