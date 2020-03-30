#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import nltk
import copy
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import OrderedDict
import math
import operator
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


dataset_path = 'G:/IR/AASIGNMENTS/ASSIGNMENT_4/Q4_DATASET'
stop_words = set(stopwords.words('english'))
directory = os.listdir(dataset_path)


# In[3]:


def preprocess(line):
    tokenizer = nltk.RegexpTokenizer('\w+')
    tokens_list = tokenizer.tokenize(line)
    for tokens in tokens_list:
        if tokens in stop_words:
            tokens_list.remove(tokens)
    lemmatizer = WordNetLemmatizer()
    words_list=[]
    for tokens in tokens_list:
        words_list.append(lemmatizer.lemmatize(tokens))

    words_list = [element.lower() for element in words_list]
    return words_list


# In[4]:


def seggregate_data(dataset,label_dataset,train_percent):
    length_train=train_percent*len(dataset)
    data=np.array(dataset)
    label=np.array(label_dataset)
    random_order=np.arange(len(dataset))
    np.random.shuffle(random_order)
    train_dataset=np.array(data[random_order[:int(length_train)]])
    train_labelset=np.array(label[random_order[:int(length_train)]])
    test_dataset=np.array(data[random_order[int(length_train):]])
    test_labelset=np.array(label[random_order[int(length_train):]])
    return list(train_dataset),list(train_labelset),list(test_dataset),list(test_labelset)


# In[5]:


def split_data(dataset,label_dataset,train_percent):
    length_train=train_percent*len(dataset)
    label=np.array(label_dataset)
    random_order=np.arange(len(dataset))
    np.random.shuffle(random_order)
    train_dataset=np.array(random_order[:int(length_train)])
    train_labelset=np.array(label[random_order[:int(length_train)]])
    test_dataset=np.array(random_order[int(length_train):])
    test_labelset=np.array(label[random_order[int(length_train):]])
    return list(train_dataset),list(train_labelset),list(test_dataset),list(test_labelset)


# In[42]:


def ROC_curve_points(l,discrimanting_fnc_list,labelset,count_one,count_two):
    posterior = copy.copy(discrimanting_fnc_list)
    #discrimanting_fnc_list.sort()
    x_points=[]
    y_points=[]
    threshold_val=np.linspace(-0.1,1.5,2000,endpoint=False)
    for threshold in threshold_val:
    #for threshold in range(0,len(discrimanting_fnc_list)):
        fp=0
        tp=0
        for i in range(0,len(test_dataset)):
            if posterior[i] > threshold :
                if labelset[i]==l:
                    tp+=1
                else:
                    fp+=1
        #print(tp,fp)
        y_points.append(tp/count_one)
        x_points.append(fp/count_two)
        #print(threshold,fp/count_two,tp/count_one)
    #print(x_points,y_points)
    return x_points,y_points


# In[7]:


##################distance calculation####################
def calculate_distance(test_data,train_data):
    dist=0
    train_keys=list(train_data.keys())
    test_keys=list(test_data.keys())
    #print(train_keys)
    keys=list(set(train_keys + test_keys))
    for k in keys:
        if k in train_keys and k in test_keys:
              dist+=abs(train_data[k]-test_data[k])**2
        elif k not in train_keys:
            dist+=test_data[k]**2
        else:
            dist+=train_data[k]**2
    #print(dist)
    return dist**0.5


# In[8]:


############################DATA PREPROCESSS########################3
label=-1
doc_id=0
flag=0
dataset=[]
dataset_label=[]
vocabs_list=[]
for folder in directory:
    files_list = os.listdir(dataset_path+'/'+folder)
    label+=1
    vocab_count=0
    doc_count=0
    class_vocab_dict=dict()
    for file in files_list:
        vocab_dict=dict()
        test_file=[]
        file = open(dataset_path+'/'+folder+'/'+file,'r')
        file_data = file.readlines()
        fileName=file.name.split("/")[-2:]
        file_name=""
        file_name=file_name.join(fileName)
        for line in file_data:
            procesed_word_list = preprocess(line)
            for word in procesed_word_list:
                    if word in vocab_dict.keys():
                        vocab_dict[word]+=1
                    else:
                        if word not in vocabs_list:
                            vocabs_list.append(word)
                        vocab_dict[word]=1
        dataset.append(vocab_dict)
        dataset_label.append(label)
        doc_id+=1 


# In[8]:


#################save distance among each other#############
distance=np.zeros((5000,5000))
for i in range(0,len(dataset)):
    dis=[]
    for j in range(0,i+1):
        if i==j:
            distance[i][j]=-1
        else:
            var=calculate_distance(dataset[i],dataset[j])
            distance[j][i]=var
            distance[i][j]=var
    if i%200==0:
        print(i)


# In[64]:





# In[13]:


#####################write to a file############################
f=open("G:/IR/AASIGNMENTS/ASSIGNMENT_5/dist_matrix.txt","w+")
for i in range(0,5000):
    for j in range(0,5000):
        f.write(str(distance[i][j])+" ")
    f.write("\n")
f.close()


# In[77]:


################reading distance matrix from file###########
distance_vec=[]
with open("G:/IR/AASIGNMENTS/ASSIGNMENT_5/distance.txt","r",encoding="latin",errors='ignore') as f:
    for line in f:
        temp=[]
        words = line.split()
        print(words)
        for w in words:
            temp.append(float(w))
    distance.append(temp)


# In[9]:


#distance_vec=np.zeros(5000,5000)
f=open("G:/IR/AASIGNMENTS/ASSIGNMENT_5/dist_matrix.txt","r")
data=f.readlines()
distance=[]
for line in data:
    temp=[]
    words = line.split()
    for w in words:
        temp.append(float(w))
    distance.append(temp)


# In[10]:


np.shape(distance)


# In[128]:


########################train test split#################
#train_dataset,train_label,test_dataset,test_label=seggregate_data(dataset,dataset_label,0.7)
train_dataset,train_label,test_dataset,test_label=split_data(dataset,dataset_label,0.5)


# In[44]:


len(train_label)


# In[132]:


#######################FOR DISTANCE MATRIX###############################
k=5
k1=5
match=0
probability=[]
c=0
labelset=[]
confusion_matrix=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
for t in test_dataset:
    #print(t)
    if c%100==0:
        print(c)
    c+=1
    dist=[]
    k=k1
    #tst_ind=dataset.index(t)
    for tr in train_dataset:
        #print("tr ",tr)
        #tr_ind=dataset.index(tr)
        if distance[t][tr]!=-1.0:
            dist.append(distance[t][tr])
        else:
            print("error..")
    dis1=copy.deepcopy(dist)
    ordered=[x for _,x in sorted(zip(dist,train_label))]
    order_label=[x for _,x in sorted(zip(dis1,train_label))]
    top_doc=[]
    top_label=[]
    i=0
    while(k>0):
        top_label.append(order_label[i])
        i+=1
        k-=1
    prob=[]
    for i in range(0,k1):
        prob.append(top_label.count(i)/k1)
    probability.append(prob)
    max_val=max(top_label,key=top_label.count)
    ind=top_label.index(max_val)
    predict_label=top_label[ind]
    if predict_label==dataset_label[t]:
               match+=1
    confusion_matrix[predict_label][dataset_label[t]]+=1
    labelset.append(dataset_label[t])
ax=sns.heatmap(confusion_matrix,annot=True,fmt='d')
plt.show()
print("accuracy: ",(match/len(test_dataset))*100)


# In[133]:


########################roc draw############################
pos_0=np.asarray(probability).transpose()
for i in range(0,k1):
    x,y=ROC_curve_points(i,pos_0[i],labelset,test_label.count(i),len(test_label)-test_label.count(i))
    plt.plot(x,y, label=str(i)+" as positive class")
plt.legend(loc="lower right")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()


# In[73]:


train_dataset1=[]
test_dataset1=[]
for t in train_dataset:
    train_dataset1.append(dataset[t])
for t in test_dataset:
    test_dataset1.append(dataset[t])
train_dataset=copy.deepcopy(train_dataset1)
test_dataset=copy.deepcopy(test_dataset1)


# In[74]:


######################data preprocess for naive bayes###############
class_list=[]
class_vocab_count=[]
class_doc_count=[]
class_doc=dict()
for doc in train_dataset:
    if train_label[train_dataset.index(doc)] in class_doc.keys():
        class_doc[train_label[train_dataset.index(doc)]].append(doc)
    else:
        temp=[]
        temp.append(doc)
        class_doc[train_label[train_dataset.index(doc)]]=temp
for i in range(0,5):
    class_vocab=dict()
    docs=class_doc[i]
    doc_count=0
    vocab_count=0
    for d in docs:
        doc_count+=1
        for word in d.keys():
            vocab_count+=1
            if word in class_vocab.keys():
                class_vocab[word]+=1
            else:
                class_vocab[word]=1
    class_list.append(class_vocab)
    class_doc_count.append(doc_count)
    class_vocab_count.append(vocab_count)


# In[104]:


####################vocabs list creation##################
vocabs_list=[]
for d in dataset:
    for word in d.keys():
        if word not in vocabs_list:
            vocabs_list.append(word)


# In[60]:


#################posterior calculation##########################
def calculate_posterior(class_label,feature):
    likelihood=0
    #print("sigma",sigma)
    for f in feature:
        #print(f,class_label)
        if f in class_list[class_label].keys():
            val=class_list[class_label][f]
        else:
            val=0
        likelihood+=np.log((val+1)/(class_vocab_count[class_label]+len(vocabs_list)))
    posterior=likelihood+np.log(class_doc_count[class_label]/int(np.sum(class_doc_count)))
    return posterior


# In[75]:


######################naive bayes comparison##########################
match=0
confusion_matrix=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
for i in range(0,len(test_dataset)):
    doc=test_dataset[i]
    feature=list(doc.keys())
    #print("feature",feature)
    posterior_list=[]
    for c in range(0,5):
        posterior_0 = calculate_posterior(c,feature)
        posterior_list.append(posterior_0)

    predict_label=posterior_list.index(max(posterior_list))
    confusion_matrix[predict_label][test_label[i]]+=1
    if predict_label == test_label[i]:
        match+=1
ax=sns.heatmap(confusion_matrix,annot=True,fmt='d')
plt.show()
print("accuracy :",match/(len(test_dataset))*100)


# In[97]:


d=[]
a=np.array([2,4])
b=np.array([5,6])
d.append(a)
d.append(b)
print(d)
[np.array_equal(b,x) for x in d].index(True)


# In[134]:


x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
y=[6239.795890664105, 3328.7328302078795, 3300.8016261893813, 3288.754780643293, 3281.7653779304783, 3277.3835155320708, 3275.2374337604087, 3274.1319529206917, 3273.682791884025, 3273.441373597945, 3273.2975055307893, 3273.2053805139813, 3273.148433917231, 3273.1059382144426, 3273.071508339264, 3273.0406814848607, 3273.024836386065, 3273.0133290577937, 3273.00761876932, 3273.0060947627376]
plt.plot(x,y, label="Convergence of kmeans",marker="o")
plt.legend(loc="upper right")
plt.xlabel('Iterations')
plt.ylabel('RSS score')
plt.show()

