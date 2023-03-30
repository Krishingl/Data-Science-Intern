#!/usr/bin/env python
# coding: utf-8

# # Lets Grow More - VIRTUAL INTERNSHIP PROGRAM 2023
# ## Name : KRUSHNA SHRIKANT INGLE
# ## Task 1 - Iris Flowers Classification  ML
# 

# Here is the little description about Iris dataset.
# 
# The data set consists of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

# ![iris-machinelearning.png](attachment:iris-machinelearning.png)

# **Objective:**
# 
# Our main objective is to classify the flowers into their respective species - Iris setosa, Iris virginica and Iris versicolor by using various possible plots.

# ## Import  Libraries

# In[43]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm, datasets


# ## Import Dataset

# In[2]:


df = pd.read_csv("F:\\Decoder\\The Spark Foundation\\Iris.csv")
df.head()


# ## Data Inspection

# In[4]:


df.shape


# In[5]:


df.info()


# `Conclusion` : From the above information we can see that it is a balanced dataset.

# In[6]:


df.describe()


# In[9]:


df.nunique()


# In[13]:


df['Species'].value_counts()


# In[14]:


#Dropping unnecessary columns
df = df.drop('Id', axis=1)
df.head(3)


# ## Analysing the Dataset

# In[15]:


df.columns


# In[33]:


for i in df.columns:
    plt.scatter(df[str(i)],df['SepalLengthCm'])
    plt.xlabel(i)
    plt.ylabel("Sepal Length Cm")
    plt.show()


# `Conclusion:` The plot doesnot convey much information about the nature of distribution of the sepal length vs sepal width. Hence, we would use different colours(based on their class type) to interpret the distribution nature.

# In[36]:


sns.pairplot(df, hue ='Species')


# **`Inferences`**
# 1. Iris-Setosa can be easily differentiated
# 2. Iris-Versicolor and Iris-Virginica are overlaping in most of the plot, making it difficult for us to differentiate
# 3. The graph of Petal Length vs Petal Width shows us the best result that can be used to segregate Iris-Versicolor and Iris-Virginica

# ## Correlation Between the numeric variables

# In[48]:


df.corr()


# ## Plotting the correlation using a heatmap

# In[55]:


plt.figure(figsize = (10,10))
sns.heatmap(df.corr(), cmap='Blues',annot = True)


# **`Conclusion:`**
# 
# 1. From the graph, it can be clearly seen that the columns Petal length and petal width hold a strong correlation (=0.96). Earlier we had got the same observations while plotting a pairplot.
# 2. Apart from this, the columns Sepal length and Petal length also hold a high corelation(=0.87).
# 3. Sepal length and Petal width alzo hold a good correlation (=0.82).

# # Observing the distribution nature of all the 4 columns (Using a Distplot)

# In[56]:


df.columns


# In[78]:


col = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
plt.figure(figsize=(35,5))
i = 1
for e in col:
    plt.subplot(1,10,i)
    sns.distplot(df[e])
    i = i + 1
plt.tight_layout()
plt.show()


# ### Conclusion:
# It can be observed from the above distplots that the distribution of the columns follow a Normal distribution

# ## Observing the distribution of the data across the various columns using a Histogram

# In[79]:


df.describe()


# Using the describe() we can see that the maximum value for the columns:
# 
# * Sepal Length = **8**
# * Sepal Width = **5**
# * Petal length = **7**
# * Petal Width = **3**
# 
# Hence, we'll set our bin size of the histogram based on the above specified values.

# In[92]:


fig, axes = plt.subplots(2, 2, figsize=(15,10))
axes[0,0].set_title("Distribution of Sepal Width")
axes[0,0].hist(df['SepalWidthCm'], bins=5);
axes[0,1].set_title("Distribution of Sepal Length")
axes[0,1].hist(df['SepalLengthCm'], bins=8);
axes[1,0].set_title("Distribution of Petal Width")
axes[1,0].hist(df['PetalWidthCm'], bins=5);
axes[1,1].set_title("Distribution of Petal Length")
axes[1,1].hist(df['PetalLengthCm'], bins=8)
plt.show()


# ### Conclusions:
# * The highest frequency of Sepal length ranges between 6.0 - 6.5 which is around 32
# * The highest frequency of Sepal Width ranges between 3.0 - 3.5 which is around 69
# * The highest frequency of Petal length ranges between 1.0 - 1.8 which is around 50
# * The highest frequency of Petal Width ranges between 0.0 - 0.5 which is around 50

# ## Univariate Analysis of all the 4 columns(using Distplots)

# In[94]:


sns.FacetGrid(df,hue="Species",height=5).map(sns.distplot,"SepalLengthCm").add_legend()


# ### Conclusion:
# It can be clearly seen that the flower species cannot be seperated based on the Sepal length as the values overlap a lot.

# In[96]:


sns.FacetGrid(df,hue="Species",height=5).map(sns.distplot,"SepalWidthCm").add_legend()


# ### Conclusion:
# It can be clearly seen that the flower species cannot be seperated based on the Sepal Width as here also, the values overlap a lot. the overlapping is more intense in this case as compared to the overlapping in the case of Sepal Length.

# In[97]:


sns.FacetGrid(df,hue="Species",height=5).map(sns.distplot,"PetalLengthCm").add_legend()


# ### Conclusion:
# From the graph, it can be seen that Setosa is easily segregable, whereas Versicolor and Virginica do overlap at some points (near 4.5-5). The column Petal length can be used to seperate the species 

# In[98]:


sns.FacetGrid(df,hue="Species",height=5).map(sns.distplot,"PetalWidthCm").add_legend()


# ### Conclusion:
# From the graph, it can be seen that Setosa is easily segregable, whereas Versicolor and Virginica do overlap at some points (near 1.5-2.0). The column Petal Width can also be used to seperate the species .

# In[99]:


fig, axes = plt.subplots(2, 2, figsize=(16,12))
axes[0,0].set_title("Distribution of Sepal Length")
sns.boxplot(y="SepalLengthCm", x= "Species", data=df,  orient='v' , ax=axes[0, 0])
axes[0,1].set_title("Distribution of Sepal Width")
sns.boxplot(y="SepalWidthCm", x= "Species", data=df,  orient='v' , ax=axes[0, 1])
axes[1,0].set_title("Distribution of Petal Length")
sns.boxplot(y="PetalLengthCm", x= "Species", data=df,  orient='v' , ax=axes[1, 0])
axes[1,1].set_title("Distribution of Petal Width")
sns.boxplot(y="PetalWidthCm", x= "Species", data=df,  orient='v' , ax=axes[1, 1])
plt.show()


# ### Conclusions:
# 
# We can see that the species Setosa doesnot have any outliers in case of Sepal Length or Sepal Width, however, it does have few outliers in Petal length and Petal Width.
# In terms of features like: Petal Width / Length, Virginca has quiet high values as compared to the other two species. Also, Setosa has the least values for the same features.
# It is also observed that for the feature Sepal Width, Setosa has a wide range of values as compared to the other species

# ### Let's Dive a lil Deeper !!
# To furthur analyze the distribution we are using a violin plot.
# 
# Violin plots are used when you want to observe the distribution of numeric data, and are especially useful when you want to make a comparison of distributions between multiple groups. The peaks, valleys, and tails of each group's density curve can be compared to see where groups are similar or different.

# In[100]:


fig, axes = plt.subplots(2, 2, figsize=(16,12))
axes[0,0].set_title("Distribution of Sepal Length")
sns.violinplot(y="SepalLengthCm", x= "Species", data=df,  orient='v' , ax=axes[0, 0])
axes[0,1].set_title("Distribution of Sepal Width")
sns.violinplot(y="SepalWidthCm", x= "Species", data=df,  orient='v' , ax=axes[0, 1])
axes[1,0].set_title("Distribution of Petal Length")
sns.violinplot(y="PetalLengthCm", x= "Species", data=df,  orient='v' , ax=axes[1, 0])
axes[1,1].set_title("Distribution of Petal Width")
sns.violinplot(y="PetalWidthCm", x= "Species", data=df,  orient='v' , ax=axes[1, 1])
plt.show()


# ### Conclusions:
# 
# The kernel density in the Violin plots helps us understand the full distribution of the data in terms of density.

# ## Final Conclusions
# 
# * The dataset is completely balanced i.e. equal number of records are present for each of the three species.
# * Here our target column is Species, as we need to segregate the flowers as per their species based on the 4 fetaures namely, Sepal Length, Sepal Width, Petal Length and Petal Width .
# * The columns Petal length and petal width hold a strong correlation (=0.96) and can be used to segregate the flowers.
# * By plotting various graphs, we can conclude that:
# * The Setosa species is easily segregable because of its small feature value range.
# * The Versicolor and Virginca species are a bit difficult to seperate because they overlap at many points in terms of their features.

# # Thank You

# In[ ]:




