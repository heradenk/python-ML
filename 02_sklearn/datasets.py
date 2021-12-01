#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris

iris_data = load_iris()
print(type(iris_data))


# In[2]:


keys = iris_data.keys()
print('붓꽃 데이터 세트의 키들:', keys)


# In[4]:


print('\n feature_names 의 type:', type(iris_data.feature_names))
print(' feature_names 의 shape:', len(iris_data.feature_names))
print(iris_data.feature_names)

print('\n target_names의 type:', type(iris_data.target_names))
print(' target_names의 shape:', len(iris_data.target_names))
print(iris_data.target_names)

print('\n data 의 type:', type(iris_data.data))
print(' data 의 shape:', iris_data.data.shape)
print(iris_data['data'])

print('\n target 의 type:', type(iris_data.target))
print(' target 의 shape:', iris_data.target.shape)
print(iris_data.target)

