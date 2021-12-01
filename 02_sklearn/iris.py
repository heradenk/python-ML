#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_iris


# In[3]:


from sklearn.tree import DecisionTreeClassifier


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


import pandas as pd
# 붓꽃 데이터 세트를 로딩합니다.
iris = load_iris()

# iris_data는 Iris 데이터 세트에서 피처(feature)만으로 된 데이터를 numpy로 가지고 있습니다.
iris_data = iris.data

# iris.target은 붓꽃 데이터 세트에서 레이블(결정 값) 데이터를 numpy로 가지고 있습니다.
iris_label = iris.target
print('iris target값:', iris_label)
print('iris target명:', iris.target_names)

# 붓꽃 데이터 세트를 자세히 보기 위해 DataFrame으로 변환합니다.
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
iris_df.head(3)


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)


# In[9]:


# DecisionTreeClassifier 객체 생성
df_clf = DecisionTreeClassifier(random_state=11)


# In[10]:


# 학습 수행
df_clf.fit(X_train, y_train)


# In[12]:


# 학습이 완료된 DecisionTreeClassifier 객체에서 테스트 데이터 세트로 예측 수행.
pred = df_clf.predict(X_test)

