#多特征线性模型
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 解决plt中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

data=pd.read_csv('C:\\Users\\s1597\\Desktop\\Python\\Machine Learning\\Machine Notes\\Linear_regression\\ex1data2.txt',names=['Size','Bedrooms','Price'])

#数据划分
X=data.iloc[:,:-1].values #特征变量
#去掉最后一列
y=data.iloc[:,-1].values #目标变量
#最后一列

#划分为训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#创建模型与训练模型
model=LinearRegression()
model.fit(X_train,y_train)

#预测
y_pred=model.predict(X_test)

#决策边界
w=model.coef_ #斜率
b=model.intercept_ #截距

#多特征可视化
