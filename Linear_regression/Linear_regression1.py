import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 解决plt中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

#读取数据
data=pd.read_csv('C:\\Users\\s1597\\Desktop\\Python\\Machine Learning\\Machine Notes\\Linear_regression\\ex1data1.txt',names=['Population','Profit'])

#数据预处理
X=data.iloc[:,0].values.reshape(-1,1) #reshape(-1,1)将一维数组转换为二维数组
y=data.iloc[:,1].values.reshape(-1,1)

#创建模型与训练模型
model=LinearRegression()
model.fit(X,y)

#划分为训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#预测
y_pred=model.predict(X_test)

#决策边界
w=model.coef_[0][0] #斜率
b=model.intercept_[0] #截距

#可视化
plt.scatter(X,y,color='blue',label='数据点')
plt.plot(X_test,y_pred,color='red',label='预测值')
plt.xlabel('人口')
plt.ylabel('利润')
plt.title('线性回归')
plt.legend()
plt.show()






