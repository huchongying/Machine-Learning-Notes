# 零基础入门逻辑回归（使用 sklearn 实现）

逻辑回归（Logistic Regression）是机器学习中最基础的分类算法之一，主要用于**二分类问题**，比如判断“是否患病”、“邮件是否是垃圾邮件”等。

本篇教程将带你使用 Python 的 `sklearn` 库来构建一个逻辑回归模型，哪怕你完全没有机器学习基础，也能轻松学会。

---

## ✅ 什么是逻辑回归？

逻辑回归并不是“回归”而是**分类**算法。它通过学习一组参数（权重），将输入特征映射到一个**0~1之间的概率值**，然后根据概率大小判断属于哪一类。

例如：  
- 若模型输出概率为 0.9，我们预测为“是”
- 若模型输出概率为 0.1，我们预测为“否”

---

## ✅ 步骤一：导入库

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
```

---

## ✅ 步骤二：准备数据

我们使用一个简单的数据集，包含两个特征（如考试成绩），目标是判断是否录取（0/1）。

```python
# 构造数据
X = np.array([
    [70, 80],
    [80, 90],
    [85, 95],
    [60, 65],
    [55, 60],
    [90, 95],
    [40, 50],
    [75, 85],
    [50, 45],
    [85, 70]
])

y = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1])  # 1表示被录取，0表示未录取
```

---

## ✅ 步骤三：划分训练集和测试集

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

---

## ✅ 步骤四：训练逻辑回归模型

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

---

## ✅ 步骤五：进行预测

```python
y_pred = model.predict(X_test)
```

---

## ✅ 步骤六：评估模型效果

```python
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred))

print("\n分类报告：")
print(classification_report(y_test, y_pred))
```

---

## ✅ 可视化（可选）

```python
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.xlabel('成绩1')
plt.ylabel('成绩2')
plt.title('逻辑回归数据分布')
plt.show()
```

---

## ✅ 总结

| 步骤         | 说明                         |
|--------------|------------------------------|
| 导入数据     | 输入特征（如考试成绩）       |
| 设置标签     | 输出为 0 或 1（二分类）       |
| 划分数据集   | 训练集 & 测试集              |
| 模型训练     | 使用 `LogisticRegression`     |
| 模型评估     | 查看预测结果是否准确         |

---

> 🎓 小贴士：
> - 如果输出结果是概率，可以用 `model.predict_proba()`。
> - 如果特征维度很高，可以使用 `StandardScaler` 做标准化。

# 逻辑回归原理图解

逻辑回归（Logistic Regression）是一种广泛应用于分类问题的算法，虽然名字中带有“回归”，但它实际上是用来处理**二分类问题**的。逻辑回归能够预测某个样本属于某个类别的概率，通常应用于“是/否”、“真/假”等问题。

在本篇文章中，我们将通过图解的方式，帮助零基础的学习者理解逻辑回归的基本原理。

---

## ✅ 1. 线性回归与逻辑回归的关系

逻辑回归与线性回归非常相似。在线性回归中，我们通过一个线性模型预测输出值（如房价），而在逻辑回归中，我们要将线性回归的输出值转换成**概率值**，并进行二分类。

### 线性回归模型

线性回归的模型公式为：

$$
h_{\theta}(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
$$

其中，$h_{\theta}(x)$ 是预测的输出值，$x_1, x_2, \dots, x_n$ 是输入特征，$\theta_0, \theta_1, \dots, \theta_n$ 是参数。

---

## ✅ 2. 为什么用逻辑回归而不是线性回归？

线性回归的输出是一个**实数值**，这对于分类任务来说并不合适，因为我们希望输出是一个**概率值**，且范围在 [0, 1] 之间。为了将线性回归的输出限制在 [0, 1]，我们使用了一个**激活函数**——**Sigmoid 函数**。

---

## ✅ 3. Sigmoid 函数

Sigmoid 函数的公式为：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

- 其中 $z$ 是线性回归的输出，即 $z = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n$。
- $e$ 是自然对数的底数（约为 2.718）。
- 通过这个公式，**Sigmoid 函数将输出限制在 0 和 1 之间**，并可以解释为“事件发生的概率”。

### Sigmoid 函数图像：

![Sigmoid 函数图像](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/2560px-Logistic-curve.svg.png)

- **横坐标**：输入 $z$ 值（即线性回归的输出）
- **纵坐标**：Sigmoid 函数输出的概率值

通过 Sigmoid 函数，逻辑回归将每个样本的预测结果转换为属于类别 1 的概率。这个概率值大于 0.5 时，我们预测为类别 1，否则预测为类别 0。

---

## ✅ 4. 逻辑回归模型

因此，逻辑回归模型的输出就是通过 Sigmoid 函数计算得到的概率值：

$$
h_{\theta}(x) = \sigma(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n)
$$

这个输出值表示样本属于类别 1 的概率。例如，$h_{\theta}(x) = 0.8$ 意味着该样本属于类别 1 的概率是 80%。

---

## ✅ 5. 代价函数（Cost Function）

为了训练逻辑回归模型，我们需要定义一个代价函数，用来衡量模型预测的准确性。逻辑回归的代价函数使用**对数损失**，公式为：

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(h_{\theta}(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\theta}(x^{(i)}))\right]
$$

- $m$ 是样本数量，$y^{(i)}$ 是第 $i$ 个样本的实际标签（0 或 1），$h_{\theta}(x^{(i)})$ 是第 $i$ 个样本的预测概率。

### 代价函数图像：

代价函数的目标是最小化预测值与实际标签之间的误差。代价函数越小，模型预测的准确度就越高。

---

## ✅ 6. 梯度下降（Gradient Descent）

梯度下降是用来最小化代价函数的优化算法。我们通过梯度下降不断调整模型参数 $\theta$，直到代价函数达到最小值。

每次更新参数的公式为：

$$
\theta_j := \theta_j - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta_j}
$$

- $\alpha$ 是学习率，决定每次更新的步长。
- $\frac{\partial J(\theta)}{\partial \theta_j}$ 是代价函数对参数 $\theta_j$ 的偏导数，表示参数更新的方向和大小。

---

## ✅ 7. 总结

- 逻辑回归是一个基于概率的分类模型，它将线性回归的输出通过 Sigmoid 函数转换为概率值。
- 逻辑回归使用代价函数（对数损失）来衡量预测值与实际标签之间的误差，通过最小化代价函数来优化模型参数。
- 梯度下降是最常用的优化算法，用来更新模型的参数，使得代价函数最小化。


