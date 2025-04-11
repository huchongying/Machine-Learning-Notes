# 零基础也能学懂：Python读取数据文件（机器学习第一步）

在机器学习的学习与实战过程中，**读取数据文件是第一步**。我们常用的训练数据往往来自 CSV 或 TXT 文件，因此学会读取并处理数据是非常关键的。

本篇内容将教你如何使用 Python 的两个最常用的数据处理库：**NumPy** 和 **Pandas**，读取数据文件并进行简单预处理，适合零基础入门！

---

## 📌 为什么要读取数据文件？

在机器学习项目中，数据一般存储在文件中（例如：CSV格式），我们需要**读取并转换为代码可以操作的结构**，比如数组或表格，之后才能进行模型训练。

---

## ✅ 使用 NumPy 读取数据

### 1. 导入 NumPy

```python
import numpy as np
```

### 2. 使用 `np.loadtxt()` 读取纯数值数据

假设 `data.txt` 文件内容如下：

```
1.0, 2.0, 3.0
4.0, 5.0, 6.0
7.0, 8.0, 9.0
```

使用以下代码读取：

```python
data = np.loadtxt('data.txt', delimiter=',')
print(data)
```

### 3. 跳过标题行（有列名）

假设 `data.txt` 内容如下：

```
feature1, feature2, feature3
1.0, 2.0, 3.0
4.0, 5.0, 6.0
```

使用以下代码跳过第一行：

```python
data = np.loadtxt('data.txt', delimiter=',', skiprows=1)
```

### 4. 读取指定列

只读取第1列和第3列（索引从0开始）：

```python
data = np.loadtxt('data.txt', delimiter=',', usecols=(0, 2))
```

---

## ✅ 使用 Pandas 读取数据

### 1. 导入 Pandas

```python
import pandas as pd
```

### 2. 使用 `pd.read_csv()` 读取数据

```python
df = pd.read_csv('data.txt')
print(df)
```

### 3. 删除缺失值

```python
df = df.dropna()
```

### 4. 用指定值填充缺失值

```python
df = df.fillna(0)
```

### 5. 读取指定列

```python
df = pd.read_csv('data.txt', usecols=['feature1', 'feature3'])
```

---

## ✅ NumPy vs Pandas 对比

| 工具    | 优势                       | 适用场景           |
|---------|----------------------------|--------------------|
| NumPy   | 快速、轻量，适合数值数组   | 纯数值数据         |
| Pandas  | 支持列名、缺失值、标签操作 | 表格结构复杂的数据 |


