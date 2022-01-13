# seaborn实现多种回归

> ​	**导读：** Seaborn就是让困难的东西更加简单。它是针对统计绘图的，一般来说，能满足数据分析**90%**的绘图需求。Seaborn其实是在matplotlib的基础上进行了更高级的API封装，从而使得作图更加容易，同时它能高度兼容numpy与pandas数据结构以及**scipy与statsmodels**等统计模式。
>
> ​	本文主要介绍回归模型图**lmplot**、线性回归图**regplot**，这两个函数的核心功能很相似，都会绘制数据**散点图**，并且拟合关于变量x,y之间的**回归曲线**，同时显示回归的**95%置信区间**。
>
> ​	另一个是线性回归残差图**residplot**，该函数绘制观察点与回归曲线上的预测点之间的**残差图**。

![image-20210602134310846](https://i.loli.net/2021/06/02/TqJgoUkfQSMEa2D.png)

### 数据准备

所有图形将使用股市数据--中国平安`sh.601318`历史k线数据。

**使用模块及数据预处理**

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import baostock as bs
bs.login()
result = bs.query_history_k_data('sh.601318', 
                                 fields = 'date,open,high, low,close,volume',
                                 start_date = '2020-01-01',
                                 end_date = '2021-05-01',
                                 frequency='d')
dataset = result.get_data().set_index('date').applymap(lambda x: float(x))
bs.logout()
dataset['Open_Close'] = (dataset['open'] - dataset['close'])/dataset['open']
dataset['High_Low'] = (dataset['high'] - dataset['low'])/dataset['low']
dataset['Increase_Decrease'] = np.where(dataset['volume'].shift(-1) > dataset['volume'],1,0)
dataset['Buy_Sell_on_Open'] = np.where(dataset['open'].shift(-1) > dataset['open'],1,0)
dataset['Buy_Sell'] = np.where(dataset['close'].shift(-1) > dataset['close'],1,0)
dataset['Returns'] = dataset['close'].pct_change()
dataset = dataset.dropna()
dataset['Up_Down'] = np.where(dataset['Returns'].shift(-1) > dataset['Returns'],'Up','Down')
dataset = dataset.dropna()
dataset.head()
```

![image-20210602134348069](https://i.loli.net/2021/06/02/XuyisLkTY25emHl.png)

## 一、回归模型图lmplot

**lmplot**是一种集合基础绘图与基于数据建立回归模型的绘图方法。通过**lmplot**我们可以直观地总览数据的内在关系。显示每个数据集的线性回归结果，`xy`变量，利用`'hue'、'col'、'row'`参数来控制绘图变量。可以把它看作分类绘图依据。

**同时可以使用模型参数来调节需要拟合的模型：`order、logistic、lowess、robust、logx`。**

### 线性回归

**lmplot**绘制散点图及线性回归拟合线非常简单，只需要指定自变量和因变量即可，**lmplot**会自动完成线性回归拟合。回归模型的置信区间用回归线周围的半透明带绘制。

**lmplot** 支持引入第三维度进行对比，例如我们设置 `hue="species"`。

```python
sns.lmplot(x="open",
           y="close",
           hue="Up_Down",
           data=dataset)
```

![image-20210602134441647](https://i.loli.net/2021/06/02/dp398VNuSzqlRiL.png)

### 局部加权线性回归

局部加权回归散点平滑法(locally weighted scatterplot smoothing，LOWESS)，是一种非参数回归拟合的方式，其主要思想是选取一定比例的局部数据，拟合多项式回归曲线，以便观察到数据的局部规律和趋势。通过设置参数`lowess=True` 。

局部加权线性回归是机器学习里的一种经典的方法，弥补了普通线性回归模型欠拟合或者过拟合的问题。其原理是给待预测点附近的每个点都赋予一定的权重，然后基于最小均方误差进行普通的线性回归。局部加权中的权重，是根据要预测的点与数据集中的点的距离来为数据集中的点赋权值。当某点离要预测的点越远，其权重越小，否则越大。

局部加权线性回归的优势就在于处理非线性关系的异方差问题。

> **lowess** bool, 可选
> 如果为True，使用统计模型来估计非参数低成本模型(局部加权线性回归)。这种方法具有最少的假设，尽管它是计算密集型的，因此目前根本不计算置信区间。

```python
sns.lmplot(x="open",
           y="close",
           hue="Up_Down",
           lowess=True,
           data=dataset)
```

![image-20210602134512345](https://i.loli.net/2021/06/02/CtS8I7c9Pgym5wv.png)

### 对数线性回归模型

通过设置参数`logx` 完成线性回归转换对数线性回归，其实质上是完成了输入空间**x**到输出空间**y**的**非线性映射**。

对数据做一些变换的目的是它能够让它符合我们所做的假设，使我们能够在已有理论上对其分析。对数变换(log transformation)是特殊的一种数据变换方式，它可以将一类我们理论上未解决的模型问题转化为已经解决的问题。

> **logx** : bool, 可选
> 如果为True，则估计y ~ log(x)形式的线性回归，在输入空间中绘制散点图和回归模型。注意x必须是正的。

```python
sns.lmplot(x="open",
           y="close",
           hue="Up_Down",
           data=dataset,
           logx=True)
```

![image-20210602134540478](https://i.loli.net/2021/06/02/bfdUKuphcLI4C1o.png)

### 稳健线性回归

在有异常值的情况下，它可以使用不同的损失函数来减小相对较大的残差，拟合一个健壮的回归模型，传入`robust=True`。

稳健回归是将稳健估计方法用于回归模型，以拟合大部分数据存在的结构，同时可识别出潜在可能的离群点、强影响点或与模型假设相偏离的结构。

稳健回归是统计学稳健估计中的一种方法，其主要思路是将对异常值十分敏感的经典最小二乘回归中的目标函数进行修改。经典最小二乘回归以使误差平方和达到最小为其目标函数。因为方差为一不稳健统计量，故最小二乘回归是一种不稳健的方法。

不同的目标函数定义了不同的稳健回归方法。常见的稳健回归方法有：**最小中位平方法、M估计法**等。

> **hue, col, row** : strings
> 定义数据子集的变量，并在不同的图像子集中绘制
>
> **height** : scalar, 可选
> 定义子图的高度
>
> **col_wrap** : int, 可选
> 设置每行子图数量
>
> **n_boot** int, 可选
> 用于估计的重采样次数`ci`。默认值试图平衡时间和稳定性。
>
> **ci** int in [ **0，100** ]或None, 可选
> 回归估计的置信区间的大小。这将使用回归线周围的半透明带绘制。置信区间是使用`bootstrap`估算的；
>
> **robust** bool, 可选
> 如果为`True`，则用于`statsmodels`估计稳健的回归。这将消除异常值的权重。并且由于使用引导程序计算回归线周围的置信区间，您可能希望将其关闭获得更快的迭代速度（使用参数`ci=None`）或减少引导重新采样的数量`(n_boot)`。

```python
sns.lmplot(x="open",
           y="volume",
           data=dataset,
           hue="Increase_Decrease",
           col="Increase_Decrease", 
           # col|hue控制子图不同的变量species
           col_wrap=2,    
           height=4,      
           robust=True)
```

![image-20210602134609921](https://i.loli.net/2021/06/02/1JpHZ5cgKtN6GFB.png)

### 多项式回归

在存在高阶关系的情况下，可以拟合多项式回归模型来拟合数据集中的简单类型的非线性趋势。通过传入参数`order`大于1，此时使用`numpy.Polyfit`估计多项式回归的方法。

多项式回归是回归分析的一种形式，其中自变量 **x** 和因变量 **y** 之间的关系被建模为关于 **x** 的 次多项式。多项式回归拟合**x**的值与 **y** 的相应条件均值之间的非线性关系，表示为 ，被用于描述非线性现象。

虽然多项式回归是拟合数据的非线性模型，但作为统计估计问题，它是线性的。在某种意义上，回归函数 在从数据估计到的未知参数中是线性的。因此，多项式回归被认为是多元线性回归的特例。

> **order** : int, 可选
> 多项式回归，设定指数

```python
sns.lmplot(x="close",
           y="volume",
           data=dataset,
           hue="Increase_Decrease",
           col="Up_Down", # col|hue控制子图不同的变量species
           col_wrap=2,    # col_wrap控制每行子图数量
           height=4,      # height控制子图高度
           order=3        # 多项式最高次幂
          )
```

![image-20210602134633245](https://i.loli.net/2021/06/02/7GPt8hxQfgdZWY5.png)

### 逻辑回归

**Logistic**回归是一种广义线性回归，**logistic**回归的因变量可以是二分类的，也可以是多分类的，但是二分类的更为常用，也更加容易解释，多类可以使用`softmax`方法进行处理。

实际中最为常用的就是二分类的**logistic**回归。

> **{x,y}_jitter** floats, 可选
> 在x或y变量中加入这个大小的均匀随机噪声。对回归拟合后的数据副本添加噪声，只影响散点图的外观。这在绘制取离散值的变量时很有用。
>
> **logistic** bool, 可选
> 如果为`True`，则假定y是一个二元变量，并使用统计模型来估计logistic回归模型。并且由于使用引导程序计算回归线周围的置信区间，您可能希望将其关闭获得更快的迭代速度（使用参数`ci=None`）或减少引导重新采样的数量`(n_boot)`。

```python
# 制作具有性别色彩的自定义调色板
pal = dict(Up= "#6495ED", Down= "#F08080")
# 买卖随开盘价与涨跌变化
g = sns.lmplot(x= "open", y= "Buy_Sell", col= "Up_Down", hue= "Up_Down", 
               data=dataset,
               palette=pal, 
               y_jitter= .02, # 回归噪声
               logistic= True)# 逻辑回归模型
```

![image-20210602134659877](https://i.loli.net/2021/06/02/Im5wrFBy7ONQs3g.png)

## 二、线性回归图regplot

`Lmplot()`与`regplot()`与两个函数之间的主要区别是`regplot()`接受变量的类型可以是**numpy数组、pandas序列(Series)**。或者直接对**data传入pandas DataFrame**对象数据。而`lmplot()`的**data**参数是必须的，且变量必须为字符串。

### 线性回归

绘制连续型数据并拟合线性回归模型。

> **fit_reg** bool，可选
> 如果为`True`，则估计并绘制与`x` 和`y`变量相关的回归模型。
>
> **ci** int in [ **0，100** ]或None，可选
> 回归估计的置信区间的大小。这将使用回归线周围的半透明带绘制。置信区间是使用自举估算的；对于大型数据集，建议将此参数设置为`"None"`，以避免该计算。
>
> **scatter** bool，可选
> 如果为`True`，则绘制一个散点图，其中包含基础观察值（或`x_estimator`值）。

```python
# 绘制线性回归拟合曲线
f, ax = plt.subplots(figsize=(8,6))
sns.regplot(x="Returns",
            y="volume",
            data=dataset,
            fit_reg=True,
            ci = 95,
            scatter=True, 
            ax=ax)
```

![](https://i.loli.net/2021/06/02/Gtapq31unViFP8c.png)

除了可以接受连续型数据，也可接受离散型数据。将连续变量离散化，并在每个独立的数据分组中对观察结果进行折叠，以绘制中心趋势的估计以及置信区间。

> **x_estimator** callable映射向量->标量，可选 
> 将此函数应用于的每个唯一值，`x`并绘制得出的估计值。当`x`是离散变量时，这很有用。如果`x_ci`给出，该估计将被引导，并得出一个置信区间。
>
> **x_bins** int或vector，可选 
> 将`x`变量分为离散的`bin`，然后估计中心趋势和置信区间。这种装箱仅影响散点图的绘制方式；回归仍然适合原始数据。该参数可以解释为均匀大小（不必要间隔）的垃圾箱数或垃圾箱中心的位置。使用此参数时，表示默认 `x_estimator`为`numpy.mean`。
>
> **x_ci** “ ci”，“ sd”，[ **0，100** ]中的int或None，可选 
> 绘制离散值的集中趋势时使用的置信区间的大小`x`。如果为`"ci"`，则遵循`ci`参数的值 。如果为`"sd"`，则跳过引导程序，并在每个箱中显示观测值的标准偏差。

```python
f, ax = plt.subplots(1,2,figsize=(15,6))
sns.regplot(x="Returns",
            y="volume",
            data=dataset,
            x_bins=10,
            x_ci="ci",
            ax=ax[0])
# 带有离散x变量的图，显示了唯一值的方差和置信区间：
sns.regplot(x="Returns",
            y="volume",
            data=dataset,
            x_bins=10,
            x_ci='sd',
            ax=ax[1])
```

![image-20210602134835083](https://i.loli.net/2021/06/02/XpWiVIurygHmkh4.png)

### 多项式回归

> **order** : int, 可选
> 多项式回归，设定指数

```python
sns.regplot(x="open",
            y="close",
            data=dataset.loc[dataset.Up_Down == "Up"],
            scatter_kws={"s": 80},
            order=5, ci=None)
```

![image-20210602134902456](https://i.loli.net/2021/06/02/dkyHv3EMnDQsXg6.png)

### 逻辑回归

> **{x,y}_jitter** floats, 可选
> 将相同大小的均匀随机噪声添加到x或y 变量中。拟合回归后，噪声会添加到数据副本中，并且只会影响散点图的外观。在绘制采用离散值的变量时，这可能会有所帮助。
>
> **n_boot** int, 可选
> 用于估计`ci`的`bootstrap`重样本数。默认值试图平衡时间和稳定性。

```python
sns.regplot(x= "volume", 
            y= "Increase_Decrease",
            data=dataset,
            logistic=True, 
            n_boot=500, 
            y_jitter=.03,)
```

![image-20210602134923489](https://i.loli.net/2021/06/02/6vIWgd1C7sfZ3iJ.png)

### 对数线性回归

> **logx** bool, 可选
> 如果为`True`，则估计`y ~ log(x)`形式的线性回归，但在输入空间中绘制散点图和回归模型。注意x必须是正的，这个才能成立。

```python
sns.regplot(x="open",
            y="volume",
            data=dataset.loc[dataset.Up_Down == "Up"],
            x_estimator=np.mean, 
            logx=True)
```

![image-20210602134945168](https://i.loli.net/2021/06/02/QW2bdtEZiMG68xu.png)

### 稳健线性回归

> **robust** 布尔值，可选
> 拟合稳健的线性回归。

```python
sns.regplot(x="open",
            y="Returns",
            data=dataset.loc[dataset.Up_Down == "Up"],
            scatter_kws={"s": 80},
            robust=True, ci=None)
```

![image-20210602135004971](https://i.loli.net/2021/06/02/lIiZRCTGHvupX1d.png)

## 三、线性回归残差图residplot

`residplot()`用于检查简单的回归模型是否拟合数据集。它拟合并移除一个简单的线性回归，然后绘制每个观察值的残差值。通过观察数据的残差分布是否具有结构性，若有则这意味着我们当前选择的模型不是很适合。

### 线性回归的残差

此函数将对**x**进行**y**回归（可能作为稳健或多项式回归），然后绘制残差的散点图。可以选择将最低平滑度拟合到残差图，这可以帮助确定残差是否存在结构

> **lowess** 布尔值，可选
> 在残留散点图上安装最低平滑度的平滑器。

```python
# 拟合线性模型后绘制残差,lowess平滑
x=dataset.open
y=dataset.Returns
sns.residplot(x=x, y=y, 
              lowess=True, 
              color="g")
```

![image-20210602135037264](https://i.loli.net/2021/06/02/yboZIAcefpxUVQu.png)

### 稳健回归残差图

> **robust** bool，可选
> 计算残差时，拟合稳健的线性回归。

```python
sns.residplot(x="open",
              y="Returns",
              data=dataset.loc[dataset.Up_Down == "Up"],
              robust=True,
              lowess=True)
```

![image-20210602135059708](https://i.loli.net/2021/06/02/FUJrTC8nQyh17t3.png)

### 多项式回归残差图

> **order** int，可选
> 计算残差时要拟合的多项式的阶数。

```python
sns.residplot(x="open",
              y="close",
              data=dataset.loc[dataset.Up_Down == "Up"],
              order=3,
              lowess=True)
```

![image-20210602135121652](https://i.loli.net/2021/06/02/K5s8PLIYmjtgDkf.png)

## 四、其他背景中添加回归

### jointplot

`jointplot()`函数在其他更大、更复杂的图形背景中使用`regplot()`。`jointplot()`可以通过`kind="reg"`来调用`regplot()`绘制线性关系。

```python
sns.jointplot("open",
              "Returns",
              data=dataset, 
              kind='reg')
# 设置kind="reg"为添加线性回归拟合
#（使用regplot()）和单变量KDE曲线
```

![image-20210602135149677](https://i.loli.net/2021/06/02/TMK2zyltZpPCdNu.png)

`jointplot()`可以通过`kind="resid"`来调用`residplot()`绘制具有单变量边际分布。

```python
sns.jointplot(x="open", 
              y="close", 
              data=dataset, 
              kind="resid")
```

![image-20210602135214117](https://i.loli.net/2021/06/02/CF6Wh4jOVqYPuXy.png)

### pairplot

给`pairplot()`传入`kind="reg"`参数则会融合`regplot()`与`PairGrid`来展示变量间的线性关系。注意这里和`lmplot()`的区别，`lmplot()`绘制的行（或列）是将一个变量的多个水平（分类、取值）展开，而在这里，`PairGrid`则是绘制了不同变量之间的线性关系。

```python
sns.pairplot(dataset, 
             x_vars=["open", "close"], 
             y_vars=["Returns"],
             height=5, 
             aspect=.8,
             kind="reg");
```

![image-20210602135235556](https://i.loli.net/2021/06/02/UJ3VHNTaty6zfCj.png)