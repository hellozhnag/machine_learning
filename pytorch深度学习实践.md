# overview
## 深度学习在ai中的位置
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713344650741-3a482793-5833-4546-b5a8-890a7ffd1f02.png" alt="image.png" style="zoom: 50%;" />

## 如何设计学习系统 ?
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713347407965-ce4d9de8-2114-43a8-ab49-5dcd6f2cbd2d.png" alt="image.png" style="zoom: 50%;" />

- 基于规则
   - 比如早期设计一个求原函数的系统，通常是在程序中写些内置的常见求原函数规则，通过对输入根据这些规则变来计算原函数，所以并不一定能求出原函数，并且随着规则变多，代码也难以维护
- 经典的机器学习方法
   - 对输入手动提取特征 用向量表示，并与输出建立映射函数
- 表示学习
   - 主要是实现降维
   - 学习器面临维度诅咒：根据大数定律，feature越多，我们需要更多的数据集来贴合真实数据，（假设一个feature需要10个数据量，，那么2个feature就需要10*10=100个数据 ... n个feature就需要10n  个数据，而带标记的数据集是很贵的）
   - 我们通常要对feature降维，（比如将n维降低成3维），<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713346614709-766e36b9-eef2-435f-ae74-d89139946fad.png" alt="image.png" style="zoom: 33%;" />
- 深度学习
   - 深度学习是一种有监督的机器学习方法
   - 我们只提取简单的特征就训练（比如输入一张图片我们就把所有像素组成向量拿过来；如果输入是个波形，就把波形这个序列拿过来；如果是数据库中的表，我们就把表中离散数据转换成连续数据，然后拿过来...)
   - 但是我们要设置一个额外的层来提取特征，然后再接入学习器（一般是多层神经网络）
   - 和表示学习区别：表示学习中的Features层和学习器层是分开的训练的，一般features层使用无监督学习没标签的（通过聚类、降维、关联分析等方法来挖掘数据的内在特征和模式。），而学习器是有监督的打标签的；但是深度学习的simple feature层和学习器层是一起的，因此也叫端到端的训练过程，我们建立一个深度神经网络来实现输入到输出的过程
- 基于规则学习和表示学习的区别
   - 前者基于规则，后者基于数据
   - <img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713347495522-7a786da0-5a4b-4073-be5a-33ca6162b2b1.png" alt="image.png" style="zoom:50%;" />

## 传统的SVM受到了哪些挑战？

- 人工提取特征是有限的，我们能想到一些特征提取方法，但是还有很多想不到的
- svm在处理大数据集的效果不是很好
- 越来越多的应用需要处理无结构数据

## 神经网络的发展
神经网络的灵感来源于神经科学，现在的深度学习来源于数学与工程学
所以现在的神经网络已经不用神经科学那一套了，用的是数学与工程学
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713348759975-e1b39006-d647-416d-af2e-bb9bf60e20dd.png" alt="image.png" style="zoom:40%;" />
真正让神经网络工作起来的算法：反向传播（核心是计算图）

### 反向传播
前馈计算e，同时也把导数公式求出来
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713349417075-27f58e49-90b2-4a30-9c3e-05cdbe472036.png" alt="image.png" style="zoom: 33%;" />
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713349239328-b762010f-6424-46b3-9428-e7c481b66cd6.png" alt="image.png" style="zoom:33%;" />
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713349413922-d3252067-2c98-4ad1-9467-2a616b47b7bf.png" alt="image.png" style="zoom:33%;" />
可以在图里传播导数，然后把这些导数一一算出来，所以我们不用去计算最终的损失关于权重的偏导，就使的我们的模型具有很好的灵活性，可以构造非常复杂的计算图

# 线性模型
> dataset - > select model  -> training  -> inferring

## example：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713353162681-71b2cf64-5015-4686-8fe2-585d54ba5f29.png" alt="image.png" style="zoom:33%;" />
模型看不到y值的俩阶段：测试阶段、上线后推理阶段

<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713353233241-5739cbea-9cc1-4836-acf1-5f3ce2960174.png" alt="image.png" style="zoom:33%;" />
测试集用来评判模型的泛化能力

<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713354391984-3c715280-2726-4250-9450-ff115fe78285.png" alt="image.png" style="zoom:33%;" />
一般我们的训练集并不能真实贴合真实的数据，导致我们在训练过程中出现过拟合（比如我们把噪声也进行了学习)，所以我们为了保证模型的泛化能力，我们把数据集分成两部分：训练集、测试集

这里我们通过观察数据集 选择最简单的线性模型
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713355356185-20ae049d-931e-4e6a-9add-81a796cdc82a.png" alt="image.png" style="zoom:33%;" />
损失：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713355469059-7f714591-9f46-46bd-96c9-2efb151afbee.png" alt="image.png" style="zoom:33%;" />

```python
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
def forward(x):
    return x * w
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)
w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    print('w=', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)

print("w_list=", w_list)
print("mse_list=", mse_list)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
```
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713358726358-a356b150-6fa1-4918-8d6c-a7b8051a9a15.png" alt="image.png" style="zoom: 50%;" />
一般也不会去用穷举，将来也不会用权重表示x轴，会用训练轮数表示

# 梯度下降算法
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713360367888-ce9ce1e6-2c7f-494c-b7f9-ef277f86c3f8.png" alt="image.png" style="zoom: 50%;" />
并不一定能得到全局最优
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713360416674-3e64ec4c-c227-4f50-b2ff-d5b0727e2c33.png" alt="非凸函数：存在局部最优" title="非凸函数：存在局部最优" style="zoom: 33%;" />

> 深度学习中局部最优很少遇到，常遇到的问题主要是鞍点

<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713360918343-e2755af6-c29b-4940-8766-94cd81d81437.png" alt="image.png" style="zoom:50%;" />

梯度下降算法：

```python
import matplotlib.pyplot as plt

# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# initial guess of weight
w = 1.0


# define the model linear model y = w*x
def forward(x):
    return x * w


# define the cost function MSE
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


# define the gradient function  gd
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


epoch_list = []
cost_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val  # 0.01 learning rate
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)

print('predict (after training)', 4, forward(4))
plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
```
## 随机梯度下降算法

```python
import matplotlib.pyplot as plt
 
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
w = 1.0
 
def forward(x):
    return x*w
 
# calculate loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2
 
# define the gradient function  sgd
def gradient(x, y):
    return 2*x*(x*w - y)
 
epoch_list = []
loss_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    for x,y in zip(x_data, y_data):
        grad = gradient(x,y)
        w = w - 0.01*grad    # update weight by every grad of sample of training set
        print("\tgrad:", x, y,grad)
        l = loss(x,y)
    print("progress:",epoch,"w=",w,"loss=",l)
    epoch_list.append(epoch)
    loss_list.append(l)
 
print('predict (after training)', 4, forward(4))
plt.plot(epoch_list,loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show() 
```
事实证明,随机梯度下降在dl中是非常有效的
随机梯度下降在dl中用的较多,有可能帮助我们越过鞍点(因为数据是有噪声的)
梯度下降是可以并行的(计算每一个梯度是可以并行计算的,因为w一样), 随机梯度下降是串行的(下一个样本的w依赖上一个样本的w)

|  | 梯度下降 | 随机梯度下降 |
| --- | --- | --- |
| 性能 | 低 | 高 |
| 时间复杂度 | 低 | 高 |

所以我们一般选择折中,使用小批量随机梯度下降算法(mini-batch),每次选择小批量进行梯度下降,各个小批量之间是随机梯度下降
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713364332763-5fa9212e-f7d5-4a34-a742-c9f6a24fbbbd.png" alt="image.png" style="zoom: 33%;" />

# 反向传播
在面对复杂的网络中,我们不可能一个一个去求导解析式
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713365764188-7c332c15-d7fc-431b-9c91-46048f0865a9.png" alt="image.png" style="zoom: 33%;" />

## 计算图:
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713366058742-843222b2-2e58-4e17-bee7-4cb6a09bde5b.png" alt="image.png" style="zoom:33%;" />
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713366140242-4b54d8c8-3aff-4c2a-91a9-8f1935c9a86d.png" alt="image.png" style="zoom: 33%;" />
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713366246073-7da0ca95-2ea9-40d9-94db-3cd68d027244.png" alt="image.png" style="zoom:33%;" />
如果我么对公式化简,发现无论多复杂的网络都能化简成1层
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713366385188-78469bf9-fc11-42b4-915b-7c58fec90c9e.png" alt="image.png" style="zoom:33%;" />
所以我们要对每一层的输出都要加一层的非线性变换函数
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713366533003-266f05ab-56c7-4748-af9e-e9cb5bb913b5.png" alt="image.png" style="zoom:33%;" />

前向传播过程
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713366995637-eeada3e7-31e9-4086-86c2-36aecd5c8c66.png" alt="image.png" style="zoom:33%;" />
反向传播过程
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713367066173-fc6281be-dd2f-49de-be94-cb954055f475.png" alt="image.png" style="zoom:33%;" />

## 线性模型的计算图过程
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713367531829-33f5e6d4-90e8-489a-a526-c069a219c6f9.png" alt="image.png" style="zoom:33%;" />

## 在pytorch中使用
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713367701718-69aad943-14f4-43bc-ad45-3daf92cc0334.png" alt="image.png" style="zoom:33%;" />

> w是Tensor(张量类型)，Tensor中包含data和grad，data和grad也是Tensor。grad初始为None，调用l.backward()方法后w.grad为Tensor，故更新w.data时需使用w.grad.data。如果w需要计算梯度，那构建的计算图中，跟w相关的tensor都默认需要计算梯度。



这段代码在构建计算图,当计算到loss时,计算图就构建出来了
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713367990699-4f455d32-6fc7-4820-bdb2-b9d5af291978.png" alt="image.png" style="zoom: 33%;" />

```python
import torch
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
w = torch.tensor([1.0]) # w的初值为1.0
w.requires_grad = True # 需要计算梯度
 
def forward(x):
    return x*w  # w是一个Tensor
 
 
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2
 
print("predict (before training)", 4, forward(4).item())
 
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        #这行代码是在构建计算图
        l =loss(x,y) # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        #执行backward,会自动把链路上所有需要梯度的地方都求出来,求完并存入变量中,(比如存到w中),
        #然后释放计算图,下一次loss计算重新构建计算图
        l.backward() #  backward,compute grad for Tensor whose requires_grad set to True
        #itm也是获取tensor中的标量并转换成python类型，也是为了防止产生计算图
        print('\tgrad:', x, y, w.grad.item())
        #w.grad也是tensor，tensor直接计算会构建计算图，所以我们要对他的dataset操作，直接取值
        #在构建计算图中我们都是直接使用张量，但是我们在权重更新时要使用data
        w.data = w.data - 0.01 * w.grad.data   # 权重更新时，注意grad也是一个tensor

        #把权重里面的梯度数据清零
        #因为上一步重新计算权重后梯度还在，如果不清零就会累加，如下图
        w.grad.data.zero_() # after update, remember set the grad to zero
 
    print('progress:', epoch, l.item()) # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）
 
print("predict (after training)", 4, forward(4).item())
```
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713369133164-9d2782be-ffcd-48da-b875-c4d534a6d08f.png" alt="image.png" style="zoom:33%;" />


# 用pytorch实现线性回归
步骤：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713412045378-01a7eeef-1d8b-4f8a-a8b2-7ba7a057d41e.png" alt="image.png" style="zoom:33%;" />

- 准备数据集
- 设计模型
- 构造损失函数和优化方法
- 反复循环训练
## 模型设计
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713493330117-4195f94a-8777-43a8-b488-45a8e898c00c.png" alt="image.png" style="zoom:50%;" />
loss最终必须是标量，y^可以任意维度向量
模版：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713494668746-69a99788-11a2-459b-947e-ed1647b6ad81.png" alt="image.png" style="zoom:50%;" />
通过继承torch.nn.Module类来构造我们自己的神经网络，forword是前向传播，反向传播自动帮我们实现了，torch.nn中内置了很多的计算块并也都实现了反向传播 供我们使用
只要想构造计算图，就需要继承Module
也可以在在Functions类中构造自己的计算块

<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713495188724-fcedf60e-9c68-4ab0-a5f2-5b631e4bb25b.png" alt="image.png" style="zoom:50%;" />
Module会把forward函数放在__call()__中，linear也是如此， 也是继承了Module类，因此也是可callable的
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713495787714-001ffe2f-2baa-49fb-8a02-97ea8a9efd50.png" alt="image.png" style="zoom:50%;" />
上面的三种w x乘法运算，都行主要是保证矩阵维度能凑上就行

## MSELoss
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713496281693-33fb995b-52e9-472a-94ad-8091b564a724.png" alt="image.png" style="zoom:67%;" />

- 当 size_average=True（这是默认设置）时，MSELoss会计算整个批次内每个样本的均方误差之和，然后再除以样本数量（batch size）。这意味着最终得到的损失值是批次内所有样本损失的平均值。
- 当 size_average=False 时，则不会对损失进行平均化处理，而是直接将所有样本的均方误差加起来，得到的结果是整个批次损失的总和，而不是平均损失。

torch.nn.MSELoss也跟torch.nn.Module有关，参与计算图的构建，
## 优化器SGD
不管我们的模型多复杂都能够通过 .parameters方法找到所有权重
在线性模型中我们的权重是w b
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713496783147-6cdc6dd1-0e61-46c3-b302-46e82233c510.png" alt="image.png" style="zoom:50%;" />
torch.optim.SGD与torch.nn.Module无关，不参与构建计算图。

## 训练过程
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713496984212-b8bca884-d827-42c5-ba76-4dceeca84828.png" alt="image.png" style="zoom:50%;" />

## 代码
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713497172451-400a4dcf-486e-4b27-8d66-c543e7f5f4ca.png" alt="image.png" style="zoom:67%;" />

```python
import torch

# prepare dataset
# x,y是矩阵，3行1列 也就是说总共有3个数据，每个数据只有1个特征
# 这一步是准备数据集
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# design model using class
"""
our model class should be inherit from nn.Module, which is base class for all neural network modules.
member methods __init__() and forward() have to be implemented
class nn.linear contain two member Tensors: weight and bias
class nn.Linear has implemented the magic method __call__(),which enable the instance of the class can
be called just like a function.Normally the forward() will be called 
"""


# 当实例化一个继承自 torch.nn.Module 的神经网络模型后，模型内部的层（如线性层、卷积层等）会自动初始化其权重和偏置。
class LinearModel(torch.nn.Module):
    def __init__(self):
        # 这一步必须有
        super(LinearModel, self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

# construct loss and optimizer
# criterion = torch.nn.MSELoss(size_average = False)
criterion = torch.nn.MSELoss(reduction='sum')
# 这个方法主要用来方便地获取模型的所有参数，以便将其传递给优化器（如 torch.optim.SGD），进行模型训练时的权重更新。
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training cycle forward, backward, update
for epoch in range(100):
    # 我们能这样操作是因为Module把forward方法放在了 __call()__中
    y_pred = model(x_data)  # forward:predict
    loss = criterion(y_pred, y_data)  # forward: loss
    print(epoch, loss.item())

    optimizer.zero_grad()  # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
    loss.backward()  # backward: autograd，自动计算梯度
    optimizer.step()  # update 参数，即更新w和b的值

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
```
# 逻辑斯蒂回归
这个模型是做分类的
二分类：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713504338909-a83b1069-3f41-444e-a1cb-a59ef5495ea3.png" alt="image.png" style="zoom:50%;" />
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713504435802-e7cfadcf-1cb1-4bc4-8954-b429197c844a.png" alt="image.png" style="zoom:50%;" />
这个函数是sigmoid中最经典的函数，一般也把这个函数叫做sigmoid函数
sigmoid函数如下：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713504669782-b8231742-718c-4c11-921c-0c998b8d26fa.png" alt="image.png" style="zoom:50%;" />
logistic函数在计算图中：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713504802332-bee2019e-6901-44ff-b15a-4933b281751f.png" alt="image.png" style="zoom:50%;" />
"Affine model" 在深度学习领域通常是指具有仿射变换特性的模型结构或者模块。仿射变换是一种线性变换（包括旋转、缩放、平移等）加上一个常数项，可以表示为：y = Ax + b，其中 A 是一个矩阵，x 是输入向量，b 是一个偏置向量，y 是输出向量。
在深度学习的上下文中，仿射层（Affine Layer）或仿射变换层通常指的就是全连接层（Fully Connected Layer），因为全连接层本质上就是对输入向量进行一次仿射变换的过程。例如，在神经网络中，线性层（Linear layer）就是一种仿射变换，它对输入特征进行加权求和后再加上一个偏置项。
二分类的损失函数：<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713507839388-02d387ff-d8a3-4be3-8366-7e19fcf4c58a.png" alt="image.png" style="zoom:50%;" />
交叉熵损失。

关于损失函数的size_average的设置，如果设置为true就会对loss*1/N，这么设置的影响我们的求导数，当为true时我们求的导数也会*1/N，而导数是要和学习率相乘来去更新权重的；所以这个地方为true/false都行，只要我们设置好学习率就行
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713508041574-d1ffd26d-1f97-4a0d-b9f6-3a51409d1758.png" alt="image.png" style="zoom:50%;" />

代码部分：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713508391095-1a6c28a6-35dd-4f7e-8d98-a74c1d7936dd.png" alt="image.png" style="zoom:67%;" />

```python
import torch

# import torch.nn.functional as F

# prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


# design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # y_pred = F.sigmoid(self.linear(x))
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

# construct loss and optimizer
# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training cycle forward, backward, update
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
```

# 处理多维特征的输入
患糖尿病数据：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713509442360-d355b91c-06cf-4aea-ab3c-2779072cbd72.png" alt="image.png" style="zoom:50%;" />
每一行称作记录，每一列称作特征

设计模型、损失函数
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713509569872-677afb06-de4a-4545-90c2-7f16fbd3a64f.png" alt="image.png" style="zoom: 50%;" />

pytorch中的函数都是向量化函数
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713509695604-720c3257-3afd-4d84-99ed-79897d41b8f9.png" alt="image.png" style="zoom:50%;" />

<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713509837875-4e1af9ac-fead-45ae-ae6e-25608f60f629.png" alt="image.png" style="zoom:50%;" />
写成矩阵能够使的并行计算，加快计算速度，如果写成for循环的话计算很慢

我们需要将linear改变为输入为8个维度，输出为1个维度
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713510391009-547ebe8a-03ca-426d-90a1-d898068e9832.png" alt="image.png" style="zoom:50%;" />

也可以输出为2个维度，然后再通过一个linear layer转换成1维度的（即8维空间向量映射到2维空间的向量）
也可以输出为6个维度，然后经过一个linear layer转换为2个维度，最后再经过一个linear layer转换成1个维度
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713510957557-a0366578-f670-4a93-bdf0-3be177b4611b.png" alt="image.png" style="zoom:50%;" />
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713512224120-1a15e11f-a7fd-4899-a21d-36cb79ad7e4b.png" alt="image.png" style="zoom:50%;" />
所以可以理解矩阵是空间线性变换函数
但是经常的空间变换并非线性的，而是比较复杂的非线性的，所以我们希望通过多个这样的线性变换层通过找到最优权重 组合起来 来去模拟这种非线性变换，而神经网络本质也是去寻找一种非线性的空间变换函数

我们在把每个线性层接起来时一定要做![](https://cdn.nlark.com/yuque/__latex/788df1ba344b3092def7590d1be6b4d4.svg#card=math&code=%5Csigma&id=Z0s76)，因为我们的linear是不做非线性的，但是我们在最后加了一个非线性函数就导致我们我们每次的空间变换都引入了非线性，这样的话只需要我们调整好每次的线性变换，我们通过这种方式来去拟合我们真正想要的空间变换。因为我们的目标是去找一个8维空间到1维空间的非线性的空间变换，所以在神经网络中通过引入![](https://cdn.nlark.com/yuque/__latex/788df1ba344b3092def7590d1be6b4d4.svg#card=math&code=%5Csigma&id=CmnZt)，也叫做激活函数，给线性变换增加非线性操作，使的我们可以去拟合相应的非线性变换

具体8D 24D 12D 这里边的数值取值，涉及到超参数搜索尝试，一般来说隐层越多，中间跳跃步骤越多，中间神经元越多，我么对非线性变换学习能力就越强

学习能力越强并非越好，因为学习能力越强容易把噪声也学习进去，数据集的噪声跟真实生产环境中的噪声是不一样的，我们要学习的是数据真值本身的规律，是要让模型有更好的泛化能力

所以我们的开发人员核心能力是：理解计算机基础架构+读文档  =>  泛化能力好，而非扣书本
以此来去理解模型的学习能力强，就好像背书一样，而泛化能力强，就如理解计算机架构+看文档

我们用这样的模型来处理糖尿病的分类：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713513006276-225e9e0a-fd29-4ea1-91c4-b36661f28506.png" alt="image.png" style="zoom:50%;" />

```python
import numpy as np
import torch
import matplotlib.pyplot as plt

# prepare dataset
# 要保证x_data和y_data取出来都是矩阵而非向量
xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])  # 第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
y_data = torch.from_numpy(xy[:, [-1]])  # [-1] 最后得到的是个矩阵

# design model using class

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 输入数据x的特征是8维，x有8个特征
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        # 我们也可以尝试使用别的激活函数
        self.sigmoid = torch.nn.Sigmoid()  # 将其看作是网络的一层，而不是简单的函数使用

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))  # y hat
        return x


model = Model()

# construct loss and optimizer
# criterion = torch.nn.BCELoss(size_average = True)
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []
# training cycle forward, backward, update
for epoch in range(100):
    # 这里我们并没有做minibatch，而是把数据全扔进来了
    # 后续我们通过dataloader来实现minibatch
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
```
也可以在这里修改使用ReLU激活函数
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713514310530-58d00003-cfe6-4695-a7dd-3096c098b76c.png" alt="image.png" style="zoom:50%;" />

# 加载数据集
梯度下降速度快；随机梯度下降性能好，能帮助我们跳过鞍点，但是耗时，所以我们一般采用小批量随机梯度下降来去平衡

epoch&batch-size&iteration
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713587323642-7e2dd42c-4e2f-4c38-a3c3-980823308cc5.png" alt="image.png" style="zoom: 40%;" />

shuffle：打乱
dataloader功能：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713587623198-abc73e90-a99c-456e-8fd2-4ac2bdfacbb4.png" alt="image.png" style="zoom:40%;" />

**dataset：**
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713587921427-9747694c-540a-43a6-8739-c6f94421554f.png" alt="image.png" style="zoom:50%;" />

**dataloader：**
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713587945976-a383a670-6e8b-469b-af86-8f3e54376d41.png" alt="image.png" style="zoom:50%;" />

读取数据的两种方式：

- 如果数据集不大，就全部读进来（比如数据库表等结构化数据）
- 如果数据集大（比如图像、语音数据集...这样的非结构化数据），仅加载必要的数据到内存，每次用到时再去加载相应的数据

<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713589377656-f35d9987-63a0-4705-a1c4-659f5e5f51dc.png" alt="image.png" style="zoom: 50%;" />
num_workers=2表示使用的子进程数量

在pytorch 0.4版本中windows中使用num_workers会报错
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713589596071-f504cf73-185b-43a4-89f0-dd701952c232.png" alt="image.png" style="zoom:50%;" />
解决方法：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713589614296-645496e2-6730-4307-a867-49115e8d5684.png" alt="image.png" style="zoom:50%;" />

```python
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# prepare dataset


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # shape(多少行，多少列)，shape[0]表示行数
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        # 我们的数据集比较简单，所以全部读入
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)  # num_workers 多线程


# design model using class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training cycle forward, backward, update
if __name__ == '__main__':
    for epoch in range(100):
        # 使用enumerate是为了获取迭代次数
        for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
            # 拿取输入x 和标签y
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
```

所以神经网络训练分这四步：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713590251793-c390bbc6-74bd-43c5-89bb-1cd35a8d2a99.png" alt="image.png" style="zoom: 50%;" />

在torchvision中内置了很多了数据集，都派生于dataset，所以都实现了init、getitem、len魔法方法，都能使用dataloader加载
以minst为例：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713590912481-5ba0198e-9a1c-4bf0-852a-22651a6d0015.png" alt="image.png" style="zoom:40%;" />
test_loader的shuffle一般设置为false，保证我们每次在测试集上测试时图像顺序一致

# 多分类问题
## softmax函数

二分类问题我们最终只需要输出一个数，但是多分类我们输出的是一个分布（每个输出 >=0，求和为1）
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713665512904-ccb051b9-d435-47eb-a261-a34ba34ed6ad.png" alt="image.png" style="zoom:40%;" />
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713666187049-1eae2534-eb8f-4283-b28c-f98df45e65ab.png" alt="image.png" style="zoom:40%;" />
softmax函数要实现：

- 最后线性层输出>=0
- 和为1

<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713666386819-6b548445-3d93-46aa-9b2c-9175e7063212.png" alt="softmax函数" title="softmax函数" style="zoom:33%;" />

## 损失函数
nll损失：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713667049235-9cd379aa-05d7-40e8-b348-5996fc97501b.png" alt="image.png" style="zoom:33%;" />
交叉熵损失：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713667162619-c507e0b4-aac4-4fd7-94fe-8c49049ac335.png" alt="image.png" style="zoom:33%;" />
使用交叉熵损失，最后一层不要做激活，做完线性后直接计算交叉熵损失
使用nll损失的话，最后一层要做激活

**举个例子：**
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713667507347-816f5343-22cc-4aa6-be0d-33658533aaee.png" alt="image.png" style="zoom:33%;" />

## MINIST dataset
### 准备数据集
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713668378773-cd3d1226-79a3-4123-9a90-6fa793b724bb.png" alt="image.png" style="zoom:33%;" />
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713668498663-8c302de0-4a35-4992-8702-d590300078a2.png" alt="image.png" style="zoom: 33%;" />
归一化，转换成0-1分布，神经网络对0-1分布的数据训练最好
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713668696649-9914c20b-9ddb-417f-b3be-3c947c90768f.png" alt="image.png" style="zoom:33%;" />

### 设计模型
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713669039569-6babc9cb-a6c8-4193-bbd7-f07c518bceb8.png" alt="image.png" style="zoom:33%;" />
这是一个全连接网络，所有的层线性连接起来
也就是说输入值与每一个输出值任意两个之间都存在权重
也就是说每一个输入节点都参与到下一层任意一个输出节点的计算上

```python
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# prepare dataset

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 归一化,均值和方差

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # 我们把图像直接展开，没有考虑空间特征
        x = x.view(-1, 784)  # -1其实就是自动获取mini_batch
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  # 最后一层不做激活，不进行非线性变换


model = Net()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 获得一个批次的数据和标签
        inputs, target = data
        optimizer.zero_grad()
        # 获得模型预测结果(64, 10)
        outputs = model(inputs)
        # 交叉熵代价函数outputs(64,10),target（64）
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 张量之间的比较运算
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```

# 卷积神经网络-CNN
## 基础篇
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713676727719-fb0d5903-3205-4731-8531-e2bb5f3594f6.png" alt="image.png" style="zoom:40%;" />
我们要注意输入输出的维度
卷积：考虑了图像的空间特征，之前是把图像直接一行一行展成一行，没考虑像素的空间联系。
下采样：减少计算元素数量，减小运算需求
卷积和下采样也叫做特征提取器 feature extraction
最终是要把图像转换成一个只有10个元素的一阶张量

图像分为

- 栅格图像（RGB -> 3个channel）
- 矢量图像（现画）

### 卷积
卷积是对每一个图像块（patch）做的，我们把从图相中取出每一个patch（3*H'*W'）（左右滑动、一行一行遍历去取）分别做卷积后，输出一个新的卷积（C*W*H）（这里的CWH都可能相对卷积前发生变化）
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713679396744-a676fc84-8e07-48ec-9604-97bb20d223ee.png" alt="image.png" style="zoom: 40%;" />
做完卷积后，通道数、高度、宽度可能会变化（假设patch做完卷积后输出4*1*1）
输出通道中的每一个值A（假设图中标红点的那一个）都包含了原来patch块中所有的像素值与权重相乘后求和，所以包含了原始patch的所有信息，所以经过多层卷积后，我们就不断地把这些信息进行融合，因此，只要我们的权重选的合适，A就代表着对原始图像patch中某种特征的扫描，满足这个特征的话A中算出来的值就比较大不满足就较小。
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713680072159-87367118-a410-4ead-9f6c-b662132156be.png" alt="image.png" style="zoom:40%;" />

#### 卷积的运算过程：
卷积核和图像中每一个红框做数乘
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713680654604-54c07884-ac68-476a-93b4-d1dc33953571.png" alt="单通道" title="单通道" style="zoom: 33%;" />

#### 多通道卷积：
通道数=卷积核数
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713681464189-9a4f14f3-4bc7-4bef-8041-523897bc6523.png" alt="image.png" style="zoom:33%;" />
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713681490785-f5513a1d-0c1f-4503-9ebc-ad5c94269086.png" alt="image.png" style="zoom:33%;" />
3*3的卷积核图像宽度高度-2
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713681709165-710e36c1-9e04-48cc-97db-fce0de839735.png" alt="image.png" style="zoom:33%;" />

输出多个通道：使用多个卷积核，然后把每个通道摞起来
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713681858298-b26772fa-5d5e-4d7f-8570-a574f3b79288.png" alt="image.png" style="zoom:33%;" />

- 卷积核的通道数=输入通道数量
- 卷积核的总数=输出通道数量
- 卷积核大小自己定，和图像大小没关系
- 卷积核一般是正方形的、奇数的

构建一个卷积层需要四个维度
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713682157316-df7ec43e-8b5e-4097-babe-3931c00c432d.png" alt="image.png" style="zoom:33%;" />
举例：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713682522450-e82e892c-18a3-4338-9915-2935514265aa.png" alt="image.png" style="zoom: 50%;" />

#### padding：
如果想让卷积前后WH一样：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713682725974-cfcba2b5-465d-45d2-bcdf-748635fa6e32.png" alt="image.png" style="zoom:33%;" />
3*3 kernel就外扩1圈
5*5 kernel就外扩2圈
默认填充0
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713682810219-60187f34-c4ec-4963-b14b-c98407ee1186.png" alt="不需要偏置" title="不需要偏置" style="zoom:50%;" />
卷积本身上依然是一个线性计算

#### stride
可以有效降低卷积后图像的WH
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713682966962-4cb69789-6db2-44e1-8ca6-544f18cef36d.png" alt="image.png" style="zoom:33%;" />
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713683020324-4cec009f-328d-4ba8-ba6c-b80b6b2a4b23.png" alt="image.png" style="zoom:50%;" />

### 下采样
#### 最大池化层
通道数量不变
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713683133583-57b8fad4-b449-4c15-afe8-9c606eae3923.png" alt="image.png" style="zoom:33%;" />
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713683174468-c2b25447-74ae-4941-8347-76161f7f2f9d.png" alt="image.png" style="zoom: 50%;" />

### 一个简单的CNN example
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713683376526-f3a951d4-9cc6-49c7-be16-da831d4e59f8.png" alt="image.png" style="zoom: 33%;" />

### 把之前全连接网络改成CNN
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713683940820-ea1725f7-fee4-4166-a3ec-100355257d44.png" alt="image.png" style="zoom:40%;" />
因为要用交叉熵损失，所以最后一层不做激活

```python
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# prepare dataset

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # flatten data from (n,1,28,28) to (n, 784)

        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # -1 此处自动算出的是320
        # print("x.shape",x.shape)
        x = self.fc(x)

        return x


model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 把模型中所有权重、参数全都迁移放在cuda里
model.to(device)

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        # 把输入迁移到GPU
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))
    return correct / total


if __name__ == '__main__':
    epoch_list = []
    acc_list = []

    for epoch in range(10):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    plt.plot(epoch_list, acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

```
错误率比之前降低了1/3
可以对模型进行进一步改造，比如：最后320->10可以再套几层 ......
### 使用显卡训练

- `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- `model.to(device)`
- 在train中：`inputs, target = inputs.to(device), target.to(device)`
- 在test中：`images, labels = images.to(device), labels.to(device)`

## 高级篇
我们之前的网络都是串行的，即一层一层下去
但是我们可能会使用更为复杂的结构，比如 不同的运行分支 ......

### GoogLeNet
例如GoogLeNet：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713687667050-a6e3409a-1c74-48a7-a001-58a37ab77410.png" alt="image.png" style="zoom: 43%;" />
在构造神经网络时，有些超参数比较难选，比如卷积核的大小......，googleNet提供了几种候选的卷积组合，到时候看哪个效果好选哪个
concatenate：把张量拼接起来
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713688602814-443488b6-3901-4bac-b5a8-5de5dbe16fc2.png" alt="image.png" style="zoom:40%;" />

我们解释下1*1的卷积：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713691548466-faf16a39-ff2f-43bc-bf34-7ec62b269a90.png" alt="image.png" style="zoom:40%;" />
融合了不同通道的相同位置的信息
1*1卷积作用：改变通道的数量，减小计算量
举个例子：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713691926184-65e0a82d-9e1f-4d01-a983-cf3b8477435c.png" alt="image.png" style="zoom:33%;" />

inception的实现：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713699953730-90518e49-959f-4a60-9071-5c50db2d6207.png" alt="image.png" style="zoom:33%;" />
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713700015871-57aeb381-88e2-4aa2-8c75-7e3d932ff49b.png" alt="image.png" style="zoom:33%;" />
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713700053528-f03f2bae-d948-45e9-9b55-db0839f1390c.png" alt="image.png" style="zoom: 50%;" />
为什么dim=1，因为根据 B,C,W,H，我们要沿着C拼接，也就是沿着第一个维度 

```python
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# prepare dataset

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 归一化,均值和方差

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)  # b,c,w,h  c对应的是dim=1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)  # 88 = 24x3 + 16

        self.incep1 = InceptionA(in_channels=10)  # 与conv1 中的10对应
        self.incep2 = InceptionA(in_channels=20)  # 与conv2 中的20对应

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)

        return x


model = Net()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713701042950-8bf5264c-462b-4cf9-b687-78e533c714e7.png" alt="image.png" style="zoom:50%;" />
可以发现网络最后发生了过拟合
我们可以在每次准确率达到极大值时对网络存盘，这样我们就能找到泛化性能最好的网络

### Go Deeper
把3*3的卷积层无限叠加，发现20层比56层效果更好，因为发生了梯度消失
因为梯度如果很小乘着乘着就 ---> 0，那么我们的离输入近的层 权重就没办法得到充分训练
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713701561506-98a9ba9b-fb12-40d5-b53c-cd73c5135de7.png" alt="image.png" style="zoom:50%;" />
怎么解决？Residule Net

### Residule Net
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713701636262-8eef16f1-cf45-49e7-a8aa-3b04c77ab8fb.png" alt="image.png" style="zoom:40%;" />
![image.png](./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713702179193-f806b5e5-6aeb-4fdd-9025-7dd75eb9a21d.png)
residule block也叫残差块。
 在我们的代码中使用残差块

```python
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# prepare dataset

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 归一化,均值和方差

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# design model using class
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # 88 = 24x3 + 16

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(512, 10)  # 暂时不知道1408咋能自动出来的

    def forward(self, x):
        in_size = x.size(0)

        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)

        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


model = Net()
model.to(device)

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```
要注意把网络中的超参数和size计算出来

可以看这篇论文，给了好多种Residule block的构造方式
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713708428012-e0ae1b9c-ca6b-474b-ba0d-42d995875056.png" alt="image.png" style="zoom:40%;" />

### DenseNet
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713708501263-26b6a199-e675-4a6d-b991-a268c6647ac4.png" alt="image.png" style="zoom:50%;" />

### 注意
这里只是介绍怎么实现这些模型，要深入理解dl，要要去看些讨论深度学习的书如花书.....
这里只是介绍pytorch基本用法，想要深入了解，去通读pytorch文档.....
想要提升dl水平，要去复现经典dl论文的代码，理解他的架构......

# 循环神经网络-RNN
和CNN一样都是神经网络的一种......
## 基础篇
回顾深度深度神经网络：
<img src="./images/pytorch%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.images/1713709208989-ec12dcae-2e26-4148-a207-3a6e70b7ee5f.png" alt="image.png" style="zoom:50%;" />


## 提高篇

