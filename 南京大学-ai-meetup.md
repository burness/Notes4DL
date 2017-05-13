## 南京大学 俞杨 强化学习与人工智能

### Goal

learn from interactions to adapt to the environment

Reward + action

example: 训练小狗与计算机智能

agent => (action/decision reward/state) environment

[image1](image1)

Agent goal: Learn ta policy to max long-term reward

### RL Vs SL

[image2](image2)

 - exploration
 - sequential decision
 - delayed callback
 - effect the environment

Example: shortest path
[image3](image3)

每一个节点充当状态，每一个节点的边为该状态的可选择的desicion

Example: learning robot skills

多部决策改变状态环境

Example: Stock prediction
个体的决策会影响世界

Example: 推荐和搜索
用户买的商品和推荐出来的有关系，同理搜索也很多， every decision changes the world

Example: 

Example: AlphaGo

Tree Search/ deep neural network/ reinforcement learning

### The history of AI

图灵： Computing machinery and intelligence // Learning machine


### Deep reinforcement leanring

 - 无需复杂的标注信息
 - 深度学习的方法有利于提取环境的状态（Deepmind Deep Q-learning on Atari， minibatch）
 - Example 三维空间中寻找某人

### Next?

推理 + 知识 + 学习 ？

Towards powerful search method
Derivative-free RL
没有梯度的强化学习
searching params



## 子午投资 朱家琪 深度强化学习在日内交易中的应用

### 量化投资中的应用

 - 信息处理
 - 策略辅助
 - 智能投资

### 动量因子



## 清华大学 刘知远 社会计算与表示学习

### 社会计算的研究对象

 - 社会网络
 - 媒体信息
挑战： 信息多源异构，难以建立语义联系

### 基于符号的表示方案
 - onehot representation，相似度计算无法计算
 - N个节点的网络， 邻接矩阵
 - 分布式表示方案 （distributed representation，稠密、实值、低维向量），优势：异质对象间的语义计算，表层数据=>深层语义

### 网络表示学习方案
 - word2vec
 - 网络表示学习，利用网络的结构信息学习每个节点的低维表示（DeepWalk：给定图，然后随机游走，形成序列（中间点和附近点），然后利用word2vec）
 - LINE （一阶临近度：直接相连和二阶临近度：共享的邻居）
 - node2vec （随机游走策略，增加随机游走策略：宽度优先搜索，深度优先搜索）
 - 网络表示学习与矩阵分解关系（Spectral Clustering， DeepWalk和GraRep等同于某种矩阵分解）

### 引入外部信息的网络表示学习

 - 引入文本信息，将节点文本信息迁入网络表示学习 （Text-Associated DeepWalk）
 - 语境感知问题 （用户的表示应该和上下文有关（Context-Aware）的，利用文本信息建模影响（CNN），是建模更准确）
 - 引入标签信息 （Max-Margin DeepWalk）

### 网络表示学习应用

 - 网络表示学习成低维向量后，社交网络与用户的移动轨迹联合建模，使用循环神经网络对用户轨迹建模


 ### 展望

 多种异质数据encode到统一向量空间

  - 探索特殊社会网络表示学习
  - 探索动态网络下的表示学习
  - 改进社会计算典型人物
  - 知识驱动的社会计算

## 王昊奋 狗尾草 聊天机器人技术面面观

### 业界几种不同聊天机器人技术比较
 - 2016年--聊天机器人元年
 - 在线客服、娱乐、教育、个人助理、智能问题
 - siri 2010/ IBM Waston 2011/ 微软小冰 2014/ MessagerM 2015/ Google Allo 2016/ 公子小白
 - task\open-domain\QA\社交网络（Facebook Messenger M, Google Now, Allo）
 - 知识图谱/深度学习，增强学习，迁移学习

### 聊天机器人新的技术挑战
### 情商更高的机器人
 
 - 语义理解
 - 知识表示
 - QA
 - 智能对话
 - 用户建模

聊天机器人到强人工智能所面临的挑战

 - 长对话
 - 开域与闭域
 - 上下文
 - 个性化对答
 - 意图和多样性
 - 情商
 ...

公子小白
上帝模式->小白模式（1，web端的学习；2，用户调教模式；3，纠错，反问）
用户要求：高娱乐性

### AI+新一代人工智能

chappie
holoarea 琥珀 虚颜















