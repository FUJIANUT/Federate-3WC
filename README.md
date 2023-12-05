# Federate-3WC BY CHUNMAO Jiang

#模拟联邦环境：将数据分割成几个子集，模拟在联邦学习环境中不同节点拥有的数据。

#分别应用FCM和联邦FCM：首先，在整个数据集上应用传统FCM算法；然后，在每个子集上分别应用FCM算法，并合并结果来模拟联邦FCM的过程。

#绘制对比图：使用不同的颜色或标记来区分由传统FCM和联邦FCM产生的聚类中心。

#说明优势：联邦FCM的优势在于能够在保持数据隐私的同时，有效地进行聚类。这意味着每个节点只需要处理自己的数据，而无需共享原始数据。

#联邦FCM的优势
#数据隐私：每个节点（数据子集）独立处理自己的数据，无需共享原始数据，从而保护了数据隐私。
#减少通信开销：只需在节点间交换汇总信息，而非整个数据集，减少了通信开销。
#灵活性：能够适应于数据分布式存储的环境，特别是在对数据隐私有严格要求的场景中。
