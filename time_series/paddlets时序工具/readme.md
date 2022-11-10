## PaddleTS简介

PaddleTS 是一个易用的深度时序建模的Python库，它基于飞桨深度学习框架PaddlePaddle，专注业界领先的深度模型，旨在为领域专家和行业用户提供可扩展的时序建模能力和便捷易用的用户体验。PaddleTS 的主要特性包括：

- 设计统一数据结构，实现对多样化时序数据的表达，支持单目标与多目标变量，支持多类型协变量
- 封装基础模型功能，如数据加载、回调设置、损失函数、训练过程控制等公共方法，帮助开发者在新模型开发过程中专注网络结构本身
- 内置业界领先的深度学习模型，包括NBEATS、NHiTS、LSTNet、TCN、Transformer, DeepAR（概率预测）、Informer等时序预测模型，以及TS2Vec等时序表征模型
- 内置多样化的数据转换算子，支持数据处理与转换，包括缺失值填充、异常值处理、归一化、时间相关的协变量提取等
- 内置经典的数据分析算子，帮助开发者便捷实现数据探索，包括数据统计量信息及数据摘要等功能
- 自动模型调优AutoTS，支持多类型HPO(Hyper Parameter Optimization)算法，在多个模型和数据集上展现显著调优效果
- 第三方机器学习模型及数据转换模块自动集成，支持包括sklearn等第三方库的时序应用
- 支持在GPU设备上运行基于PaddlePaddle的时序模型



**官方地址**：https://github.com/PaddlePaddle/PaddleTS



**视频学习资料**：

**时序建模算法库PaddleTS技术与实践**

https://aistudio.baidu.com/aistudio/education/group/info/27786



## 关于 PaddleTS

具体来说，PaddleTS 时序库包含以下子模块：

| 模块                                                         | 简述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**paddlets.datasets**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/datasets/overview.html) | 时序数据模块，统一的时序数据结构和预定义的数据处理方法       |
| [**paddlets.autots**](https://paddlets.readthedocs.io/en/latest/source/modules/autots/overview.html) | 自动超参寻优                                                 |
| [**paddlets.transform**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/transform/overview.html) | 数据转换模块，提供数据预处理和特征工程相关能力               |
| [**paddlets.models.forecasting**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/models/overview.html) | 时序模型模块，基于飞桨深度学习框架PaddlePaddle的时序预测模型 |
| [**paddlets.models.representation**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/models/representation.html) | 时序模型模块，基于飞桨深度学习框架PaddlePaddle的时序表征模型 |
| [**paddlets.models.anomaly**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/models/anomaly.html) | 时序模型模块，基于飞桨深度学习框架PaddlePaddle的时序异常检测模型 |
| [**paddlets.pipeline**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/pipeline/overview.html) | 建模任务流模块，支持特征工程、模型训练、模型评估的任务流实现 |
| [**paddlets.metrics**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/metrics/overview.html) | 效果评估模块，提供多维度模型评估能力                         |
| [**paddlets.analysis**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/analysis/overview.html) | 数据分析模块，提供高效的时序特色数据分析能力                 |
| [**paddlets.ensemble**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/ensemble/overview.html) | 时序集成学习模块，基于模型集成提供时序预测能力               |
| [**paddlets.utils**](https://paddlets.readthedocs.io/zh_CN/latest/source/modules/backtest/overview.html) | 工具集模块，提供回测等基础功能                               |



