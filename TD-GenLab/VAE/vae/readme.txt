tool.py:
根据config.json内的敌人数据还有waves的格式制作的工具，用于生成测试的样本。(Waves_n.json,Level_n_Summary.csv由该文件生成的数据经过修改制成)
以及模型生成的数据格式转换，使结果能用于游戏内。（Generated_Level_Diff_n_Generated.csv/json）

vae.py
输入：
关卡特征数据：每个波次5种怪物数量：[goblin, goblin_priest, skeleton, slime, slime_king]和金币奖励：Coin Reward (归一化后的值),波次间隔:wave_interval
条件变量：
难度标签和归一化波次编号（难度标签由人工测试并标记）

输出：
生成的关卡数据（Generated_Level_Diff_n_Generated.csv，包括每个波次敌人的数量，奖励的金币和波次间隔）

训练损失函数设计：
总损失函数：L = L_rec + β × L_kl
重建损失 (L_rec)：均方误差(MSE)损失，衡量重建数据与原始数据的差距
L_rec = MSE(x, x̂) = (1/n) × Σ(x_i - x̂_i)²
KL散度正则化 (L_kl)：约束潜在空间分布接近标准正态分布
L_kl = -0.5 × Σ[1 + log(σ²) - μ² - σ²]
β系数 (β=0.005)：平衡重建质量与潜在空间正则化，小值确保模型优先关注数据重建
训练策略：
400训练轮次(epochs)，确保充分收敛
Adam优化器 (学习率=0.001)
批量大小=8 (匹配每关波次数量)
早停机制：监控验证损失防止过拟合

使用：输入样本位置以及需要生成的难度值然后运行文件，将输出的关卡csv用tool.py转换之后就能用于游戏

优点：难度可控，且波次之间难度设计更合理，确保敌人的分布更合理，能根据难度调整节奏；
模型轻量高效，由pytorch实现，生成快速

未来修改方向：从map-based拓展到game-based，将融合玩家行为数据实现个性化自适应难度生成。