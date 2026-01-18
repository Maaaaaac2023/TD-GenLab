# Game Frontend相关

## About

In this tower defence game prototype inspired by KingdomRush, you can not only build different towers using coins earned by defeating waves of enemies, but control a character with skills to battle.

- Click on Empty Tile: Build towers
- Click on Home Tile: Upgrade towers
- Press W/A/S/D: Move the character
- Press 1/2: Release character skills


Short descriptions about the core functional classes which forms multiple extensible gameplay systems are listed below, notice that all managers are derived from singleton `Manager` base class

- General Game Management
  - `GameManager`: Maintains game main loop that handles all updating, rendering and inputs
  - `ProcessManager`: Maintains the in-game runtime data and offers related interfaces
- Infrastructure Classes
  - `StateMachine`: Acts as component of entity to control its changing of states derived from `State`
  - `ObjectPool`: Generic template container to reuse entities, can automatically expand and shrink
  - `Vector2`: Basic 2D vector with mathematical operators overloaded
  - `Timer`: Will trigger self-defined callback function when time out
  - `Animation`: Can be rendered with rotation, also offers spritesheet cutting
- Resource Assets Loading
  - `ResourceManager`: Loads assets by paths into particular resource pools including textures, sounds, musics and fonts
- Configuration Files Loading
  - `ConfigManager`: Loads json and csv files including configs for basic game properties, enemy waves info and tilemap info
- Tile-based Map
  - `Tile`: Struct of single tile that forms tilemap
  - `Map`: Stores basic tilemap and other info including home and enemy spawn points
  - `Route`: Set of indices of consecutive acyclic tiles on which is for enemies to move
- Enemies and Their Drops
  - `Enemy`: Organized by `EnemyManager`, aims at attacking home
    - `Slime`: Derives from `Enemy`, weak vitality and slow
    - `SlimeKing`: Derives from `Enemy`, medium vitality and slow, able to heal itself
    - `Skeleton`: Derives from `Enemy`, medium vitality and fast
    - `Goblin`: Derives from `Enemy`, medium vitality and fast
    - `GoblinPriest`: Derives from `Enemy`, strong vitality and slow, able to heal in range
  - `Drop`: Organized by `DropManager`, collides with Player with particular effects
    - `Coin`: Derives from `Drop`, picked up by player to earn coins
  - `Wave`: Struct organized by `WaveManager` including multiple enemy spawn events
- Towers and Their Bullets
  - `Tower`: Organized by `TowerManager`, can be built for attacking and upgraded by coins
    - `Archer`: Derives from `Tower`, cheap price and shoots with arrows 
    - `Axeman`: Derives from `Tower`, medium price and shoots with axes
    - `Gunner`: Derives from `Tower`, expensive price and shoots with shells
  - `Bullet`: Organized by `BulletManager` and collides with `Enemy` with damage and effects
    - `Arrow`: Derives from `Bullet`, attacks single enemy with high frequency
    - `Axe`: Derives from `Bullet`, attacks single enemy with slowdown effect
    - `Shell`: Derives from `Bullet`, attacks multiple enemies in range with high damage
- User Interface
  - `UIManager`: Offers general rendering methods and organizes all the UI components
    - `StatusUI`: Renders the real-time status of health, coin number and cooldown of player skills
    - `TowerPanel`: Derives `TowerBuildPanel` and `TowerUpgradePanel` for building or upgrading towers
    - `GameOverUI`: Notice whether win or not when game is over
- Controlable Character
  - `Player`: Organized by `PlayerManager`, controled by state machine
    - `PlayerDragon`: Derives from `Player`, can attack enemy by skills including flash and impact

## Modify

This prototype is designed data-driven, you can create your personal experience by editing the corresponding files below in `root\KingdomRushLite\Data\` after you clone this repo locally (Run `Main.cpp` under required IDE mode)

- `Map.csv`: Defines tilemap by denoting each tile in `a\b\c\d` form, see `Tile.h`
- `Configs.json`: Defines window resolution and stats of player, towers and enemies
- `Waves.json`: Defines the details of enemy waves spawned in game

```javascript
// List of spawn waves for series of enemies
[
    // Wave 0
    {
        // Wave properties
        "coin_rewards":     300,
        "wave_interval":    10,
        // List of spawn events for one enemy of particular type
        "spawn_list":
        [
            // Event 0 
            {
                // Event properties
                "event_interval":   1,
                "spawn_point":      2,
                "enemy_type":       "slime"
            },
            // Event n
            {
            }
        ]
    },
    // Wave n
    {
    }
]
```

## Dependency

| Lib                                                          | Version                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [cJSON](https://github.com/DaveGamble/cJSON)                 | cJSON-1.7.18                                                 |
| [SDL](https://github.com/libsdl-org/SDL)                     | SDL2-devel-2.30.2-VC                                         |
| [SDL_image](https://github.com/libsdl-org/SDL_image)         | SDL2_image-devel-2.8.2-VC                                    |
| [SDL_gfx](https://www.ferzkopp.net/wordpress/2016/01/02/sdl_gfx-sdl2_gfx/) | [SDL2_gfx-1.0.4-VC](https://github.com/giroletm/SDL2_gfx/releases/tag/release-1.0.4) |
| [SDL_ttf](https://github.com/libsdl-org/SDL_ttf)             | SDL2_ttf-devel-2.22.0-VC                                     |
| [SDL_mixer](https://github.com/libsdl-org/SDL_mixer)         | SDL2_mixer-devel-2.8.0-VC                                    |


Notice that the depended libraries are all in MSVC version, if you want to build the project by MinGW, remember to replace the dependencies with corresponding versions


# Stable Diffusion相关

1、提取原始地图Mask，放入stable diffusion用于segmentation（参考ppt第22页的图片），得到丰富的背景map（参考第23页ppt的图片）
2、得到48*48的decorate tile，比如石头（参考22页下方的图片与游戏asset内的资源），用于装饰游戏地图

# VAE相关

## 概述

本项目使用条件变分自编码器（Conditional Variational Autoencoder, CVAE）实现基于难度的关卡自动生成系统。通过训练模型学习现有关卡的设计模式，能够根据指定的难度值生成符合游戏设计原则的新关卡，确保难度递增和敌人分布的合理性。

## 项目结构

```
VAE/vae/
├── train.py              # 模型训练脚本
├── test.py               # 关卡生成脚本（推理）
├── utils/
│   └── convert_utils.py  # 数据格式转换工具（JSON ↔ CSV）
├── Level_1_Summary.csv   # 训练数据：关卡1摘要
├── Level_2_Summary.csv   # 训练数据：关卡2摘要
├── Level_3_Summary.csv   # 训练数据：关卡3摘要
└── cvae_model.pth        # 训练好的模型权重（训练后生成）
```

## 核心组件

### 1. 数据转换工具 (`utils/convert_utils.py`)

提供三种工作模式，支持数据格式的双向转换：

- **模式1：生成新关卡** - 根据难度等级生成CSV和JSON文件
- **模式2：json2csv** - 将游戏JSON格式转换为训练用的CSV格式
- **模式3：csv2json** - 将生成的CSV格式转换回游戏可用的JSON格式

**命令行使用：**

```bash
# JSON转CSV（准备训练数据）
python convert_utils.py json2csv Waves.json -o Level_1_Summary.csv -d 1

# CSV转JSON（生成游戏数据）
python convert_utils.py csv2json Generated_Level_Diff_0.15.csv -o Waves.json
```

### 2. 训练脚本 (`train.py`)

**功能：**

- 加载预处理训练数据（`Level_1/2/3_Summary.csv`）
- 训练CVAE模型并保存权重
- 保存数据标准化器（scaler）和特征列表供推理使用
- 生成训练损失曲线可视化

**输入数据格式：**
CSV文件包含以下特征列：

- `Wave Number`: 波次编号（1-8）
- `goblin`: 哥布林数量
- `goblin_priest`: 哥布林祭司数量
- `skeleton`: 骷髅数量
- `slime`: 史莱姆数量
- `slime_king`: 史莱姆王数量
- `Coin Reward`: 金币奖励
- `Wave Interval`: 波次间隔

**条件变量：**

- 难度标签：人工标注的难度值（0.0-1.0范围）
- 归一化波次编号：将波次1-8归一化到[0,1]区间

**输出：**

- `cvae_model.pth`: 模型权重文件
- `scaler.pkl`: 数据标准化器
- `features.pkl`: 特征列表
- `training_loss.png`: 训练损失曲线

### 3. 推理脚本 (`test.py`)

**功能：**

- 加载训练好的模型和标准化器
- 根据指定难度生成新关卡
- 自动应用难度曲线控制
- 保存生成的CSV文件和可视化图表
- 验证生成的关卡是否符合难度递增原则

**生成策略：**

- 根据基础难度选择不同的难度曲线类型：
  - 简单关卡（<0.3）：平缓线性增长
  - 中等关卡（0.3-0.6）：中等线性增长
  - 困难关卡（>0.6）：指数增长
- 应用游戏设计规则：
  - 史莱姆王和哥布林祭司不在第1波出现
  - 金币奖励为10的倍数，最低50
  - 波次间隔根据难度自动调整

**输出：**

- `Generated_Level_Diff_X.XX.csv`: 生成的关卡数据
- `level_diff_X.XX_monsters.png`: 怪物分布可视化
- `level_diff_X.XX_coins.png`: 金币奖励可视化

## 模型架构

### CVAE结构

**编码器（Encoder）：**

- 输入：关卡特征（6维）+ 条件变量（2维：难度+波次）
- 隐藏层1：128维，ReLU激活
- 隐藏层2：64维，ReLU激活
- 输出：潜在空间参数（μ, log(σ²)），12维

**解码器（Decoder）：**

- 输入：潜在向量（12维）+ 条件变量（2维）
- 隐藏层1：64维，ReLU激活
- 隐藏层2：128维，ReLU激活
- 输出：重建的关卡特征（6维），Sigmoid激活

### 损失函数设计

**总损失函数：** L = L_rec + β × L_kl

- **重建损失 (L_rec)**：均方误差(MSE)损失，衡量重建数据与原始数据的差距

  ```
  L_rec = MSE(x, x̂) = (1/n) × Σ(x_i - x̂_i)²
  ```

- **KL散度正则化 (L_kl)**：约束潜在空间分布接近标准正态分布

  ```
  L_kl = -0.5 × Σ[1 + log(σ²) - μ² - σ²]
  ```

- **β系数 (β=0.005)**：平衡重建质量与潜在空间正则化，小值确保模型优先关注数据重建

### 训练策略

- **训练轮次**：400 epochs，确保充分收敛
- **优化器**：Adam，学习率=0.001
- **批量大小**：8（匹配每关波次数量）
- **数据标准化**：使用MinMaxScaler将特征归一化到[0,1]区间

## 完整工作流程

### 阶段1：准备训练数据

```bash
# 将游戏JSON格式转换为训练CSV格式
python convert_utils.py json2csv Waves_1.json -o Level_1_Summary.csv -d 1
python convert_utils.py json2csv Waves_2.json -o Level_2_Summary.csv -d 2
python convert_utils.py json2csv Waves_3.json -o Level_3_Summary.csv -d 3
```

### 阶段2：训练模型

```bash
# 运行训练脚本
python train.py
```

训练过程会：

1. 加载三个关卡的CSV数据
2. 应用难度矩阵创建条件变量
3. 训练CVAE模型400个epoch
4. 保存模型、scaler和features

### 阶段3：生成新关卡

```bash
# 运行推理脚本
python test.py
```

默认生成三个难度级别（0.2, 0.5, 0.8）的关卡，也可以修改`test.py`中的`target_difficulties`列表自定义。

### 阶段4：转换为游戏格式

```bash
# 将生成的CSV转换为游戏JSON格式
python convert_utils.py csv2json Generated_Level_Diff_0.15.csv
```

生成的JSON文件可直接用于游戏，格式符合`Waves.json`规范。

## 
