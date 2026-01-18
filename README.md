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
