import json
import random
import os
import csv
from collections import defaultdict
import re

# 根据 Configs.json 中的敌人强度进行分级
ENEMY_TIERS = {
    1: ["slime"],
    2: ["skeleton", "slime_king"],
    3: ["goblin"],
    4: ["goblin_priest"]
}

# 假设地图中有 5 个可用的出生点
AVAILABLE_SPAWN_POINTS = [1, 2, 3, 4, 5]


# 为每个难度级别定义生成参数
DIFFICULTY_CONFIG = {
    # 难度 1: 极简
    1: {
        "num_waves": 8,
        "allowed_tiers": [1],
        "enemies_per_wave": (3, 6),
        "coin_reward_per_wave": 200,
        "wave_interval": 6,
        "event_interval_range": (1.5, 3.0),
        "spawn_points_to_use": 1,
    },
    # 难度 2: 简单
    2: {
        "num_waves": 8,
        "allowed_tiers": [1],
        "enemies_per_wave": (6, 9),
        "coin_reward_per_wave": 180,
        "wave_interval": 5,
        "event_interval_range": (1.2, 2.5),
        "spawn_points_to_use": 1,
    },
    # 难度 3: 中等
    3: {
        "num_waves": 8,
        "allowed_tiers": [1, 2],
        "enemies_per_wave": (7, 12),
        "coin_reward_per_wave": 180,
        "wave_interval": 4,
        "event_interval_range": (0.8, 2.0),
        "spawn_points_to_use": 2,
    },
    # 难度 4: 困难
    4: {
        "num_waves": 8,
        "allowed_tiers": [1, 2],
        "enemies_per_wave": (8, 12),
        "coin_reward_per_wave": 180,
        "wave_interval": 4,
        "event_interval_range": (0.6, 1.8),
        "spawn_points_to_use": 2,
    },
    # 难度 5: 专家 (基于原来的难度2增强)
    5: {
        "num_waves": 8,
        "allowed_tiers": [1, 2, 3],
        "enemies_per_wave": (10, 15),
        "coin_reward_per_wave": 170,
        "wave_interval": 3,
        "event_interval_range": (0.4, 1.4),
        "spawn_points_to_use": 3,
    }
}

# 获取所有可能的敌人类型
ALL_ENEMY_TYPES = sorted(set(enemy for enemies in ENEMY_TIERS.values() for enemy in enemies))
def generate_waves(difficulty: int):
    """
    根据指定的难度生成敌人波次数据，随着波次增加难度上升
    """
    if difficulty not in DIFFICULTY_CONFIG:
        raise ValueError(f"无效的难度: {difficulty}. 请输入 1-5 之间的整数。")

    config = DIFFICULTY_CONFIG[difficulty]
    all_waves = []

    # 获取该难度下所有可用的出生点
    spawn_points = random.sample(AVAILABLE_SPAWN_POINTS, k=config["spawn_points_to_use"])

    for i in range(config["num_waves"]):
        # 计算当前波次的难度系数 (0.0 到 1.0)
        wave_progress = i / (config["num_waves"] - 1) if config["num_waves"] > 1 else 0.0
        
        # 动态调整允许的敌人等级 - 随着波次增加解锁更高级敌人
        current_tiers = []
        max_tier = max(config["allowed_tiers"])
        for tier in sorted(config["allowed_tiers"]):
            # 前1/3波次：只允许最低等级
            # 中间1/3波次：允许中等等级
            # 后1/3波次：允许最高等级
            if tier <= max_tier * (0.3 + wave_progress * 0.7):
                current_tiers.append(tier)
        
        possible_enemies = []
        for tier in current_tiers:
            possible_enemies.extend(ENEMY_TIERS[tier])
        
        # 动态调整敌人数量 - 随着波次增加敌人变多
        min_enemies, max_enemies = config["enemies_per_wave"]
        wave_enemies_range = (
            min_enemies + int((max_enemies - min_enemies) * 0.3 * wave_progress),
            max_enemies + int((max_enemies - min_enemies) * 0.7 * wave_progress)
        )
        num_enemies = random.randint(*wave_enemies_range)
        
        # 动态调整事件间隔 - 随着波次增加敌人出现更快
        min_interval, max_interval = config["event_interval_range"]
        wave_interval_range = (
            max(min_interval * (1 - wave_progress * 0.7), 0.0),
            max(max_interval * (1 - wave_progress * 0.5), 0.0)
        )
        
        spawn_list = []
        for _ in range(num_enemies):
            spawn_event = {
                "event_interval": round(random.uniform(*wave_interval_range), 2),
                "spawn_point": random.choice(spawn_points),
                "enemy_type": random.choice(possible_enemies)
            }
            spawn_list.append(spawn_event)
        
        # 随着波次增加，"小兵海"效果几率增加
        min_difficulty_for_swarm = 3
        swarm_chance = 0.1 + (0.3 * wave_progress) if difficulty >= min_difficulty_for_swarm else 0.0
        
        if random.random() < swarm_chance and spawn_list:
            # 小兵海规模随波次增加
            swarm_size = min(3 + int(3 * wave_progress), len(spawn_list))
            for _ in range(swarm_size):
                random.choice(spawn_list)["event_interval"] = 0

        wave = {
            "coin_rewards": config["coin_reward_per_wave"],
            "wave_interval": config["wave_interval"] if i < config["num_waves"] - 1 else 0,
            "spawn_list": spawn_list
        }
        all_waves.append(wave)

    return all_waves

def generate_csv_summary(waves_data, difficulty):
    """
    生成CSV格式的关卡数据摘要
    """
    # 准备CSV数据
    csv_data = []
    headers = ["Wave Number"] + ALL_ENEMY_TYPES + ["Coin Reward", "Wave Interval"]
    
    for wave_idx, wave in enumerate(waves_data, 1):
        # 统计每种敌人的数量
        enemy_counts = defaultdict(int)
        for spawn in wave["spawn_list"]:
            enemy_type = spawn["enemy_type"]
            enemy_counts[enemy_type] += 1
        
        # 创建CSV行数据
        row = {
            "Wave Number": wave_idx,
            "Coin Reward": wave["coin_rewards"],
            "Wave Interval": wave["wave_interval"]
        }
        
        # 添加每种敌人的数量
        for enemy_type in ALL_ENEMY_TYPES:
            row[enemy_type] = enemy_counts.get(enemy_type, 0)
        
        csv_data.append(row)
    
    # 写入CSV文件
    csv_filename = f"Level_{difficulty}_Summary.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_data)
    
    return csv_filename

def generate_waves_from_csv(csv_filename):
    """
    根据CSV文件生成敌人波次数据，直接使用CSV中的所有参数
    """
    waves_data = []
    
    # 读取CSV文件
    try:
        with open(csv_filename, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV文件未找到: {csv_filename}")
    except Exception as e:
        raise Exception(f"读取CSV文件时出错: {e}")
    
    # 智能确定需要使用的出生点数量（基于敌人总数）
    total_enemies = 0
    for row in rows:
        for enemy_type in ALL_ENEMY_TYPES:
            if enemy_type in row and row[enemy_type]:
                try:
                    count = float(row[enemy_type])  # 处理CSV中可能的浮点数格式
                    total_enemies += int(round(count))
                except ValueError:
                    continue
    
    # 根据总敌人数量确定使用的出生点数量
    if total_enemies <= 40:
        spawn_points_to_use = 1
    elif total_enemies <= 80:
        spawn_points_to_use = 2
    elif total_enemies <= 120:
        spawn_points_to_use = 3
    elif total_enemies <= 160:
        spawn_points_to_use = 4
    else:
        spawn_points_to_use = 5
    
    spawn_points = random.sample(AVAILABLE_SPAWN_POINTS, k=min(spawn_points_to_use, len(AVAILABLE_SPAWN_POINTS)))
    
    # 智能确定事件间隔范围
    avg_enemies_per_wave = total_enemies / max(len(rows), 1)
    if avg_enemies_per_wave <= 8:
        event_interval_range = (1.0, 2.0)  # 低密度
    elif avg_enemies_per_wave <= 15:
        event_interval_range = (0.5, 1.5)  # 中等密度
    elif avg_enemies_per_wave <= 25:
        event_interval_range = (0.2, 1.0)  # 高密度
    else:
        event_interval_range = (0.0, 0.5)   # 极高密度
    
    # 处理每一波
    for row in rows:
        try:
            wave_num = int(float(row["Wave Number"]))  # 处理可能的浮点数格式
            coin_reward = int(float(row["Coin Reward"]))
            wave_interval = float(row["Wave Interval"])
        except (KeyError, ValueError) as e:
            raise ValueError(f"CSV格式错误: 缺少必要列或数据类型错误 - {e}")
        
        # 收集本波的敌人
        spawn_list = []
        for enemy_type in ALL_ENEMY_TYPES:
            if enemy_type in row and row[enemy_type]:
                try:
                    # 处理CSV中的浮点数格式（例如 "2.0"）
                    count = int(round(float(row[enemy_type])))
                    for _ in range(count):
                        spawn_event = {
                            "event_interval": round(random.uniform(*event_interval_range), 2),
                            "spawn_point": random.choice(spawn_points),
                            "enemy_type": enemy_type
                        }
                        spawn_list.append(spawn_event)
                except ValueError:
                    continue
        
        # 添加"小兵海"效果（基于敌人密度智能判断）
        if avg_enemies_per_wave > 15 and random.random() < 0.3 and spawn_list:  # 30%的几率
            for _ in range(min(5, len(spawn_list) // 3)):  # 根据敌人数量调整
                random.choice(spawn_list)["event_interval"] = 0
        
        wave = {
            "coin_rewards": coin_reward,
            "wave_interval": wave_interval,
            "spawn_list": spawn_list
        }
        waves_data.append(wave)
    
    return waves_data

def sanitize_filename(filename):
    """清理文件名中的非法字符"""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def main():
    """
    主函数，提供两种模式:
    1. 生成新模式: 输入难度生成CSV和JSON
    2. CSV导入模式: 从CSV文件生成JSON，直接使用CSV中的所有参数
    """
    print("=== 敌人波次生成工具 ===")
    print("1. 生成新关卡 (根据难度等级)")
    print("2. 从CSV文件导入关卡设计")
    
    while True:
        mode = input("\n请选择操作模式 (1或2): ").strip()
        if mode in ["1", "2"]:
            break
        print("无效输入，请输入1或2")
    
    if mode == "1":
        # 生成新模式
        while True:
            try:
                difficulty_input = input("\n请输入难度等级 (1-5): ")
                difficulty = int(difficulty_input)
                if 1 <= difficulty <= 5:
                    break
                else:
                    print("无效的输入，请输入 1 到 5 之间的整数。")
            except ValueError:
                print("无效的输入，请输入一个整数。")

        try:
            waves_data = generate_waves(difficulty)
            
            # 生成CSV摘要文件
            csv_filename = generate_csv_summary(waves_data, difficulty)
            print(f"\n成功生成关卡摘要CSV: {csv_filename}")
            
            # 生成JSON文件
            json_filename = f"Waves_{difficulty}.json"
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(waves_data, f, indent=4, ensure_ascii=False)

            print(f"成功生成波次JSON: {json_filename}")
            print("\n文件已保存到当前目录。")
            print("您可以将JSON文件重命名为 'Waves.json' 或 'WavesTest.json' 以在游戏中使用。")

        except Exception as e:
            print(f"\n生成文件时发生错误: {e}")
    
    else:  # mode == "2"
        # CSV导入模式
        while True:
            csv_filename = input("\n请输入CSV文件名 (例如: Generated_Level_0.5.csv): ").strip()
            if os.path.exists(csv_filename):
                break
            print(f"文件 '{csv_filename}' 不存在，请重新输入。")
        
        try:
            # 从CSV生成波次数据，不再需要难度参数
            waves_data = generate_waves_from_csv(csv_filename)
            
            # 生成JSON文件
            base_name = os.path.splitext(os.path.basename(csv_filename))[0]
            json_filename = f"{sanitize_filename(base_name)}_Generated.json"
            
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(waves_data, f, indent=4, ensure_ascii=False)
            
            print(f"\n成功从CSV生成波次JSON: {json_filename}")
            print("文件已保存到当前目录。")
            print("您可以将此文件重命名为 'Waves.json' 或 'WavesTest.json' 以在游戏中使用。")
        
        except Exception as e:
            print(f"\n处理CSV文件时发生错误: {e}")

if __name__ == "__main__":
    main()