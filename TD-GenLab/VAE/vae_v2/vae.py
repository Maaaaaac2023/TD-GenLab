import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
import time

# 配置常量
BASE_DIR = 'vae'
MODEL_PATH = os.path.join(BASE_DIR, 'cvae_model.pth')
TRAINING_LOSS_PATH = os.path.join(BASE_DIR, 'training_loss.png')

# 确保输出目录存在
os.makedirs(BASE_DIR, exist_ok=True)

# 2. 数据加载与预处理
def load_and_preprocess_data(difficulty_matrix):
    """
    加载和预处理关卡数据，使用标准化波次而非波次倒数
    """
    # 加载关卡数据
    level_files = [
        os.path.join(BASE_DIR, 'Level_1_Summary.csv'),
        os.path.join(BASE_DIR, 'Level_2_Summary.csv'), 
        os.path.join(BASE_DIR, 'Level_3_Summary.csv')
    ]
    
    # 提取特征 (排除Wave Number和Wave Interval列)
    features = ['goblin', 'goblin_priest', 'skeleton', 'slime', 'slime_king', 'Coin Reward']
    
    # 构建数据集
    all_data = []
    all_difficulties = []
    all_wave_numbers = []
    
    for i, level_file in enumerate(level_files):
        if not os.path.exists(level_file):
            print(f"⚠️  警告: 文件不存在 {level_file}")
            continue
            
        level = pd.read_csv(level_file)
        
        # 每个关卡取8波数据
        wave_data = level[features].values[:8]
        all_data.append(wave_data)
        
        # 使用新的难度矩阵
        difficulties = difficulty_matrix[i].reshape(-1, 1)
        all_difficulties.append(difficulties)
        
        # 使用标准化波次 (0-1)
        wave_numbers = np.arange(1, 9).reshape(-1, 1)
        normalized_wave_numbers = (wave_numbers - 1) / 7  # 归一化到[0,1]
        all_wave_numbers.append(normalized_wave_numbers)
    
    if not all_data:
        raise FileNotFoundError("未找到任何关卡数据文件，请检查vae文件夹中的CSV文件")
    
    # 合并数据
    X = np.vstack(all_data)  # 形状: (24, 6)
    y_difficulty = np.vstack(all_difficulties)  # 形状: (24, 1)
    y_wave_number = np.vstack(all_wave_numbers)  # 形状: (24, 1)
    
    # 标准化数据
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_difficulty, y_wave_number, scaler, features

# 3. CVAE 模型定义
class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim=12):
        super(CVAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 潜在空间参数
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        return self.decoder(inputs)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# 4. 难度曲线控制器
def apply_difficulty_curve(generated_data, base_difficulty, wave_numbers):
    """
    应用显式的难度曲线控制，确保生成的关卡符合游戏设计原则
    """
    num_waves = generated_data.shape[0]
    
    # 选择难度曲线类型（根据基础难度）
    if base_difficulty < 0.3:
        curve_type = "linear_gentle"   # 简单关卡：平缓增长
    elif base_difficulty < 0.6:
        curve_type = "linear_medium"   # 中等关卡：中等增长
    else:
        curve_type = "exponential"     # 困难关卡：指数增长
    
    # 计算难度系数
    difficulty_factors = np.zeros(num_waves)
    
    for i, wave_num in enumerate(range(1, num_waves + 1)):
        # 根据曲线类型计算难度系数
        normalized_wave = (wave_num - 1) / (num_waves - 1)  # 0-1范围
        
        if curve_type == "linear_gentle":
            # 简单关卡：平缓增长 (0.6 → 1.0)
            factor = 0.6 + 0.4 * normalized_wave
        elif curve_type == "linear_medium":
            # 中等关卡：中等增长 (0.5 → 1.2)
            factor = 0.5 + 0.7 * normalized_wave
        else:  # "exponential"
            # 困难关卡：指数增长，后期急剧上升
            factor = 0.4 + 1.0 * (normalized_wave ** 1.5)
        
        # 添加小的随机波动，使曲线更自然
        noise = np.random.uniform(0.95, 1.05)
        difficulty_factors[i] = factor * noise
    
    # 应用难度系数到怪物数量
    for wave_idx in range(num_waves):
        factor = difficulty_factors[wave_idx]
        
        # 哥布林、哥布林祭司、骷髅、史莱姆、史莱姆王
        for monster_idx in range(5):
            # 增加怪物数量，但保留一些变化
            generated_data[wave_idx, monster_idx] *= factor
            
            # 确保史莱姆王只在后期出现（游戏设计原则）
            if monster_idx == 4 and wave_idx < 3:  # 前3波
                generated_data[wave_idx, monster_idx] *= 0.3  # 减少史莱姆王数量
        
        # 金币奖励应随难度增加
        generated_data[wave_idx, 5] = max(50, generated_data[wave_idx, 5] * (0.7 + 0.5 * factor))
    
    return generated_data

# 5. 训练和保存模型函数
def train_and_save_cvae(model, dataloader, optimizer, epochs=300):
    model.train()
    losses = []
    
    print("🏋️ 开始训练CVAE模型...")
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, c_batch in dataloader:
            recon_x, mu, logvar = model(x_batch, c_batch)
            
            # 计算损失
            recon_loss = nn.MSELoss()(recon_x, x_batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + 0.005 * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, 耗时: {elapsed:.1f}s')
    
    # 保存训练好的模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ 模型已保存至 {MODEL_PATH}")
    
    return losses

# 6. 生成新关卡
def generate_level(model, scaler, difficulty, num_waves=8):
    """
    生成符合难度递增原则的新关卡
    """
    model.eval()
    
    # 1. 创建条件变量
    difficulties = torch.full((num_waves, 1), difficulty, dtype=torch.float32)
    wave_numbers = torch.linspace(0, 1, num_waves, dtype=torch.float32).reshape(-1, 1)
    
    conditions = torch.cat([difficulties, wave_numbers], dim=1)
    
    # 2. 生成基础数据
    with torch.no_grad():
        z = torch.randn(num_waves, model.fc_mu.out_features)
        generated = model.decode(z, conditions).numpy()
    
    # 3. 反标准化
    generated_original = scaler.inverse_transform(generated)
    
    # 4. 应用显式难度曲线
    generated_original = apply_difficulty_curve(generated_original, difficulty, wave_numbers)
    
    # 5. 后处理
    for i in range(5):  # 前5列是怪物数量
        generated_original[:, i] = np.round(np.maximum(generated_original[:, i], 0))
        
        # 特殊规则：第1波不能有史莱姆王或哥布林祭司
        if i in [1, 4]:  # 哥布林祭司(1)和史莱姆王(4)
            generated_original[0, i] = 0
    
    # 确保Coin Reward为10的倍数
    generated_original[:, 5] = np.round(generated_original[:, 5] / 10) * 10
    generated_original[:, 5] = np.maximum(generated_original[:, 5], 50)  # 最低50金币
    
    # 添加Wave Interval列
    if difficulty < 0.3:
        base_interval = 5.0  # 简单关卡
    elif difficulty < 0.6:
        base_interval = 4.0  # 中等关卡
    else:
        base_interval = 3.0  # 困难关卡
    
    wave_intervals = np.full((num_waves, 1), base_interval)
    wave_intervals[-1] = 0  # 最后一波间隔为0
    
    # 添加Wave Number列
    wave_numbers_np = np.arange(1, num_waves + 1).reshape(-1, 1)
    
    # 合并所有数据
    final_level_data = np.hstack([wave_numbers_np, generated_original, wave_intervals])
    
    return final_level_data

# 7. 保存和可视化生成的关卡
def save_and_visualize_level(level_data, features, difficulty, base_dir=BASE_DIR):
    filename = os.path.join(base_dir, f'Generated_Level_Diff_{difficulty:.2f}.csv')
    
    # 创建DataFrame
    all_columns = ['Wave Number'] + features + ['Wave Interval']
    df = pd.DataFrame(level_data, columns=all_columns)
    
    # 保存到CSV
    df.to_csv(filename, index=False)
    print(f'✅ 生成的关卡已保存到 {filename}')
    
    # 可视化1: 怪物分布
    monster_plot_path = os.path.join(base_dir, f'level_diff_{difficulty:.2f}_monsters.png')
    plt.figure(figsize=(12, 6))
    monster_names = ['goblin', 'skeleton', 'slime', 'slime_king']
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, monster in enumerate(monster_names):
        if monster in features:
            idx = all_columns.index(monster)
            plt.plot(level_data[:, 0], level_data[:, idx], 'o-', 
                     linewidth=2.5, markersize=8, color=colors[i], label=monster)
    
    plt.title(f'生成关卡 (基础难度: {difficulty:.2f}) - 怪物分布', fontsize=14)
    plt.xlabel('波次', fontsize=12)
    plt.ylabel('怪物数量', fontsize=12)
    plt.xticks(range(1, 9))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(monster_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 可视化2: 金币奖励
    coin_plot_path = os.path.join(base_dir, f'level_diff_{difficulty:.2f}_coins.png')
    plt.figure(figsize=(10, 5))
    coin_idx = all_columns.index('Coin Reward')
    plt.bar(level_data[:, 0], level_data[:, coin_idx], color='gold', edgecolor='darkgoldenrod', alpha=0.8)
    plt.title(f'生成关卡 (基础难度: {difficulty:.2f}) - 金币奖励', fontsize=14)
    plt.xlabel('波次', fontsize=12)
    plt.ylabel('金币数量', fontsize=12)
    plt.xticks(range(1, 9))
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(coin_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 已生成可视化图表: '{os.path.basename(monster_plot_path)}' 和 '{os.path.basename(coin_plot_path)}'")
    return df

# 8. 验证生成的关卡是否符合难度递增原则
def validate_difficulty_progression(level_data):
    """
    验证生成的关卡是否符合难度递增原则
    """
    print("\n🔍 难度递增验证分析:")
    print("-" * 60)
    
    # 计算每波的"难度分数"（怪物数量加权和）
    weights = [1.0, 1.5, 1.0, 0.8, 2.0]  # 不同怪物的难度权重
    difficulty_scores = []
    
    for wave_idx in range(8):
        score = 0
        for monster_idx, weight in enumerate(weights):
            count = level_data[wave_idx, monster_idx + 1]  # +1跳过Wave Number
            score += count * weight
        difficulty_scores.append(score)
    
    # 打印分析
    print("波次 | 难度分数 | 与前一波变化")
    print("-" * 40)
    
    for i, score in enumerate(difficulty_scores):
        if i == 0:
            change = "基准"
        else:
            change_percent = (score - difficulty_scores[i-1]) / difficulty_scores[i-1] * 100
            change = f"{change_percent:+.1f}%"
        
        print(f"{i+1:2d} | {score:8.1f} | {change}")
    
    # 验证难度递增
    increasing = all(difficulty_scores[i] >= difficulty_scores[i-1] * 0.9 for i in range(1, 8))
    final_vs_first = difficulty_scores[-1] / difficulty_scores[0]
    
    print("-" * 60)
    print(f"✅ 难度总体递增: {'是' if increasing else '否'}")
    print(f"📈 最终波 vs 第一波: {final_vs_first:.1f}倍难度")
    
    if increasing and final_vs_first > 1.5:
        print("🎉 生成的关卡符合游戏设计原则！")
    else:
        print("⚠️  建议重新生成或调整难度参数")

# 主程序
def main():
    # 1. 创建符合游戏设计原则的难度数据
    print("🎯 创建符合游戏设计原则的难度曲线数据...")
    difficulty_matrix = np.array([
        [0.10, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30],  # Level 1
        [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],  # Level 2
        [0.50, 0.58, 0.66, 0.74, 0.82, 0.86, 0.90, 0.95]   # Level 3
    ])
    
    # 2. 加载和预处理数据
    print("\n📊 加载和预处理关卡数据...")
    try:
        X_scaled, y_difficulty, y_wave_number, scaler, features = load_and_preprocess_data(difficulty_matrix)
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        return
    
    # 3. 准备PyTorch数据
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    conditions = np.hstack([y_difficulty, y_wave_number])
    c_tensor = torch.tensor(conditions, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, c_tensor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 4. 初始化CVAE模型
    input_dim = X_scaled.shape[1]  # 6个特征
    condition_dim = conditions.shape[1]  # 2个条件变量
    
    print("\n🧠 初始化CVAE模型...")
    model = CVAE(input_dim, condition_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. 检查是否存在预训练模型
    if os.path.exists(MODEL_PATH):
        print(f"📥 检测到已训练模型，加载中: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH))
        print("✅ 模型加载成功！跳过训练阶段。")
    else:
        print("\n🆕 未找到预训练模型，开始训练新模型...")
        # 训练模型
        losses = train_and_save_cvae(model, dataloader, optimizer, epochs=400)
        
        # 可视化训练过程
        plt.figure(figsize=(10, 5))
        plt.plot(losses, linewidth=2, color='blue')
        plt.title('CVAE 训练损失曲线', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(TRAINING_LOSS_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 训练损失曲线已保存为 '{TRAINING_LOSS_PATH}'")
    
    # 6. 生成不同难度的关卡
    target_difficulties = [0.2, 0.5, 0.8]  # 低、中、高难度
    
    for diff in target_difficulties:
        print(f"\n🎲 生成基础难度为 {diff:.2f} 的新关卡 (符合难度递增原则)...")
        generated_level = generate_level(model, scaler, diff)
        
        # 保存和可视化
        df = save_and_visualize_level(generated_level, features, diff)
        
        # 验证难度递增
        validate_difficulty_progression(generated_level)
        
        # 打印详细数据
        print("\n📋 生成的关卡详细数据:")
        print(df.to_string(index=False))
    
    print("\n🎉 完成! 所有关卡已成功生成并保存。")
    print(f"   生成的文件均保存在 '{BASE_DIR}' 文件夹中:")
    print(f"   - 模型文件: {os.path.basename(MODEL_PATH)}")
    print(f"   - 训练损失图: {os.path.basename(TRAINING_LOSS_PATH)}")
    print("   - 生成的关卡文件 (3个难度级别)")
    print("   - 可视化图表 (怪物分布和金币奖励)")

if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行主程序
    main()