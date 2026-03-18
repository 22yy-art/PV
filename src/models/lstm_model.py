# # import torch
# # import torch.nn as nn


# # class PVLSTM(nn.Module):
# #     def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
# #         super().__init__()
# #         self.hidden_size = hidden_size
# #         self.num_layers = num_layers

# #         self.lstm = nn.LSTM(
# #             input_size=input_size,
# #             hidden_size=hidden_size,
# #             num_layers=num_layers,
# #             batch_first=True,
# #             dropout=dropout if num_layers > 1 else 0.0
# #         )
# #         self.dropout = nn.Dropout(dropout)
# #         self.fc = nn.Linear(hidden_size, output_size)

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
# #         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

# #         out, _ = self.lstm(x, (h0, c0))
# #         out = self.dropout(out[:, -1, :])
# #         return torch.relu(self.fc(out))
# import torch
# import torch.nn as nn

# class PVLSTM(nn.Module):
#     def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0.0
#         )
        
#         # 优化解码器：从单层变为双层 MLP，增强非线性推理能力
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_size // 2, output_size)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

#         out, _ = self.lstm(x, (h0, c0))
        
#         # 取最后一个时间步，去掉了粗暴的 dropout 以防关键特征丢失
#         last_out = out[:, -1, :] 
#         return torch.relu(self.fc(last_out))
import torch
import torch.nn as nn
import torch.nn.functional as F

class PVLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2, num_stations: int = 0, embedding_dim: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_stations = num_stations

        # --- 新增: 场站实体嵌入层 (Entity Embedding) ---
        # 如果传入了大于0的场站数量，则初始化Embedding层，并计算扩展后的 LSTM 输入维度
        if self.num_stations > 0:
            self.station_embedding = nn.Embedding(num_embeddings=num_stations, embedding_dim=embedding_dim)
            # 因为要把原始气象特征和8维Embedding特征拼接在一起，所以输入维度 = 原始特征数 + 8
            lstm_input_size = input_size + embedding_dim
        else:
            lstm_input_size = input_size

        # 核心序列提取器保持不变
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 解码器升级：融合 Context 向量与最后时刻隐状态的 MLP
        # 注意这里的输入维度是 hidden_size * 2，因为我们要拼接两个矩阵特征
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size), # 引入层归一化，加速收敛并防止过拟合
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- 新增: 拆分特征并进行 Embedding 拼接 ---
        if self.num_stations > 0:
            # 约定输入特征矩阵的最后一列为场站 ID 的编码
            site_idx = x[:, :, -1].long()             # (batch_size, seq_len)
            weather_features = x[:, :, :-1]           # 剩下的部分为气象数值特征 (batch_size, seq_len, input_size)
            
            # 经过 Embedding 层，得到场站的高维隐式表示
            site_emb = self.station_embedding(site_idx) # (batch_size, seq_len, embedding_dim)
            
            # 将气象特征与场站嵌入特征在最后一个维度拼接起来
            x_new = torch.cat([weather_features, site_emb], dim=-1)
        else:
            x_new = x

        h0 = torch.zeros(self.num_layers, x_new.size(0), self.hidden_size, device=x_new.device)
        c0 = torch.zeros(self.num_layers, x_new.size(0), self.hidden_size, device=x_new.device)

        # out 的形状: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x_new, (h0, c0))
        
        # --- 核心矩阵运算：点积注意力机制 (Dot-Product Attention) ---
        
        # 1. 提取序列最后一个时刻的隐状态作为 Query
        # query 形状: (batch_size, hidden_size, 1)
        query = out[:, -1, :].unsqueeze(2) 
        
        # 2. 计算 Query 与所有历史隐状态 (Keys) 的点积，得到相关性打分
        # (batch_size, seq_len, hidden_size) @ (batch_size, hidden_size, 1) 
        # -> (batch_size, seq_len, 1) -> squeeze 变成 (batch_size, seq_len)
        attn_scores = torch.bmm(out, query).squeeze(2)
        
        # 3. 对分数进行 Softmax 归一化，得到权重矩阵
        # attn_weights 形状: (batch_size, 1, seq_len)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1)
        
        # 4. 将权重矩阵乘回所有隐状态 (Values)，提取出上下文特征矩阵
        # (batch_size, 1, seq_len) @ (batch_size, seq_len, hidden_size) 
        # -> (batch_size, 1, hidden_size) -> squeeze 变成 (batch_size, hidden_size)
        context = torch.bmm(attn_weights, out).squeeze(1)
        
        # 5. 将 Attention 提取的全局上下文特征与局部最后状态进行拼接
        # combined 形状: (batch_size, hidden_size * 2)
        combined = torch.cat((context, out[:, -1, :]), dim=1)

        # 最终经过 MLP 输出预测结果
        return torch.relu(self.fc(combined))