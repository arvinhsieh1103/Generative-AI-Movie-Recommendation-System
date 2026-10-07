import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 0. 環境設定
# ==========================================
# 檢查是否有 GPU，若無則使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. 載入資料 (真實數據)
# ==========================================
try:
    print("--- Loading Data ---")
    raw = pd.read_csv("useractions.csv")
    movies = pd.read_csv("movies.csv")
except FileNotFoundError:
    print("錯誤：找不到 'useractions.csv' 或 'movies.csv'，請確保檔案在同一目錄下。")
    exit()

# ==========================================
# 2. 資料前處理
# ==========================================
print("--- Preprocessing Data ---")

# 2.1 定義權重 (將行為轉為數值)
action_weights = {"click": 2.0, "add_to_favorite": 4.0, "view": 1.0}
rating_map = {1: 0.5, 2: 1.0, 3: 3.0, 4: 4.0, 5: 5.0}

def map_action(row):
    # 優先權：收藏 > 評分 > 點擊
    if row.get("Action_Type") == "add_to_favorite":
        return action_weights["add_to_favorite"]
    
    if pd.notna(row.get("Rating")):
        try:
            r = int(row["Rating"])
            return rating_map.get(r, float(r))
        except:
            pass
            
    return action_weights.get(row.get("Action_Type"), 0.0)

raw["Preference"] = raw.apply(map_action, axis=1)

# 2.2 聚合 User-Item 分數 (同一人對同一片多次行為加總)
combined = raw.groupby(["UserID", "MovieID"], as_index=False)["Preference"].sum()

# 2.3 ID Mapping (建立 ID 到 索引 0~N 的對照表)
user_ids_unique = combined["UserID"].unique()
movie_ids_unique = movies["MovieID"].unique()

user2idx = {uid: i for i, uid in enumerate(user_ids_unique)}
idx2user = {i: uid for i, uid in enumerate(user_ids_unique)}
movie2idx = {mid: i for i, mid in enumerate(movie_ids_unique)}
idx2movie = {i: mid for i, mid in enumerate(movie_ids_unique)}

# 過濾掉不在 movie list 中的資料，並轉換為 Index
combined = combined[combined["MovieID"].isin(movie2idx)].copy()
combined["user_idx"] = combined["UserID"].map(user2idx)
combined["movie_idx"] = combined["MovieID"].map(movie2idx)

num_users = len(user2idx)
num_items = len(movie2idx)
print(f"Total Users: {num_users}, Total Items: {num_items}")

# ==========================================
# 3. 定義 Dataset 與 Neural CF 模型
# ==========================================
class RatingDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["Preference"].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32, layers=[64, 32]):
        super(NeuralCF, self).__init__()
        # Embedding 層
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        
        # MLP 層
        self.fc_layers = nn.ModuleList()
        input_dim = embed_dim * 2 
        
        for layer_dim in layers:
            self.fc_layers.append(nn.Linear(input_dim, layer_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(0.2)) # Dropout 防止過擬合
            input_dim = layer_dim
            
        # 輸出層 (預測分數)
        self.output_layer = nn.Linear(layers[-1], 1)
        
    def forward(self, user_idx, item_idx):
        u_emb = self.user_embedding(user_idx)
        i_emb = self.item_embedding(item_idx)
        x = torch.cat([u_emb, i_emb], dim=-1) # 串接
        for layer in self.fc_layers:
            x = layer(x)
        pred = self.output_layer(x)
        return pred.squeeze()

# ==========================================
# 4. 訓練模型 (含 Validation 與 記錄 Loss)
# ==========================================
print("--- Start Training ---")

# 超參數設定
EMBED_DIM = 32
HIDDEN_LAYERS = [64, 32]
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5  # L2 正則化，防止過擬合
BATCH_SIZE = 256
EPOCHS = 30  # 真實數據建議 30~50 輪

# 資料分割 (Train 80%, Validation 20%)
train_data, val_data = train_test_split(combined, test_size=0.2, random_state=42)
train_loader = DataLoader(RatingDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(RatingDataset(val_data), batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型與優化器
model = NeuralCF(num_users, num_items, EMBED_DIM, HIDDEN_LAYERS).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 記錄 Loss 用於繪圖
train_losses = []
val_losses = []

best_loss = float('inf')

for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    total_train_loss = 0
    for u, i, r in train_loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        
        optimizer.zero_grad()
        prediction = model(u, i)
        loss = criterion(prediction, r)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # --- Validation Phase ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for u, i, r in val_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            prediction = model(u, i)
            loss = criterion(prediction, r)
            total_val_loss += loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # 印出進度
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 記錄最佳 Loss
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss

print(f"Training Complete. Best Validation Loss: {best_loss:.4f}")

# ==========================================
# 5. 繪製 Loss 曲線圖
# ==========================================
print("--- Plotting Loss Curve ---")
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2, linestyle='--')
plt.title('Neural CF Training Progress')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show() # 顯示圖表

# ==========================================
# 6. 推薦函數
# ==========================================
def recommend_ncf(user_id, n=5):
    # 1. 新用戶處理
    if user_id not in user2idx:
        print(f"User {user_id} is new (Cold Start). Returning popular movies.")
        popular = combined.groupby("MovieID")["Preference"].sum().nlargest(n).index.tolist()
        return [(mid, 0.0) for mid in popular]

    u_idx_int = user2idx[user_id]
    
    # 2. 過濾已看過的
    seen_movies = set(combined[combined["UserID"] == user_id]["MovieID"])
    all_movie_ids = list(movie2idx.keys())
    candidate_movies = [m for m in all_movie_ids if m not in seen_movies]
    
    if not candidate_movies:
        return []

    # 3. 準備預測資料
    candidate_indices = [movie2idx[m] for m in candidate_movies]
    
    u_tensor = torch.LongTensor([u_idx_int] * len(candidate_indices)).to(device)
    i_tensor = torch.LongTensor(candidate_indices).to(device)
    
    # 4. 模型預測
    model.eval()
    with torch.no_grad():
        predictions = model(u_tensor, i_tensor).cpu().numpy()
    
    # 5. 排序並回傳
    recommendations = []
    for idx, score in enumerate(predictions):
        recommendations.append((candidate_movies[idx], score))
        
    top_n = sorted(recommendations, key=lambda x: x[1], reverse=True)[:n]
    return top_n

# ==========================================
# 7. 測試推薦結果
# ==========================================
if len(user_ids_unique) > 0:
    test_user = user_ids_unique[0] # 抓取第一個 User 來測試
    print(f"\n--- Recommendations for UserID: {test_user} ---")
    
    recs = recommend_ncf(test_user, n=5)
    
    for mid, score in recs:
        # 抓取電影標題
        title_row = movies[movies["MovieID"] == mid]
        title = title_row["Title"].values[0] if not title_row.empty else "Unknown"
        print(f"Movie: {title} (ID: {mid}) | Predicted Score: {score:.4f}")
else:
    print("No users found in dataset.")