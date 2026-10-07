import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, ndcg_score
from tqdm import tqdm
import os

# ==========================================
# 0. 全域參數
# ==========================================
BETA_LIST = [0,0.1, 0.2, 0.3, 0.4, 0.5]  # 實驗變數
THRESHOLD = 3.5     # 定義 "喜歡" 的門檻
EPOCHS = 20         # 每個 Beta 跑幾輪
BATCH_SIZE = 256
EMBED_DIM = 32
LR = 0.001
OUTPUT_FILE = "hybrid_experiment_results.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. 資料載入與預處理
# ==========================================
print("--- Loading & Preprocessing Data ---")
try:
    raw = pd.read_csv("useractions.csv")
    movies = pd.read_csv("movies.csv")
except FileNotFoundError:
    print("Warning: CSV not found. Generating Mock Data for testing...")
    # 產生假資料
    raw = pd.DataFrame({
        "UserID": np.random.randint(1, 101, 3000),
        "MovieID": np.random.randint(1, 101, 3000),
        "Action_Type": np.random.choice(['click', 'add_to_favorite', 'view'], 3000),
        "Rating": np.random.randint(1, 6, 3000)
    })
    movies = pd.DataFrame({
        "MovieID": range(1, 101),
        "Genres": ["Action|Sci-Fi" if i%2==0 else "Comedy|Drama" for i in range(1, 101)]
    })

# 1.1 行為轉數值
action_weights = {"click": 2.0, "add_to_favorite": 4.0, "view": 1.0}
rating_map = {1: 0.5, 2: 1.0, 3: 3.0, 4: 4.0, 5: 5.0}

def map_action(row):
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
combined = raw.groupby(["UserID", "MovieID"], as_index=False)["Preference"].sum()

# 1.2 ID Mapping
user_ids_unique = combined["UserID"].unique()
movie_ids_unique = movies["MovieID"].unique()

user2idx = {uid: i for i, uid in enumerate(user_ids_unique)}
movie2idx = {mid: i for i, mid in enumerate(movie_ids_unique)}

combined = combined[combined["MovieID"].isin(movie2idx)].copy()
combined["user_idx"] = combined["UserID"].map(user2idx)
combined["movie_idx"] = combined["MovieID"].map(movie2idx)

num_users = len(user2idx)
num_items = len(movie2idx)

# 1.3 Genre 向量化計算
print("--- Calculating User-Genre Profile ---")
ui_matrix = combined.pivot(index="UserID", columns="MovieID", values="Preference").fillna(0)
mg_matrix = movies.set_index("MovieID")["Genres"].str.get_dummies(sep="|")

common_movies = ui_matrix.columns.intersection(mg_matrix.index)
ui_matrix = ui_matrix[common_movies]
mg_matrix = mg_matrix.loc[common_movies]

user_genre_scores = ui_matrix.dot(mg_matrix)
user_genre_profile = user_genre_scores.div(user_genre_scores.sum(axis=1) + 1e-9, axis=0)

# ==========================================
# 2. 模型與 Dataset
# ==========================================
class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32):
        super(NeuralCF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, u, i):
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=-1)
        return self.fc(x).squeeze()

class RatingDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["Preference"].values, dtype=torch.float32)
        self.raw_uids = df["UserID"].values
        self.raw_mids = df["MovieID"].values
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx], self.raw_uids[idx], self.raw_mids[idx]

# 資料分割
train_data, val_data = train_test_split(combined, test_size=0.2, random_state=42)
train_loader = DataLoader(RatingDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(RatingDataset(val_data), batch_size=512, shuffle=False)

# ==========================================
# 3. 評估函數
# ==========================================
def evaluate_metrics(model, val_loader, alpha, beta):
    model.eval()
    results = {"user_idx": [], "true": [], "pred": []}
    
    with torch.no_grad():
        for u_idx, m_idx, r, raw_u, raw_m in val_loader:
            u_idx_dev, m_idx_dev = u_idx.to(device), m_idx.to(device)
            
            # A. Neural CF
            ncf_score = model(u_idx_dev, m_idx_dev).cpu().numpy()
            
            # B. Genre (Vector lookup on CPU)
            batch_genre_scores = []
            for k in range(len(raw_u)):
                uid = raw_u[k].item()
                mid = raw_m[k].item()
                g_score = 0.0
                if uid in user_genre_profile.index:
                    u_vec = user_genre_profile.loc[uid]
                    if mid in movies["MovieID"].values:
                        g_str = movies[movies["MovieID"] == mid]["Genres"].values[0]
                        g_list = g_str.split("|") if isinstance(g_str, str) else []
                        g_score = sum(u_vec.get(g, 0.0) for g in g_list)
                batch_genre_scores.append(g_score)
            
            genre_score = np.array(batch_genre_scores)
            
            # C. Hybrid Formula (Scale genre by 5)
            final_score = (alpha * ncf_score) + (beta * genre_score * 5.0)
            
            results["user_idx"].extend(u_idx.numpy())
            results["true"].extend(r.numpy())
            results["pred"].extend(final_score)

    # Convert to DF
    df_res = pd.DataFrame(results)
    y_true = df_res["true"].values
    y_pred = df_res["pred"].values
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    y_true_bin = (y_true >= THRESHOLD).astype(int)
    y_pred_bin = (y_pred >= THRESHOLD).astype(int)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    
    # nDCG (Validation Set)
    ndcg_list = []
    for _, group in df_res.groupby("user_idx"):
        if len(group) < 2: continue
        ndcg_list.append(ndcg_score([group["true"].values], [group["pred"].values]))
    avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
    
    return rmse, mae, precision, recall, avg_ndcg

# ==========================================
# 4. 主實驗迴圈
# ==========================================
all_experiment_data = []

print(f"\n========================================")
print(f"Starting Experiment: Betas {BETA_LIST}")
print(f"Results will be saved to: {OUTPUT_FILE}")
print(f"========================================\n")

# 如果檔案存在，先刪除舊的 (或可改為 append 模式)
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
    
# 寫入 CSV Header
header_df = pd.DataFrame(columns=["Beta", "Alpha", "Epoch", "RMSE", "MAE", "Precision", "Recall", "nDCG"])
header_df.to_csv(OUTPUT_FILE, index=False)

for beta in BETA_LIST:
    alpha = round(1.0 - beta, 2)
    print(f"\n>>> Running Experiment: Beta={beta} (Genre), Alpha={alpha} (NCF)")
    
    # 重置模型 (確保每次都是從頭訓練)
    model = NeuralCF(num_users, num_items, EMBED_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # 訓練迴圈
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # 進度條
        loop = tqdm(train_loader, desc=f"[Beta={beta}] Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for u, i, r, _, _ in loop:
            u, i, r = u.to(device), i.to(device), r.to(device)
            optimizer.zero_grad()
            loss = criterion(model(u, i), r)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        # 評估
        rmse, mae, prec, rec, ndcg = evaluate_metrics(model, val_loader, alpha, beta)
        
        # 記錄數據
        result_row = {
            "Beta": beta,
            "Alpha": alpha,
            "Epoch": epoch + 1,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "nDCG": round(ndcg, 4)
        }
        all_experiment_data.append(result_row)
        
        # 即時顯示最後一輪的結果 (或每5輪)
        if (epoch+1) % 5 == 0:
             print(f"   Epoch {epoch+1}: RMSE={rmse:.4f}, nDCG={ndcg:.4f}")

    # === 每個 Beta 跑完後，將數據 Append 到 CSV ===
    # 這裡我們把這一個 Beta 的所有 Epoch 數據都存進去
    current_beta_df = pd.DataFrame([
        row for row in all_experiment_data 
        if row["Beta"] == beta
    ])
    
    # 存檔 (Append mode, 不寫入 header)
    current_beta_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    print(f">>> Beta={beta} Finished. Data saved.")
    
    # 清空暫存 list 以避免重複寫入 (因為我們是用 append mode 寫檔)
    all_experiment_data = [] 

print(f"\n========================================")
print(f"All Experiments Completed!")
print(f"Full results saved to {OUTPUT_FILE}")
print(f"========================================")

# 顯示最終摘要 (每個 Beta 的最佳 RMSE)
final_df = pd.read_csv(OUTPUT_FILE)
summary = final_df.loc[final_df.groupby("Beta")["RMSE"].idxmin()] # 找出每個 Beta 中 RMSE 最低的那一行
print("\n--- Summary: Best Result per Beta ---")
print(summary[["Beta", "Epoch", "RMSE", "nDCG", "Precision", "Recall"]])