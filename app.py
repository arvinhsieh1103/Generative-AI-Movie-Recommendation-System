from flask import Flask, request, redirect, render_template, url_for, session, jsonify
from pymongo import MongoClient, DESCENDING
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from sklearn.decomposition import TruncatedSVD
import logging
import re
from agent import generate_movie_genres
from dotenv import load_dotenv
from werkzeug.security import (generate_password_hash, check_password_hash, )

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ["FLASK_SECRET_KEY"]

# 連接 MongoDB Atlas
client = MongoClient(os.environ["MONGODB_URI"])
db = client["Data"]

collection = db["Users"]
movies_collection = db["Movies"]
actions_collection = db["UserActions"]

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get("username").strip()
        email = request.form.get("email").strip()
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        gender = request.form.get("gender")
        age = request.form.get("age")
        occupation = request.form.get("occupation")

        # 檢查是否已存在帳號或信箱
        existing_user = collection.find_one({
            "$or": [{"username": username}, {"email": email}]
        })
        if existing_user:
            return render_template("register.html", error="Username or email already exists.")

        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match.")

        # 取得最大 UserID
        last_user = collection.find_one(sort=[("UserID", DESCENDING)])
        next_user_id = last_user["UserID"] + 1 if last_user else 1

        # 儲存使用者（加密密碼）
        user_data = {
            "UserID": next_user_id,
            "username": username,
            "email": email,
            "password": generate_password_hash(password),
            "gender": gender,
            "age": age,
            "occupation": occupation
        }

        collection.insert_one(user_data)
        return redirect("/login")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        user = collection.find_one({"username": username})

        if user and check_password_hash(
            user.get("password", ""), password
        ):
            session["username"] = username
            session["user_id"] = user["UserID"]

            # 先嘗試載入快取模型
            global predicted_ratings, user_ids, movie_ids, user_means, user_genre_pref, combined
            predicted_ratings, user_ids, movie_ids, user_means, user_genre_pref, combined = load_recommender_from_file()

            # 若沒有快取或過期 -> 重訓並存檔
            if predicted_ratings is None:
                print("[Recommender] 快取無效，重新建模中...")
                predicted_ratings, user_ids, movie_ids, user_means, user_genre_pref = build_recommender_with_genre()
                save_recommender_to_file(predicted_ratings, user_ids, movie_ids, user_means, user_genre_pref, combined)

            # 建立推薦
            recommendations = recommend_movies(user["UserID"], n=10)
            session["recommendations"] = recommendations
            session["recommendation_ids"] = [rec["MovieID"] for rec in recommendations]

            return redirect(url_for("search_page"))

        return render_template("login.html", error="帳號或密碼錯誤")
    return render_template("login.html")

def get_top_movies_fallback(n=10):
    """當推薦模型無法使用時，用平均評分最高的電影作為推薦"""
    pipeline = [
        {"$match": {"Action_Type": "rate", "Rating": {"$ne": None}}},
        {"$group": {"_id": "$MovieID", "avg_rating": {"$avg": "$Rating"}, "count": {"$sum": 1}}},
        {"$sort": {"avg_rating": -1, "count": -1}},
        {"$limit": n}
    ]
    top = list(actions_collection.aggregate(pipeline))

    if not top:  # 若資料庫完全沒評分紀錄，就直接取前 n 部電影
        docs = list(movies_collection.find().limit(n))
    else:
        movie_ids_top = [t["_id"] for t in top]
        docs = [movies_collection.find_one({"MovieID": mid}, {"_id": 0}) for mid in movie_ids_top]
        docs = [d for d in docs if d]

    return [
        {
            "MovieID": d.get("MovieID"),
            "Title": d.get("Title", "Unknown"),
            "image": d.get("image", "")
        }
        for d in docs
    ]

@app.route('/search')
def search_page():
    if 'username' not in session:
        return redirect("/login")

    rec_ids = session.get("recommendation_ids", [])
    recommendations = list(movies_collection.find(
        {"MovieID": {"$in": rec_ids}},
        {"_id": 0, "MovieID": 1, "Title": 1, "image": 1, "avg_rating": 1}
    ))

    return render_template("search.html", username=session["username"], recommendations=recommendations)

@app.route('/results')
def search_results():
    if 'username' not in session:
        return redirect("/login")

    keyword = request.args.get('keyword', '').strip()
    results = []

    if keyword:
        movies_collection = db["Movies"]
        results = list(movies_collection.find({
            "Title": {"$regex": keyword, "$options": "i"}
        }))

    return render_template('results.html', results=results, keyword=keyword, username=session['username'])

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')

@app.route('/movie/<int:movie_id>')
def movie_page(movie_id):
    if 'username' not in session:
        return redirect("/login")

    user = db["Users"].find_one({"username": session["username"]})
    user_id = user.get("UserID")

    # 紀錄點擊行為
    db["UserActions"].insert_one({
        "UserID": user_id,
        "Action_Type": "click",
        "MovieID": movie_id,
        "Rating": None,
    })

    # 轉跳至靜態網頁（已事先產出）
    return redirect(f"/static/movie_pages/{movie_id}.html")

@app.route('/api/check_favorite/<int:movie_id>')
def check_favorite(movie_id):
    if 'username' not in session:
        return jsonify({"favorite": False})

    user = db["Users"].find_one({"username": session["username"]})
    if not user:
        return jsonify({"favorite": False})

    user_id = user.get("UserID")

    fav_action = actions_collection.find_one({
        "UserID": user_id,
        "MovieID": movie_id,
        "Action_Type": "addtofav",
        "Favorite": True
    })

    return jsonify({"favorite": fav_action is not None})

@app.route('/rate', methods=['POST'])
def rate_movie():
    if 'username' not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "message": "Invalid JSON"
        }), 400

    movie_id = data.get("movie_id")
    rating = data.get("rating")

    if type(movie_id) is not int or movie_id <= 0:
        return jsonify({
            "success": False,
            "message": "Invalid movie ID"
        }), 400

    if type(rating) is not int or not 1 <= rating <= 5:
        return jsonify({
            "success": False,
            "message": "Rating must be an integer from 1 to 5"
        }), 400

    if not movies_collection.find_one({"MovieID": movie_id}):
        return jsonify({
            "success": False,
            "message": "Movie not found"
        }), 404

    user = collection.find_one({
        "username": session["username"]
    })
    if not user:
        return jsonify({
            "success": False,
            "message": "User not found"
        }), 404

    user_id = user["UserID"]

    # 更新已有評分，或新增一筆評分紀錄
    actions_collection.update_one(
        {
            "UserID": user_id,
            "MovieID": movie_id,
            "Action_Type": "rate"
        },
        {
            "$set": {
                "UserID": user_id,
                "MovieID": movie_id,
                "Action_Type": "rate",
                "Rating": rating
            }
        },
        upsert=True
    )

    return jsonify({"success": True})


@app.route('/favorite', methods=['POST'])
def favorite_movie():
    if 'username' not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "message": "Invalid JSON"
        }), 400

    movie_id = data.get("movie_id")
    favorite = data.get("favorite")

    if type(movie_id) is not int or movie_id <= 0:
        return jsonify({
            "success": False,
            "message": "Invalid movie ID"
        }), 400

    if type(favorite) is not bool:
        return jsonify({
            "success": False,
            "message": "Favorite must be true or false"
        }), 400

    if not movies_collection.find_one({"MovieID": movie_id}):
        return jsonify({
            "success": False,
            "message": "Movie not found"
        }), 404

    user = collection.find_one({
        "username": session["username"]
    })
    if not user:
        return jsonify({
            "success": False,
            "message": "User not found"
        }), 404

    user_id = user["UserID"]

    if favorite:
        # 加入收藏；重複操作時更新同一筆紀錄
        actions_collection.update_one(
            {
                "UserID": user_id,
                "MovieID": movie_id,
                "Action_Type": "addtofav"
            },
            {
                "$set": {
                    "UserID": user_id,
                    "MovieID": movie_id,
                    "Action_Type": "addtofav",
                    "Favorite": True
                }
            },
            upsert=True
        )
    else:
        # 取消收藏，保留評分與點擊紀錄
        actions_collection.delete_many({
            "UserID": user_id,
            "MovieID": movie_id,
            "Action_Type": "addtofav"
        })

    return jsonify({"success": True})

@app.route('/favorites')
def favorites():
    if 'username' not in session:
        return redirect("/login")

    user = collection.find_one({"username": session["username"]})
    if not user:
        return redirect("/login")

    user_id = user.get("UserID")
    actions_collection = db["UserActions"]
    movies_collection = db["Movies"]

    # 找出加到最愛的 MovieID
    fav_actions = actions_collection.find({
        "UserID": user_id,
        "Action_Type": "addtofav",
        "Favorite": True
    })

    movie_ids = [fav["MovieID"] for fav in fav_actions]

    # 取得這些電影的詳細資訊
    favorites = list(movies_collection.find({
        "MovieID": { "$in": movie_ids }
    }))

    return render_template("favorites.html", favorites=favorites, username=session["username"])

@app.route('/rankings')
def rankings():
    if 'username' not in session:
        return redirect("/login")
    return render_template("rankings.html", rankings=rankings, username=session["username"])

# 安全的檔案名稱（不含 Windows 不合法字元）
MODEL_FILENAME = "recommender_cache.pkl"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)
MODEL_EXPIRY_DAYS = 3   # 超過 3 天會自動重訓

def save_recommender_to_file(predicted_ratings, user_ids, movie_ids, user_means, user_genre_pref, combined):
    """安全儲存推薦模型"""
    data = {
        "predicted_ratings": predicted_ratings,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "user_means": user_means,
        "user_genre_pref": user_genre_pref,
        "combined": combined,
        "timestamp": datetime.now()
    }

    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(data, f)
        print(f"[Recommender] 模型快取已儲存: {MODEL_PATH}")
    except OSError as e:
        print(f"[Recommender] 儲存快取失敗: {e}")

def load_recommender_from_file():
    """安全載入推薦模型"""
    if not os.path.exists(MODEL_PATH):
        print("[Recommender] 快取不存在")
        return None, [], [], None, {}, None

    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
    except (OSError, pickle.UnpicklingError) as e:
        print(f"[Recommender] 載入快取失敗: {e}")
        return None, [], [], None, {}, None

    # 檢查是否過期
    timestamp = data.get("timestamp", datetime.min)
    if datetime.now() - timestamp > timedelta(days=MODEL_EXPIRY_DAYS):
        print("[Recommender] 快取過期，需要重建模型")
        return None, [], [], None, {}, None

    print("[Recommender] 成功載入快取")
    return (
        data.get("predicted_ratings"),
        data.get("user_ids", []),
        data.get("movie_ids", []),
        data.get("user_means"),
        data.get("user_genre_pref", {}),
        data.get("combined")
    )

# === 全域變數（推薦系統使用） ===
predicted_ratings = None
user_ids = []        # 原始 user id 列表（保留原型別）
movie_ids = []       # 原始 movie id 列表
user_means = None
user_genre_pref = {}   # keys 都會是 str(user_id)
combined = pd.DataFrame()
user_index_map = {}    # map: str(user_id) -> row index

# === 建立推薦模型（行為 + 評分 + 類型偏好） ===
def build_recommender_with_genre():
    global combined

    # 讀取使用者行為與電影資料
    raw = pd.DataFrame(list(actions_collection.find({}, {"_id": 0})))
    movies = pd.DataFrame(list(movies_collection.find({}, {"_id": 0})))

    if raw.empty or movies.empty:
        return None, None, None, None, {}

    # 權重與評分對應表
    action_weights = {"click": 2.0, "addtofav": 4.0}
    rating_map = {1: 0.5, 2: 1.0, 3: 3.0, 4: 4.0, 5: 5.0}

    # === 行為轉換成偏好分數 ===
    def map_action(row):
        if row.get("Action_Type") == "addtofav":
            return action_weights["addtofav"]

        if pd.notna(row.get("Rating")):
            try:
                r = int(row["Rating"])
            except:
                return 0.0
            return rating_map.get(r, float(r))

        return action_weights.get(row.get("Action_Type"), 0.0)

    raw["Preference"] = raw.apply(map_action, axis=1)

    # 聚合同一 user-movie 的多次行為
    combined = raw.groupby(["UserID", "MovieID"], as_index=False)["Preference"].sum()

    if combined.empty:
        return None, None, None, None, {}

    # 建立 User-Item 矩陣
    ratings = combined.rename(columns={"Preference": "Rating"})
    user_item_matrix = ratings.pivot(index="UserID", columns="MovieID", values="Rating")

    # 中心化
    user_means = user_item_matrix.mean(axis=1)
    user_ids = user_item_matrix.index.tolist()
    movie_ids = user_item_matrix.columns.tolist()
    X = user_item_matrix.sub(user_means, axis=0).fillna(0).values

    # 資料不足時，改用備用推薦
    if min(X.shape) < 2:
        return None, None, None, None, {}

    # === SVD 分解 ===
    k = min(20, min(X.shape) - 1)  # 避免資料太少時出錯
    svd = TruncatedSVD(n_components=k, random_state=42)
    user_factors = svd.fit_transform(X)
    item_factors = svd.components_.T
    predicted_ratings = np.dot(user_factors, item_factors.T) + user_means.values[:, None]

    # === 計算使用者的 genre 偏好 ===
    user_genre_pref = {}
    for user_id in ratings["UserID"].unique():
        prefs = ratings[ratings["UserID"] == user_id]
        genre_scores = {}
        for _, row in prefs.iterrows():
            movie_id = row["MovieID"]
            pref = row["Rating"]
            genre_series = movies[movies["MovieID"] == movie_id]["Genres"]
            if not genre_series.empty:
                genres = genre_series.values[0].split("|")
                for g in genres:
                    genre_scores[g] = genre_scores.get(g, 0) + pref
        total = sum(genre_scores.values())
        if total > 0:
            for g in genre_scores:
                genre_scores[g] /= total
        user_genre_pref[user_id] = genre_scores

    return predicted_ratings, user_ids, movie_ids, user_means, user_genre_pref

# === 推薦函數 ===
def recommend_movies(user_id, n=10, alpha=0.7, beta=0.3):
    global predicted_ratings, user_ids, movie_ids, user_means, user_genre_pref, combined

    # 若沒有行為紀錄或 SVD 無法生成 → 推薦評分最高的電影
    if predicted_ratings is None or user_id not in user_ids or combined.empty:
        top_movies = list(movies_collection.find().sort("avg_rating", -1).limit(n))
        return [
            {
                "MovieID": m.get("MovieID"),
                "Title": m.get("Title"),
                "image": m.get("image", ""),
            }
            for m in top_movies
        ]

    user_idx = user_ids.index(user_id)
    user_pred = predicted_ratings[user_idx]

    # 使用者已看過或互動過的電影
    seen_movies = combined[combined["UserID"] == user_id]["MovieID"].tolist()

    recommendations = []
    for i, score in enumerate(user_pred):
        movie_id = movie_ids[i]
        if movie_id in seen_movies:
            continue

        movie_doc = movies_collection.find_one({"MovieID": movie_id})
        if not movie_doc:
            continue

        genres = movie_doc.get("Genres", "").split("|") if movie_doc.get("Genres") else []
        genre_bonus = sum(user_genre_pref.get(user_id, {}).get(g, 0) for g in genres)
        final_score = alpha * score + beta * genre_bonus

        recommendations.append({
            "MovieID": movie_id,
            "Title": movie_doc.get("Title", "Unknown"),
            "image": movie_doc.get("image", ""),
            "Score": round(final_score, 2),
        })

    recommendations = sorted(recommendations, key=lambda x: x["Score"], reverse=True)[:n]
    return recommendations

AGENT_API_URL = "http://127.0.0.1:8000/extract_genres"  # 這是 agent.py 的 API

@app.route("/assistant", methods=["GET"])
def assistant_page():
    if "username" not in session:
        return redirect("/login")
    return render_template("assistant.html", username=session["username"])

@app.route("/assistant/recommend", methods=["POST"])
def assistant_recommend():
    if "username" not in session:
        return redirect("/login")

    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return render_template("assistant.html", error="請輸入描述。", username=session["username"])

    # 🔹 呼叫 Gemini AI 產生電影類型 (接收 Wanted 和 Banned)
    wanted_genres, banned_genres = generate_movie_genres(prompt)
    session["assistant_genres"] = wanted_genres # 顯示時只顯示想要的類型
    
    logging.basicConfig(filename="assistant.log", level=logging.INFO, encoding="utf-8")
    logging.info(f"[AI Assistant] User: {prompt} -> Wanted: {wanted_genres}, Banned: {banned_genres}")

    # 🔹 定義清理函式 (因為有兩個 list 要清，寫成函式比較整潔)
    def clean_genre_list(g_list):
        allowed_genres = [
            "Action", "Adventure", "Animation", "Children's",
            "Comedy", "Crime", "Documentary", "Drama",
            "Fantasy", "Film-Noir", "Horror", "Musical",
            "Mystery", "Romance", "Sci-Fi", "Thriller",
            "War", "Western",
        ]

        genre_lookup = {
            genre.lower(): genre
            for genre in allowed_genres
        }

        if not isinstance(g_list, list):
            return []

        safe_list = []
        for genre in g_list:
            if not isinstance(genre, str):
                continue

            canonical = genre_lookup.get(genre.strip().lower())
            if canonical:
                escaped = re.escape(canonical)
                if escaped not in safe_list:
                    safe_list.append(escaped)

        return safe_list

    safe_wanted = clean_genre_list(wanted_genres)
    safe_banned = clean_genre_list(banned_genres)

    if not safe_wanted and not safe_banned:
        return render_template("assistant.html", error="AI 未產生有效類型，請重新描述。", username=session["username"])

    # 🔹 建立 MongoDB 查詢 (邏輯：包含 Wanted 且 排除 Banned)
    query_conditions = []

    # 條件 1: 包含任一 Wanted 類型 (OR)
    if safe_wanted:
        query_conditions.append({
            "$or": [{"Genres": {"$regex": g, "$options": "i"}} for g in safe_wanted]
        })
    
    # 條件 2: 排除所有 Banned 類型 (NOT REGEX)
    for b in safe_banned:
        query_conditions.append({
            "Genres": {"$not": {"$regex": b, "$options": "i"}}
        })

    # 合併查詢 (AND)
    query = {"$and": query_conditions} if query_conditions else {}

    movies = list(movies_collection.find(
        query,
        {"_id": 0, "MovieID": 1, "Title": 1, "image": 1, "Genres": 1, "avg_rating": 1}
    ))
    movies = sorted(movies, key=lambda x: x.get("avg_rating", 0), reverse=True)[:10]

    return render_template("assistant_results.html", genres=wanted_genres, movies=movies, username=session["username"])
if __name__ == '__main__':
    app.run(debug=False)
