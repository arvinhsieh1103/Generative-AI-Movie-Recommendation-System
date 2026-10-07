import json
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def generate_movie_genres(prompt: str):
    """
    根據使用者輸入，分析出「想要」與「避開」的電影類型。
    回傳: (wanted_genres, banned_genres) 兩個 list
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
Role: You are a semantic classifier for a movie recommendation system using the MovieLens dataset.
Task: Analyze the User Input to extract TWO lists:
1. **Wanted Genres**: Based on positive sentiments, desired moods, or themes.
2. **Banned Genres**: Based on negative sentiments (e.g., "no horror", "avoid sad movies", "not for kids").

Allowed Genre List:
[Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western]

User Input: "{prompt}"

Instructions:
1. **Semantic Analysis**:
   - If the user says "I want to cry", Wanted=['Drama', 'Romance'].
   - If the user says "I want excitement but no ghosts", Wanted=['Action', 'Thriller'], Banned=['Horror'].
   - If the input is vague, infer the most likely popular genres for 'Wanted'.
2. **Constraint - No Overlap**: A genre cannot appear in both lists. If in doubt, prioritize the User's explicit 'Banned' request.
3. **Output Constraint**: Output ONLY a raw JSON object with no markdown formatting (no ```json code blocks). 

Output JSON Structure:
{{
  "wanted": ["Genre A", "Genre B"],
  "banned": ["Genre C"]
}}
"""
    )

    text = response.text.strip()
    
    # 移除可能被模型加入的 markdown 標記 (防呆)
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "")
    
    try:
        data = json.loads(text)
        wanted = data.get("wanted", [])
        banned = data.get("banned", [])
    except json.JSONDecodeError:
        # 萬一模型回傳格式爛掉的備用方案 (或是你可以 print(text) 來看錯誤訊息)
        print(f"JSON Parsing Error. Raw text: {text}")
        wanted = []
        banned = []

    return wanted, banned