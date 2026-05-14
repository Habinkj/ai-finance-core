from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import os
from dotenv import load_dotenv
import google.generativeai as genai
import warnings

warnings.filterwarnings('ignore')

# --- SECURE LLM CONFIGURATION ---
load_dotenv() 
api_key = os.getenv("GEMINI_API_KEY") 

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    print("⚠️ WARNING: GEMINI_API_KEY not found in environment.")
    model = None
# --------------------------------

app = FastAPI()

print("🧠 Booting Hybrid AI Core: Training Scikit-Learn Models...")
np.random.seed(42)

# 🔥 FIX 2: Added 'tx_count' so the React graph has a Y-axis metric
df_train = pd.DataFrame({
    'total_spend': np.random.normal(loc=1500, scale=600, size=200),
    'tx_count': np.random.randint(1, 30, size=200), 
    'max_category': np.random.normal(loc=1500, scale=600, size=200) * np.random.uniform(0.3, 0.9, size=200)
})

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(df_train)
iso_forest = IsolationForest(contamination=0.05, random_state=42).fit(df_train)

# Attach the cluster labels to the training data so we can map it on the frontend
df_train['cluster'] = kmeans.labels_

cluster_labels = {
    0: "MODERATE SPENDER",
    1: "AGGRESSIVE SAVER",
    2: "HIGH-RISK IMPULSIVE"
}
print("✅ ML Models Ready.")

class CategorySpend(BaseModel):
    category: str
    amount: float

# 🔥 FIX 4: Upgraded the Pydantic model to explicitly accept the clusterData list
class AIAnalysis(BaseModel):
    riskLevel: str
    behaviorPattern: str
    flaggedTransactions: List[str]
    humanAdvice: str 
    clusterData: List[Dict[str, Any]]

@app.post("/analyze", response_model=AIAnalysis)
async def analyze_behavior(spend_data: List[CategorySpend]):
    
    if not spend_data:
        return AIAnalysis(riskLevel="UNKNOWN", behaviorPattern="NO DATA", flaggedTransactions=[], humanAdvice="No data available.", clusterData=[])

    # 🔥 INDENTATION FIXED 🔥
    total_spent = sum(item.amount for item in spend_data)
    max_category_spend = max(item.amount for item in spend_data)
    max_category_name = next(item.category for item in spend_data if item.amount == max_category_spend)
    
    # We estimate the user's transaction frequency based on how many items were sent
    tx_count_current = len(spend_data)

    # Predict the user's cluster
    df_current = pd.DataFrame({'total_spend': [total_spent], 'tx_count': [tx_count_current], 'max_category': [max_category_spend]})
    cluster_id = kmeans.predict(df_current)[0]
    behavior_pattern = cluster_labels.get(int(cluster_id), "UNKNOWN")
    anomaly_score = iso_forest.predict(df_current)[0] 

    risk = "LOW"
    flagged = []

    if anomaly_score == -1:
        risk = "HIGH"
        flagged.append(f"Anomaly: {max_category_name} (${max_category_spend})")
    elif total_spent > 1500:
        risk = "MEDIUM"

    # --- THE HYBRID BRIDGE (LLM API CALL) ---
    print("🤖 Generating humanized LLM advice...")
    prompt = f"""
    You are an empathetic, professional financial advisor built into a mobile app.
    A user's spending data was just run through a machine learning model.
    Their mathematical classification is: {behavior_pattern}.
    Their risk level is: {risk}.
    Top spending category: {max_category_name} (${max_category_spend}).
    
    Write 2 short, encouraging, and highly specific sentences of advice for this user. 
    Do not use robotic language. Be conversational.
    """
    
    try:
        if model:
            llm_response = model.generate_content(prompt)
            advice = llm_response.text.strip()
        else:
            advice = "Our AI is currently offline, but keep an eye on your budget!"
    except Exception as e:
        print(f"❌ LLM CRASH REPORT: {e}")  
        advice = "We analyzed your spending, but our advisory system is currently resting. Keep an eye on your tech budget!"
    # ----------------------------------------

    # 🔥 FIX 1: Loop over df_train instead of the undefined 'df'
    cluster_points = []
    
    # We send a sample of the background training map (75 points) to keep the graph fast
    for index, row in df_train.head(75).iterrows():
        cluster_points.append({
            "spend": round(row['total_spend'], 2),
            "frequency": int(row['tx_count']), 
            "cluster": int(row['cluster'])
        })

    # We append the user's current LIVE data point to the graph
    cluster_points.append({
        "spend": round(total_spent, 2),
        "frequency": int(tx_count_current),
        "cluster": int(cluster_id)
    })

    # 🔥 FIX 3: Corrected variable names (risk -> riskLevel, behavior_pattern -> behaviorPattern)
    return {
        "riskLevel": risk,
        "behaviorPattern": behavior_pattern,
        "flaggedTransactions": flagged,
        "humanAdvice": advice,
        "clusterData": cluster_points  
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)