# app.py
import os, json
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
CONFIG_PATH = DATA_DIR / "embedding_config.json"

# --- Load Embeddings ---
def get_embeddings_from_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    if cfg["provider"] == "hf":
        return HuggingFaceEmbeddings(model_name=cfg["model"])
    return OpenAIEmbeddings(model=cfg["model"])

def detect_industry(startup_idea):
    """Detect the relevant industry/domain from the startup idea."""
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    if not llm:
        st.error("OpenAI API key is missing. Please set it in the environment variables.")
        return "Unknown Industry"

    prompt = f"Identify the industry/domain for the following startup idea: {startup_idea}. Keep the response concise."
    industry = llm.invoke(prompt).content.strip()
    return industry or "Unknown Industry"

def generate_swot(startup_idea, feasibility_score):
    """Generate a context-aware SWOT analysis based on the feasibility score."""
    if feasibility_score < 30:
        # Very few strengths/opportunities, highlight flaws
        swot = {
            "Strengths": ["Nostalgia", "Low cost"],
            "Weaknesses": ["Obsolete technology", "No demand"],
            "Opportunities": ["Niche audience"],
            "Threats": ["Competition from modern alternatives", "Regulatory risks"]
        }
    elif 30 <= feasibility_score < 60:
        # Balanced but cautious SWOT
        swot = {
            "Strengths": ["Moderate scalability", "Some market interest"],
            "Weaknesses": ["High initial costs", "Limited market data"],
            "Opportunities": ["Emerging trends", "Potential partnerships"],
            "Threats": ["Economic instability", "Regulatory hurdles"]
        }
    else:
        # More optimistic SWOT allowed
        swot = {
            "Strengths": ["Innovative idea", "Scalable model"],
            "Weaknesses": ["High competition", "Resource constraints"],
            "Opportunities": ["Growing market", "Tech advancements"],
            "Threats": ["Market saturation", "Regulatory risks"]
        }

    # Limit to 2 points per quadrant
    for key in swot:
        swot[key] = swot[key][:2]

    return swot

def calculate_feasibility(startup_idea, budget):
    """Calculate a feasibility score based on the startup idea and budget."""
    if budget < 30000:
        st.warning("Budget must be at least ₹30,000 to proceed.")
        return None

    # Stricter evaluation for outdated or irrelevant ideas
    if "DVD" in startup_idea or "floppy disk" in startup_idea or "pager" in startup_idea:
        return 20  # Low score for outdated ideas

    # Mocked feasibility score for now
    feasibility_score = 75
    return feasibility_score

def generate_recommendation(feasibility_score, swot):
    """Generate a final recommendation based on feasibility score and SWOT analysis."""
    if feasibility_score < 30:
        return "❌ Not Recommended: The idea is outdated or has very low demand."
    elif feasibility_score > 70:
        return "✅ Recommended: The idea has strong potential based on the analysis."
    else:
        return "⚠️ Consider with Caution: Address the weaknesses identified before proceeding."

def predict_future_trends(industry):
    """Predict 3–4 concise future trends for the given industry."""
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    if not llm:
        st.error("OpenAI API key is missing. Please set it in the environment variables.")
        return None, None

    prompt = f"Forecast 3–4 key market trends for the next 3–5 years in the {industry} industry. Ensure relevance to 2025 and beyond."
    trends_text = llm.invoke(prompt).content

    # Mocked numeric data for visualization
    trend_data = pd.DataFrame({
        "Year": [2025, 2026, 2027, 2028],
        "Growth Forecast (CAGR %)": [5, 6, 7, 8]
    })

    return trends_text, trend_data

def generate_growth_forecast(feasibility_score):
    """Generate a growth forecast trend based on the feasibility score."""
    if feasibility_score < 30:
        # Declining trend
        trend_data = pd.DataFrame({
            "Year": [2025, 2026, 2027, 2028, 2029],
            "Growth Forecast (CAGR %)": [5, 3, 2, 1, 0]
        })
    elif 30 <= feasibility_score < 60:
        # Flat or unstable trend
        trend_data = pd.DataFrame({
            "Year": [2025, 2026, 2027, 2028, 2029],
            "Growth Forecast (CAGR %)": [3, 4, 3, 4, 3]
        })
    else:
        # Increasing trend
        trend_data = pd.DataFrame({
            "Year": [2025, 2026, 2027, 2028, 2029],
            "Growth Forecast (CAGR %)": [5, 7, 8, 10, 12]
        })

    return trend_data

# --- Main App ---
def main():
    load_dotenv()
    st.set_page_config(page_title="Startup Analyzer", page_icon="🚀", layout="wide")
    st.title("🚀 AI Startup Analyzer")

    # Input fields
    startup_idea = st.text_area("💡 Enter your Startup Idea:", placeholder="e.g., AI-powered food delivery platform")
    budget = st.number_input("💰 Enter your Budget (in ₹ INR):", min_value=0, step=1000)

    if st.button("Analyze Startup"):
        if not startup_idea:
            st.error("Please enter a startup idea.")
            return

        # Step 1: Validate Budget
        if budget < 30000:
            st.warning("Budget must be at least ₹30,000 to proceed.")
            return

        # Step 2: Detect Industry
        industry = detect_industry(startup_idea)
        st.subheader("🏭 Detected Industry")
        st.write(industry)

        # Step 3: Calculate Feasibility Score
        feasibility_score = calculate_feasibility(startup_idea, budget)
        if feasibility_score is not None:
            st.subheader("📊 Feasibility Score")
            st.write(f"**Score:** {feasibility_score}/100")

        # Step 4: Generate SWOT Analysis
        swot = generate_swot(startup_idea, feasibility_score)
        if swot:
            st.subheader("🗂️ SWOT Analysis")
            swot_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in swot.items()]))
            st.table(swot_df)

        # Step 5: Generate Final Recommendation
        recommendation = generate_recommendation(feasibility_score, swot)
        st.subheader("✅ / ❌ Final Recommendation")
        st.success(recommendation)

        # Step 6: Generate Growth Forecast Chart
        trend_data = generate_growth_forecast(feasibility_score)
        st.subheader("📈 Growth Forecast")
        st.line_chart(trend_data.set_index("Year")[["Growth Forecast (CAGR %)"]])

        # Step 7: Predict Future Trends
        trends_text, _ = predict_future_trends(industry)
        if trends_text:
            st.subheader("🔮 Future Trends")
            st.write(trends_text)

if __name__ == "__main__":
    main()
