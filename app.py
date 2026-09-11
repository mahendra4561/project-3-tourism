import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from data_preparation import prepare_dataset

final_dataset, encoders, scaler = prepare_dataset()

st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")

page_backgrounds = {
    "Overview": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",
    "Predictions": "https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d",
    "Recommendations": "https://images.unsplash.com/photo-1504674900247-0877df9cc836",
    "Trends": "https://images.unsplash.com/photo-1500534314209-a25ddb2bd429",
    "Model Performance": "https://images.unsplash.com/photo-1504384308090-c894fdcc538d",
    "Data Explorer": "https://images.unsplash.com/photo-1524995997946-a1c2e315a42f",
    "Geo Insights": "https://images.unsplash.com/photo-1473187983305-f615310e7daa",
    "Documentation": "https://images.unsplash.com/photo-1515879218367-8466d910aaa4",
    "Settings": "https://images.unsplash.com/photo-1519389950473-47ba0277781c",
}

def set_background(page):
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("{page_backgrounds[page]}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    [data-testid="stHeader"] {{background: rgba(0,0,0,0);}}
    [data-testid="stSidebar"] {{background: rgba(255,255,255,0.85);}}
    .kpi-card {{
        background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%);
        padding: 20px; border-radius: 12px; text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .kpi-title {{font-size: 1.2rem; font-weight: bold; color: #333;}}
    .kpi-value {{font-size: 1.5rem; color: #111;}}
    .card {{
        padding: 20px; margin: 15px 0; border-radius: 15px;
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        text-align: center;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

nav_options = {
    "🏠 Overview": "Overview",
    "🔮 Predictions": "Predictions",
    "🎁 Recommendations": "Recommendations",
    "📊 Trends": "Trends",
    "📈 Model Performance": "Model Performance",
    "📂 Data Explorer": "Data Explorer",
    "🗺️ Geo Insights": "Geo Insights",
    "📑 Documentation": "Documentation",
    "⚙️ Settings": "Settings"
}
selected_page = st.radio("Navigation", list(nav_options.keys()), horizontal=True, label_visibility="collapsed")
page = nav_options[selected_page]

set_background(page)

if page == "Overview":
    st.header("📊 Preview of Clean Dataset")
    st.dataframe(final_dataset.head())

    col1, col2, col3 = st.columns(3)
    with col1:
        users_by_continent = final_dataset["Continent"].nunique()
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>🌍 Users by Continent</div><div class='kpi-value'>{users_by_continent}</div></div>", unsafe_allow_html=True)
    with col2:
        top_attraction = final_dataset.groupby("Attraction")["Rating"].mean().idxmax()
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>🏝️ Top Attraction</div><div class='kpi-value'>{top_attraction}</div></div>", unsafe_allow_html=True)
    with col3:
        avg_rating = final_dataset["Rating"].mean()
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>⭐ Average Rating</div><div class='kpi-value'>{avg_rating:.2f}</div></div>", unsafe_allow_html=True)

    st.subheader("📈 Visit Mode Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="VisitMode", data=final_dataset, palette="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("🌍 Average Ratings by Continent")
    avg_rating_continent = final_dataset.groupby("Continent")["Rating"].mean()
    st.bar_chart(avg_rating_continent)

elif page == "Predictions":
    st.header("🔮 Predictions")
    user_id = st.selectbox("Select UserId", final_dataset["UserId"].unique())
    attraction_id = st.selectbox("Select AttractionId", final_dataset["AttractionId"].unique())

    X = final_dataset[["UserId", "AttractionId"]]
    y = final_dataset["Rating"]
    reg_model = LinearRegression().fit(X, y)
    sample = pd.DataFrame([[user_id, attraction_id]], columns=["UserId", "AttractionId"])
    predicted_rating = reg_model.predict(sample)[0]

    clf_model = RandomForestClassifier().fit(X, final_dataset["VisitMode"])
    predicted_mode = clf_model.predict(sample)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>⭐ Predicted Rating</div><div class='kpi-value'>{predicted_rating:.2f}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>👥 Predicted Visit Mode</div><div class='kpi-value'>{predicted_mode}</div></div>", unsafe_allow_html=True)

elif page == "Recommendations":
    st.header("🎁 Recommended Attractions")
    user_id = st.selectbox("Select UserId for Recommendations", sorted(final_dataset["UserId"].unique()))

    TYPE_ICONS = {
        "Ancient Ruins": "🏛️", "Ballets": "🩰", "Beaches": "🏖️",
        "Caverns & Caves": "🕳️", "Flea & Street Markets": "🛍️",
        "Historic Sites": "🏯", "History Museums": "🏛️", "National Parks": "🌲",
        "Nature & Wildlife Areas": "🐒", "Neighborhoods": "🏘️",
        "Points of Interest & Landmarks": "📍", "Religious Sites": "🛕",
        "Spas": "💆", "Speciality Museums": "🖼️", "Volcanos": "🌋",
        "Water Parks": "💦", "Waterfalls": "🌊",
    }

    @st.cache_data
    def build_item_similarity(data):
        rating_matrix = data.pivot_table(
            index="UserId", columns="AttractionId", values="Rating", aggfunc="mean"
        ).fillna(0)
        sim = cosine_similarity(rating_matrix.T)
        sim_df = pd.DataFrame(sim, index=rating_matrix.columns, columns=rating_matrix.columns)
        return rating_matrix, sim_df

    def recommend_for_user(data, user_id, top_n=5):
        rating_matrix, item_sim_df = build_item_similarity(data)
        seen = data.loc[data["UserId"] == user_id, "AttractionId"].unique()

        if user_id in rating_matrix.index and rating_matrix.loc[user_id].sum() > 0:
            rated = rating_matrix.loc[user_id]
            rated = rated[rated > 0]
            scores = pd.Series(0.0, index=rating_matrix.columns)
            for attraction_id, r in rated.items():
                scores = scores.add(item_sim_df[attraction_id] * r, fill_value=0)
        else:
            scores = data.groupby("AttractionId")["Rating"].mean()

        scores = scores.drop(index=[a for a in seen if a in scores.index], errors="ignore")
        return scores.sort_values(ascending=False).head(top_n)

    top_recs = recommend_for_user(final_dataset, user_id)

    if len(top_recs) == 0:
        st.info("No new recommendations — this user has rated every attraction we have data on.")
    else:
        attraction_info = final_dataset.drop_duplicates(subset="AttractionId").set_index("AttractionId")
        avg_ratings = final_dataset.groupby("AttractionId")["Rating"].mean()

        cols = st.columns(len(top_recs))
        for col, (attraction_id, _) in zip(cols, top_recs.items()):
            info = attraction
