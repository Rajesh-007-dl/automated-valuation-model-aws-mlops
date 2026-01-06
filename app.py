import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import boto3, os
from pathlib import Path
st.set_page_config(
    page_title="Housing Price AI",
    page_icon="https://cdn-icons-png.flaticon.com/512/1040/1040993.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* GLOBAL FONTS & SPACING */
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        h1, h2, h3 {font-family: 'Helvetica Neue', sans-serif; font-weight: 700;}
        
        /* MODERN BUTTON LAYOUT & STYLE */
        /* Targets the main 'Run Predictions' button */
        div.stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            color: white;
            border: none;
            padding: 0.6rem 1rem;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Button Hover Effect */
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
            background: linear-gradient(90deg, #182848 0%, #4b6cb7 100%);
            color: #ffffff;
            border: none;
        }

        /* METRIC CARDS (Dark/Light Mode Compatible) */
        div[data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.05); /* Transparent white */
            border: 1px solid rgba(128, 128, 128, 0.2);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            backdrop-filter: blur(10px); /* Glassmorphism effect */
        }
        
        /* Sidebar Styling Fixes */
        section[data-testid="stSidebar"] {
            padding-top: 1rem;
        }
    </style>
""",
    unsafe_allow_html=True,
)
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
S3_BUCKET = os.getenv("S3_BUCKET", "housing-regression-data")
REGION = os.getenv("AWS_REGION", "Asia Pacific (Mumbai)")

s3 = boto3.client("s3", region_name=REGION)


def load_from_s3(key, local_path):
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        with st.spinner(f"📥 Fetching resources..."):
            s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)


# Paths
HOLDOUT_ENGINEERED_PATH = load_from_s3(
    "processed/feature_engineered_holdout.csv",
    "data/processed/feature_engineered_holdout.csv",
)
HOLDOUT_META_PATH = load_from_s3(
    "processed/cleaning_holdout.csv", "data/processed/cleaning_holdout.csv"
)


@st.cache_data
def load_data():
    fe = pd.read_csv(HOLDOUT_ENGINEERED_PATH)
    meta = pd.read_csv(HOLDOUT_META_PATH, parse_dates=["date"])[["date", "city_full"]]

    if len(fe) != len(meta):
        min_len = min(len(fe), len(meta))
        fe = fe.iloc[:min_len].copy()
        meta = meta.iloc[:min_len].copy()

    disp = pd.DataFrame(index=fe.index)
    disp["date"] = meta["date"]
    disp["region"] = meta["city_full"]
    disp["year"] = disp["date"].dt.year
    disp["month"] = disp["date"].dt.month
    disp["actual_price"] = fe["price"]
    return fe, disp


fe_df, disp_df = load_data()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1040/1040993.png", width=60)
    st.title("Control Panel")

    with st.container():
        st.write("**Filters**")

        years = sorted(disp_df["year"].unique())
        months = list(range(1, 13))
        month_names = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        regions = ["All"] + sorted(disp_df["region"].dropna().unique())

        selected_year = st.selectbox("Year", years, index=0)
        selected_month = st.selectbox(
            "Month", months, format_func=lambda x: f"{x} - {month_names[x]}", index=0
        )
        selected_region = st.selectbox("Region", regions, index=0)

    st.markdown("---")


    st.write("**Action**")
    run_btn = st.button("✨ Generate Predictions", use_container_width=True)

    st.caption(f"Connected to: `{S3_BUCKET}`")

st.title(" Housing Price AI")
st.markdown("### Market Analysis & Inference Engine")

if run_btn:
    # Filter Data
    mask = (disp_df["year"] == selected_year) & (disp_df["month"] == selected_month)
    if selected_region != "All":
        mask &= disp_df["region"] == selected_region
    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning(
            f"No records found for **{month_names[selected_month]} {selected_year}** in **{selected_region}**."
        )
    else:
        with st.spinner("🤖 Crunching numbers..."):
            try:
         
                payload = fe_df.loc[idx].to_dict(orient="records")
                resp = requests.post(API_URL, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                # DataFrame Prep
                view = disp_df.loc[idx].copy().sort_values("date")
                view["prediction"] = data.get("predictions", [])

                if data.get("actuals"):
                    view["actual_price"] = data["actuals"]

                view["error"] = view["prediction"] - view["actual_price"]
                view["pct_error"] = (view["error"].abs() / view["actual_price"]) * 100

                mae = view["error"].abs().mean()
                rmse = (view["error"] ** 2).mean() ** 0.5
                acc = 100 - view["pct_error"].mean()

              
                st.markdown("#### 🎯 Accuracy Report")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Predicted Avg", f"${view['prediction'].mean():,.0f}")
                c2.metric(
                    "MAE (Error)",
                    f"${mae:,.0f}",
                    delta="Lower is better",
                    delta_color="inverse",
                )
                c3.metric("RMSE", f"${rmse:,.0f}", delta_color="off")
                c4.metric("Model Accuracy", f"{acc:.1f}%")

                st.markdown("---")

                col_main, col_side = st.columns([2, 1])

       
                if selected_region == "All":
                    trend_df = disp_df[disp_df["year"] == selected_year].copy()
                else:
                    trend_df = disp_df[
                        (disp_df["year"] == selected_year)
                        & (disp_df["region"] == selected_region)
                    ].copy()

        
                trend_payload = fe_df.loc[trend_df.index].to_dict(orient="records")
                trend_resp = requests.post(API_URL, json=trend_payload, timeout=60)
                trend_df["prediction"] = trend_resp.json().get("predictions", [])

                monthly = (
                    trend_df.groupby("month")[["actual_price", "prediction"]]
                    .mean()
                    .reset_index()
                )

                fig_trend = go.Figure()
                fig_trend.add_trace(
                    go.Scatter(
                        x=monthly["month"],
                        y=monthly["actual_price"],
                        name="Actual",
                        line=dict(color="#00CC96", width=3),
                    )
                )
                fig_trend.add_trace(
                    go.Scatter(
                        x=monthly["month"],
                        y=monthly["prediction"],
                        name="Predicted",
                        line=dict(color="#636EFA", width=3, dash="dot"),
                    )
                )

                fig_trend.add_shape(
                    type="rect",
                    x0=selected_month - 0.4,
                    x1=selected_month + 0.4,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    fillcolor="white",
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                )

                fig_trend.update_layout(
                    title="Yearly Price Trend",
                    xaxis_title="Month",
                    yaxis_title="Price ($)",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=380,
                    legend=dict(orientation="h", y=1.1),
                )

                with col_main:
                    st.plotly_chart(fig_trend, use_container_width=True)

                with col_side:
                    st.markdown("#### Error Distribution")
                    fig_dist = px.histogram(
                        view, x="error", nbins=15, color_discrete_sequence=["#EF553B"]
                    )
                    fig_dist.update_layout(
                        showlegend=False,
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=350,
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

        
                st.markdown("### 📋 Data Breakdown")
                with st.expander("Expand to view full record list"):
                    st.dataframe(
                        view[
                            [
                                "date",
                                "region",
                                "actual_price",
                                "prediction",
                                "error",
                                "pct_error",
                            ]
                        ],
                        use_container_width=True,
                        column_config={
                            "actual_price": st.column_config.NumberColumn(
                                "Actual", format="$%d"
                            ),
                            "prediction": st.column_config.NumberColumn(
                                "Predicted", format="$%d"
                            ),
                            "pct_error": st.column_config.NumberColumn(
                                "Err %", format="%.2f%%"
                            ),
                            "date": st.column_config.DateColumn("Date"),
                        },
                    )

            except Exception as e:
                st.error(f"Error during inference: {str(e)}")
else:

    st.info(
        "👈 Select your filters in the sidebar and click 'Generate Predictions' to start."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"### {len(regions)-1}\n**Regions Tracked**")
    with c2:
        st.markdown(f"### {len(disp_df):,}\n**Data Points**")
    with c3:
        st.markdown(f"### {disp_df['year'].nunique()}\n**Years History**")
