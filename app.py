import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fuel Price Optimizer",
    page_icon="⛽",
    layout="wide"
)

# -----------------------------
# Load trained ML model with error handling
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("demand_model.pkl")
        return model, None
    except FileNotFoundError:
        return None, "Model file 'demand_model.pkl' not found. Please ensure the model is trained and saved."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

model, error_msg = load_model()

# -----------------------------
# Header
# -----------------------------
st.title("⛽ Fuel Price Optimization – ML Based Recommender")
st.write("Recommend the optimal daily price to maximize profit using machine learning predictions.")

if error_msg:
    st.error(error_msg)
    st.info("📝 **Demo Mode:** The app interface is functional. Add your trained model file to enable predictions.")
    st.stop()

# -----------------------------
# Sidebar – Business Guardrails
# -----------------------------
st.sidebar.header("🎯 Business Rules / Constraints")
st.sidebar.write("Set guardrails to ensure price recommendations align with business policies.")

MAX_DAILY_CHANGE = st.sidebar.slider(
    "Max price change per day (₹)", 
    0.1, 2.0, 0.75, 0.05,
    help="Maximum allowed price change from current price"
)
MIN_MARGIN = st.sidebar.slider(
    "Minimum margin per liter (₹)", 
    0.1, 2.0, 0.5, 0.05,
    help="Minimum profit margin required per liter"
)
MAX_COMP_GAP = st.sidebar.slider(
    "Max price above competitors (₹)", 
    0.1, 3.0, 1.5, 0.1,
    help="Maximum price difference above competitor average"
)

st.sidebar.divider()
st.sidebar.info("💡 **Tip:** Adjust these constraints based on your market positioning and business strategy.")

# -----------------------------
# User Input Form
# -----------------------------
st.subheader("📥 Enter Today's Market Inputs")

col1, col2 = st.columns(2)

with col1:
    date = st.date_input("Date", datetime.today())
    price = st.number_input("Last Observed Company Price (₹)", min_value=0.0, step=0.1, value=0.0)
    cost = st.number_input("Today's Cost per Liter (₹)", min_value=0.0, step=0.1, value=0.0)

with col2:
    comp1 = st.number_input("Competitor 1 Price (₹)", min_value=0.0, step=0.1, value=0.0)
    comp2 = st.number_input("Competitor 2 Price (₹)", min_value=0.0, step=0.1, value=0.0)
    comp3 = st.number_input("Competitor 3 Price (₹)", min_value=0.0, step=0.1, value=0.0)

# Input validation
def validate_inputs():
    errors = []
    if price <= 0:
        errors.append("Company price must be greater than 0")
    if cost <= 0:
        errors.append("Cost must be greater than 0")
    if cost >= price:
        errors.append("Cost should be less than current price")
    if comp1 <= 0 or comp2 <= 0 or comp3 <= 0:
        errors.append("All competitor prices must be greater than 0")
    return errors

submitted = st.button("🔍 Run Price Optimization", type="primary", use_container_width=True)

# -----------------------------
# Helper – Build Feature Row
# -----------------------------
def build_feature_row(data):
    """Construct feature dataframe for model prediction"""
    row = pd.DataFrame([data])

    row["comp_avg"] = row[["comp1_price","comp2_price","comp3_price"]].mean(axis=1)
    row["price_gap"] = row["price"] - row["comp_avg"]
    row["margin"] = row["price"] - row["cost"]

    # Placeholder lags (in production these come from pipeline storage)
    row["lag_price"] = row["price"]
    row["lag_volume"] = 0
    row["ma7_price"] = row["price"]
    row["ma7_volume"] = 0

    row["dayofweek"] = pd.to_datetime(row["date"]).dt.dayofweek
    row["month"] = pd.to_datetime(row["date"]).dt.month

    return row

# -----------------------------
# Run Optimization
# -----------------------------
if submitted:
    # Validate inputs
    validation_errors = validate_inputs()
    if validation_errors:
        for error in validation_errors:
            st.error(f"❌ {error}")
        st.stop()

    with st.spinner("🔄 Running optimization algorithm..."):
        today = {
            "date": str(date),
            "price": price,
            "cost": cost,
            "comp1_price": comp1,
            "comp2_price": comp2,
            "comp3_price": comp3
        }

        base_row = build_feature_row(today)
        current_price = price

        # Generate candidate prices
        candidate_prices = np.round(np.arange(
            current_price - 1.0,
            current_price + 1.01,
            0.10
        ), 2)

        results = []
        feature_cols = [
            "price","cost","margin",
            "comp1_price","comp2_price","comp3_price","comp_avg","price_gap",
            "lag_price","lag_volume","ma7_price","ma7_volume",
            "dayofweek","month"
        ]

        # Evaluate each candidate price
        for p in candidate_prices:
            temp = base_row.copy()
            temp["price"] = p
            temp["margin"] = p - temp["cost"]
            temp["price_gap"] = p - temp["comp_avg"]

            # ---------- Guardrails ----------
            if abs(p - current_price) > MAX_DAILY_CHANGE:
                continue
            if temp["margin"].iloc[0] < MIN_MARGIN:
                continue
            if temp["price_gap"].iloc[0] > MAX_COMP_GAP:
                continue

            X = temp[feature_cols]
            volume_pred = model.predict(X)[0]
            profit = (p - temp["cost"].iloc[0]) * volume_pred

            results.append({
                "price": float(p),
                "expected_volume": round(volume_pred, 2),
                "expected_profit": round(profit, 2),
                "margin": round(temp["margin"].iloc[0], 2),
                "price_gap": round(temp["price_gap"].iloc[0], 2)
            })

        results_df = pd.DataFrame(results)

        # Display results
        if len(results_df) == 0:
            st.error("⚠️ No valid price options after applying guardrails. Try relaxing constraints.")
            st.info("**Suggestions:**\n- Increase max daily change limit\n- Decrease minimum margin requirement\n- Increase max competitor gap")
        else:
            best = results_df.sort_values("expected_profit", ascending=False).iloc[0]
            
            # Success metrics
            st.success("✅ Price Recommendation Generated Successfully")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Recommended Price", 
                    f"₹{best['price']:.2f}",
                    delta=f"{best['price'] - current_price:+.2f}"
                )
            with col2:
                st.metric(
                    "Expected Volume", 
                    f"{best['expected_volume']:,.0f} L"
                )
            with col3:
                st.metric(
                    "Expected Profit", 
                    f"₹{best['expected_profit']:,.2f}"
                )
            with col4:
                st.metric(
                    "Profit Margin", 
                    f"₹{best['margin']:.2f}/L"
                )

            # Visualizations
            st.subheader("📊 Price Optimization Analysis")
            
            tab1, tab2, tab3 = st.tabs(["Profit Curve", "Volume vs Price", "Data Table"])
            
            with tab1:
                # Profit vs Price chart
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=results_df['price'],
                    y=results_df['expected_profit'],
                    mode='lines+markers',
                    name='Expected Profit',
                    line=dict(color='#00cc96', width=3),
                    marker=dict(size=8)
                ))
                fig1.add_vline(
                    x=best['price'], 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Optimal Price"
                )
                fig1.add_vline(
                    x=current_price, 
                    line_dash="dot", 
                    line_color="gray",
                    annotation_text="Current Price"
                )
                fig1.update_layout(
                    title="Expected Profit vs Price",
                    xaxis_title="Price (₹)",
                    yaxis_title="Expected Profit (₹)",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                # Volume vs Price chart
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=results_df['price'],
                    y=results_df['expected_volume'],
                    mode='lines+markers',
                    name='Expected Volume',
                    line=dict(color='#636efa', width=3),
                    marker=dict(size=8)
                ))
                fig2.add_vline(
                    x=best['price'], 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Optimal Price"
                )
                fig2.update_layout(
                    title="Expected Volume vs Price",
                    xaxis_title="Price (₹)",
                    yaxis_title="Expected Volume (Liters)",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                # Detailed data table
                st.dataframe(
                    results_df.sort_values("price").style.format({
                        'price': '₹{:.2f}',
                        'expected_volume': '{:,.2f}',
                        'expected_profit': '₹{:,.2f}',
                        'margin': '₹{:.2f}',
                        'price_gap': '{:+.2f}'
                    }).highlight_max(subset=['expected_profit'], color='lightgreen'),
                    use_container_width=True
                )

            # Key insights
            st.subheader("🔍 Key Insights")
            comp_avg = (comp1 + comp2 + comp3) / 3
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Market Position:**")
                st.write(f"- Competitor Average: ₹{comp_avg:.2f}")
                st.write(f"- Recommended Gap: ₹{best['price_gap']:.2f}")
                st.write(f"- Price Change: {((best['price'] - current_price) / current_price * 100):+.1f}%")
            
            with col2:
                st.write("**Profitability:**")
                st.write(f"- Margin per Liter: ₹{best['margin']:.2f}")
                current_profit = (current_price - cost) * results_df[results_df['price'] == current_price]['expected_volume'].values[0] if current_price in results_df['price'].values else 0
                profit_improvement = ((best['expected_profit'] - current_profit) / current_profit * 100) if current_profit > 0 else 0
                st.write(f"- Profit Improvement: {profit_improvement:+.1f}%")
                st.write(f"- Total Options Evaluated: {len(results_df)}")

            # Export option
            st.divider()
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"price_optimization_{date}.csv",
                mime="text/csv"
            )

# Footer
st.divider()
st.caption("💡 This tool uses machine learning to predict demand and optimize pricing. Always review recommendations with business context.")