import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="AI Inventory Forecaster", layout="centered")
st.title("📦 AI Inventory Demand Forecaster")

# Upload file
uploaded_file = st.file_uploader("📤 Upload your sales data (CSV)", type="csv")

# Forecast duration
forecast_days = st.slider("🗓️ Select forecast duration (days)", min_value=30, max_value=365, value=90, step=30)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Ensure required columns
        required_cols = ['Order Date', 'Sales', 'Sub-Category']
        if not all(col in df.columns for col in required_cols):
            st.error("❌ CSV must contain: 'Order Date', 'Sales', and 'Sub-Category'")
        else:
            df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

            # Select product sub-category
            subcategories = df['Sub-Category'].unique()
            selected_subcat = st.selectbox("📂 Choose a Product Sub-Category", subcategories)

            # Filter and aggregate data
            filtered_df = df[df['Sub-Category'] == selected_subcat]
            daily_sales = filtered_df.groupby('Order Date').agg({'Sales': 'sum'}).reset_index()
            daily_sales = daily_sales.rename(columns={'Order Date': 'ds', 'Sales': 'y'})

            if daily_sales.shape[0] < 30:
                st.warning("⚠️ Not enough data for this sub-category to make a reliable forecast.")
            else:
                # Train Prophet
                model = Prophet()
                model.fit(daily_sales)

                future = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)

                # Add alert column
                def get_alert(yhat):
                    if yhat >= 500:
                        return "⚠️ High Demand - Restock"
                    elif yhat <= 100:
                        return "🟢 Low Demand"
                    else:
                        return "✅ Normal"
                forecast['Alert'] = forecast['yhat'].apply(get_alert)

                # Main forecast chart
                st.subheader(f"📈 Forecasted Sales for {selected_subcat}")
                last_train_date = daily_sales['ds'].max()
                only_future = st.checkbox("🧪 Show only future forecast (hide past data)", value=False)

                if only_future:
                    forecast_plot_data = forecast[forecast['ds'] > last_train_date]
                else:
                    forecast_plot_data = forecast

                # Plot main chart
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(forecast_plot_data['ds'], forecast_plot_data['yhat'], label='yhat (prediction)', color='blue')
                ax.fill_between(forecast_plot_data['ds'],
                                forecast_plot_data['yhat_lower'],
                                forecast_plot_data['yhat_upper'],
                                color='skyblue', alpha=0.3, label='Confidence Interval')

                if not only_future:
                    ax.axvline(last_train_date, color='gray', linestyle='--', label='Forecast Starts')

                ax.set_ylabel("Sales")
                ax.set_xlabel("Date")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                # Actual vs predicted overlay
                if st.checkbox("📊 Show actual past sales", value=True):
                    combined = pd.merge(forecast[['ds', 'yhat']], daily_sales, on='ds', how='left')
                    st.line_chart(combined.set_index('ds'))

                # Alert Table
                st.subheader("🔍 Forecast with Alerts")
                st.dataframe(forecast[['ds', 'yhat', 'Alert']].tail(forecast_days))

                # CSV Download
                csv = forecast.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Forecast CSV", csv, "AI_Inventory_Forecast.csv", "text/csv")


    except Exception as e:
        st.error(f"An error occurred: {e}")
