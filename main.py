import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError, ResponseError
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

def main():
    st.title('Google Search Analysis')
    # Set up Google Trends for keywords
    trends = TrendReq(hl='en-US', tz=360)

    # Input keywords using a text input widget
    keywords_input = st.text_input("Enter keywords (comma-separated)", value='Robotics')
    if keywords_input:
        kw_list = [kw.strip() for kw in keywords_input.split(',')]

        # Handle rate limiting and timeout errors
        try:
            trends.build_payload(kw_list=kw_list)
            data_over_time = trends.interest_over_time()
        except TooManyRequestsError:
            st.error("Too many requests. Waiting before retrying...")
            time.sleep(60)  # Wait for 60 seconds before retrying
            trends.build_payload(kw_list=kw_list)
            data_over_time = trends.interest_over_time()
        except ResponseError as e:
            st.error(f"Error fetching data from Google Trends: {e}")
            return  # Exit the function if there's an error

        # Total Google Searches over Time
        st.subheader('Total Google Searches over Time')
        st.line_chart(data_over_time)

        # Moving Average Analysis
        moving_avg_window = st.number_input("Moving Average Window", min_value=1, max_value=len(data_over_time),
                                            value=30)
        moving_avg = data_over_time.rolling(window=moving_avg_window).mean()
        st.subheader(f'Moving Average (Window: {moving_avg_window} days)')
        st.line_chart(moving_avg)

        # Seasonal Decomposition
        seasonal_decomposition = st.checkbox("Seasonal Decomposition")
        if seasonal_decomposition:
            st.subheader('Seasonal Decomposition')
            for kw in kw_list:
                # Extract the search interest column from data_over_time
                search_interest = data_over_time[kw]
                decomposition = seasonal_decompose(search_interest, model='additive', period=12)
                st.write(f'Keyword: {kw}')
                st.line_chart(decomposition.trend)
                st.line_chart(decomposition.seasonal)
                st.line_chart(decomposition.resid)

        # Autocorrelation Plot
        st.subheader('Autocorrelation Plot')
        for kw in kw_list:
            fig, ax = plt.subplots()
            plot_acf(data_over_time[kw], lags=20, ax=ax)
            st.pyplot(fig)

        # Partial Autocorrelation Plot
        st.subheader('Partial Autocorrelation Plot')
        for kw in kw_list:
            fig, ax = plt.subplots()
            plot_pacf(data_over_time[kw], lags=20, ax=ax)
            st.pyplot(fig)

        # Histogram of Search Interest
        st.subheader('Histogram of Search Interest')
        for kw in kw_list:
            st.subheader(f'Keyword: {kw}')
            st.bar_chart(data_over_time[kw])

        # Boxplot by Month
        data_over_time['Month'] = data_over_time.index.month
        st.subheader('Boxplot of Search Interest by Month')
        st.bar_chart(data_over_time.groupby('Month').mean())

        # Trend Comparison
        st.subheader('Trend Comparison')
        st.line_chart(data_over_time[kw_list])

        # Correlation Matrix
        st.subheader('Correlation Matrix')
        st.write(data_over_time[kw_list].corr())

        # Seasonal Trend Analysis
        st.subheader('Seasonal Trend Analysis')
        for kw in kw_list:
            seasonal_trend = data_over_time[kw].groupby(data_over_time.index.month).mean()
            st.write(f'Keyword: {kw}')
            st.line_chart(seasonal_trend)

        # Histogram of Seasonal Trends
        st.subheader('Histogram of Seasonal Trends')
        for kw in kw_list:
            seasonal_trend = data_over_time[kw].groupby(data_over_time.index.month).mean()
            st.write(f'Keyword: {kw}')
            st.bar_chart(seasonal_trend)

        # Heatmap of Search Interest
        st.subheader('Heatmap of Search Interest')
        st.write(data_over_time[kw_list].corr().style.background_gradient(cmap='viridis'))

if __name__ == "__main__":
    main()
