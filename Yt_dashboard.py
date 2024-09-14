import streamlit as st
import base64
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dateutil import parser
import matplotlib.pyplot as plt
import pydeck as pdk
import seaborn as sns

# Function to load an image and convert it to a base64 string
def load_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode()
    return base64_image


logo_path = "Youtube_logo.png"  
stat_path = "stat.png"  


views_path = "img/view.png"  
comment_path = "img/comments.jpeg" 
likes_path = "img/likes.png" 
shares_path = "img/shares.png" 
sub_path = "img/sub.png" 


# Load the image as base64
logo_base64 = load_image_as_base64(logo_path)
stat_base64 = load_image_as_base64(stat_path)
view_base64 = load_image_as_base64(views_path)
comment_base64 = load_image_as_base64(comment_path)
likes_base64 = load_image_as_base64(likes_path)
shares_base64 = load_image_as_base64(shares_path)
sub_base64 = load_image_as_base64(sub_path)

## Apply styling

def style_negative(v, props=''):
    """ Style negative values in dataframe"""
    try: 
        return props if v < 0 else None
    except:
        pass
    
def style_positive(v, props=''):
    """Style positive values in dataframe"""
    try: 
        return props if v > 0 else None
    except:
        pass    

def audience_simple(country):
    """Show top represented countries"""
    if country == 'US':
        return 'USA'
    elif country == 'TUN':
        return 'Tunisia'
    elif country == 'IN':
        return 'India'
    elif country == 'FR':
        return 'France'
    else:
        return 'Other' 

# data load

def load_data():
    # Load and clean Aggregated_Metrics_By_Video.csv
    df_agg = pd.read_csv('data/Aggregated_Metrics_By_Video.csv').iloc[1:, :]
    df_agg.columns = ['Video', 'Video title', 'Video publish time', 'Comments added', 'Shares', 'Dislikes', 'Likes',
                      'Subscribers lost', 'Subscribers gained', 'RPM(USD)', 'CPM(USD)', 'Average % viewed', 'Average view duration',
                      'Views', 'Watch time (hours)', 'Subscribers', 'Your estimated revenue (USD)', 'Impressions', 'Impressions ctr(%)']
    
    # Parse Video publish time using dateutil.parser for flexibility
    df_agg['Video publish time'] = df_agg['Video publish time'].apply(lambda x: parser.parse(x) if pd.notnull(x) else None)
    
    # Parse average view duration
    df_agg['Average view duration'] = df_agg['Average view duration'].apply(lambda x: datetime.strptime(str(x), '%H:%M:%S') if pd.notnull(x) else None)
    
    # Calculate average duration in seconds
    df_agg['Avg_duration_sec'] = df_agg['Average view duration'].apply(lambda x: x.second + x.minute * 60 + x.hour * 3600 if pd.notnull(x) else None)
    
    # Calculate engagement ratio
    df_agg['Engagement_ratio'] = (df_agg['Comments added'] + df_agg['Shares'] + df_agg['Dislikes'] + df_agg['Likes']) / df_agg['Views']
    
    # Calculate views per subscriber gained
    df_agg['Views / sub gained'] = df_agg['Views'] / df_agg['Subscribers gained']
    
    # Sort by Video publish time
    df_agg.sort_values('Video publish time', ascending=False, inplace=True)
    
    # Load other datasets
    df_agg_sub = pd.read_csv('data/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    df_comments = pd.read_csv('data/Aggregated_Metrics_By_Video.csv')
    df_time = pd.read_csv('data/Video_Performance_Over_Time.csv')
    
    # Parse dates in df_time using dateutil.parser for flexibility
    df_time['Date'] = df_time['Date'].apply(lambda x: parser.parse(x) if pd.notnull(x) else None)
    
    return df_agg, df_agg_sub, df_comments, df_time

# Load the data
df_agg, df_agg_sub, df_comments, df_time = load_data()

# Define df_time_diff correctly
df_time_diff = pd.merge(df_time, df_agg.loc[:,['Video','Video publish time']], left_on ='External Video ID', right_on = 'Video')
df_time_diff['days_published'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days

# Get last 12 months of data rather than all data
date_12mo = df_agg['Video publish time'].max() - pd.DateOffset(months =12)
df_time_diff_yr = df_time_diff[df_time_diff['Video publish time'] >= date_12mo]

# Get daily view data (first 30), median & percentiles
views_days = pd.pivot_table(df_time_diff_yr, index='days_published', values='Views', aggfunc=[np.mean, np.median, lambda x: np.percentile(x, 80), lambda x: np.percentile(x, 20)]).reset_index()
views_days.columns = ['days_published', 'mean_views', 'median_views', '80pct_views', '20pct_views']
views_days = views_days[views_days['days_published'].between(0, 30)]
views_cumulative = views_days.loc[:, ['days_published', 'median_views', '80pct_views', '20pct_views']]
views_cumulative.loc[:, ['median_views', '80pct_views', '20pct_views']] = views_cumulative.loc[:, ['median_views', '80pct_views', '20pct_views']].cumsum()

# Streamlit sidebar with custom title and styling
st.sidebar.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center;">
        <img src="data:image/png;base64,{stat_base64}" alt="YouTube stat" style="width:50px;height:50px;margin-right:15px;">
        <h1 style="display:inline;">YouTube Statistics</h1>
    </div>
    """, unsafe_allow_html=True)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
add_sidebar = st.sidebar.radio('Navigation', ('Aggregate Metrics', 'Individual Video Analysis'))

if add_sidebar == 'Aggregate Metrics':
    st.markdown(f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" alt="YouTube Logo" style="width:50px;height:50px;margin-right:15px;">
        <h1 style="display:inline;">YouTube Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Key Metrics")

    df_agg_metrics = df_agg[['Video publish time', 'Views', 'Likes', 'Subscribers', 'Shares', 'Comments added', 'RPM(USD)', 'Average % viewed',
                             'Avg_duration_sec', 'Engagement_ratio', 'Views / sub gained']]

    metric_date_6mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=6)
    metric_date_12mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=12)

    # Filter only numeric columns
    numeric_columns = df_agg_metrics.select_dtypes(include='number').columns

    metric_medians6mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_6mo][numeric_columns].median()
    metric_medians12mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_12mo][numeric_columns].median()

    # Calculate deltas
    deltas = {}
    for i in metric_medians6mo.index:
        if i in metric_medians12mo.index:
            try:
                delta = (metric_medians6mo[i] - metric_medians12mo[i]) / metric_medians12mo[i]
                deltas[i] = delta
            except TypeError as e:
                deltas[i] = None

    # Round the values
    metric_medians6mo = metric_medians6mo.round(2)
    metric_medians12mo = metric_medians12mo.round(2)
    deltas = {k: round(v, 2) if v is not None else v for k, v in deltas.items()}

    # Display the metrics using columns
    columns = st.columns(5)
    count = 0
    for i in metric_medians6mo.index:
        with columns[count]:
            delta = (metric_medians6mo[i] - metric_medians12mo[i]) / metric_medians12mo[i]
            st.metric(label=i, value=round(metric_medians6mo[i], 1), delta="{:.2%}".format(delta))
            count += 1
            if count >= 5:
                count = 0

    # Get date information / trim to relevant data
    df_agg_diff = df_agg.copy()
    df_agg_diff['Publish_date'] = df_agg_diff['Video publish time'].apply(lambda x: x.date())
    df_agg_diff_final = df_agg_diff.loc[:, ['Video title', 'Publish_date', 'Views', 'Likes', 'Subscribers', 'Shares', 'Comments added', 'RPM(USD)', 'Average % viewed',
                                            'Avg_duration_sec', 'Engagement_ratio', 'Views / sub gained']]
 
    # Apply styling
    numeric_columns_final = df_agg_diff_final.select_dtypes(include='number').columns
    df_to_pct = {col: '{:.1%}'.format for col in ['Average % viewed', 'Engagement_ratio']}
    df_to_round = {col: '{:.2f}'.format for col in numeric_columns_final if col not in df_to_pct}

    st.dataframe(df_agg_diff_final.style.applymap(style_negative, subset=numeric_columns_final, props='color:red;')
                                     .applymap(style_positive, subset=numeric_columns_final, props='color:green;')
                                     .format(df_to_round)
                                     .format(df_to_pct))
    

    
    # CSS for hover effect
    st.markdown("""
    <style>
    .video-box {
        border: 1px solid #FE0000;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
        text-align: center;
        transition: transform 0.2s; /* Animation */
        cursor: pointer;
    }

    .video-box:hover {
        transform: scale(1.05); /* Scale up the box */
    }
    </style>
    """, unsafe_allow_html=True)
        # Top 3 videos based on views from the last 12 months
    date_12mo = df_agg['Video publish time'].max() - pd.DateOffset(months=12)
    df_agg_last_12mo = df_agg[df_agg['Video publish time'] >= date_12mo]
    st.markdown("### Top 3 Videos (Last 12 Months)")
    top_videos_last_12mo = df_agg_last_12mo.nlargest(3, 'Views')[['Video title', 'Video publish time', 'Views', 'Likes', 'Engagement_ratio']]
   
    # Using columns to ensure horizontal layout
    columns = st.columns(3)
    for idx, (col, row) in enumerate(zip(columns, top_videos_last_12mo.itertuples()), 1):
        with col:
            show_details = st.checkbox(f"Show details for Top {idx}", key=f"show_details_{idx}")
            if show_details:
                st.markdown(f"""
    <div class="video-box">
        <h3 style="color: #333;">Top {idx}</h3>
        <p style="font-size: 16px;">Publish Date: {row._2}</p>
        <p style="font-size: 16px;">Views: {row.Views}</p>
        <p style="font-size: 16px;">Likes: {row.Likes}</p>
        <p style="font-size: 16px;">Engagement Ratio: {row.Engagement_ratio:.2f}</p>
    </div>
    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="video-box">
                    <h3>Top {idx}</h3>
                    <h4>{row._1}</h4>
                </div>
                """, unsafe_allow_html=True)
          
    ### Views Over Time
    st.markdown("### Views Over Time")
    views_over_time = df_agg.groupby('Video publish time')['Views'].sum()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(views_over_time.index, views_over_time.values, color='#FF0000', marker='o', linewidth=2)
    
    ax.set_xlabel('Date', fontsize=14, color='black')
    ax.set_ylabel('Views', fontsize=14, color='black')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
   
    st.pyplot(fig)

    ### Engagement Metrics Distribution
    youtube_colors = ['#FE0000', '#f7f7f7', '#FFFAE7', '#EFEFEF', '#EA906C']
    st.markdown("### Engagement Metrics Distribution")
    engagement_metrics = df_agg[['Likes', 'Comments added', 'Shares']].sum()
    fig, ax = plt.subplots()
    ax.pie(engagement_metrics, labels=engagement_metrics.index, autopct='%1.1f%%', startangle=90, colors=youtube_colors)
    ax.axis('equal')
    st.pyplot(fig)
    
    st.markdown("### Views by Location")
    country_coords = {
    'US': {'lat': 37.0902, 'lon': -95.7129},
    'TUN': {'lat': 33.8869, 'lon': 9.5375},
    'IN': {'lat': 20.5937, 'lon': 78.9629},
    'FR': {'lat': 46.6034, 'lon': 1.8883},
    # Add other countries as needed
}
    data = pd.read_csv('data/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')  
    views_by_country = data.groupby('Country Code')['Views'].sum().reset_index()
    views_by_country['lat'] = views_by_country['Country Code'].apply(lambda x: country_coords.get(x, {}).get('lat'))
    views_by_country['lon'] = views_by_country['Country Code'].apply(lambda x: country_coords.get(x, {}).get('lon'))
    views_by_country = views_by_country.dropna(subset=['lat', 'lon'])
  
    layer = pdk.Layer(
    'ScatterplotLayer',
    views_by_country,
    get_position=['lon', 'lat'],
    get_color='[200, 30, 0, 160]',
    get_radius='Views',
    pickable=True
)
    view_state = pdk.ViewState(
    latitude=views_by_country['lat'].mean(),
    longitude=views_by_country['lon'].mean(),
    zoom=2,
    pitch=0
)
    r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "{Views} views"}
)
    st.pydeck_chart(r)
    
 
                

    
if add_sidebar == 'Individual Video Analysis':
    # Title with black color and centered text
    st.markdown("<h2 style='text-align: center; color: black;'>Individual Video Performance</h2>", unsafe_allow_html=True)
    
    # Create a select box for choosing a video
    videos = tuple(df_agg['Video title'])  # Create a tuple from the video titles
    video_select = st.selectbox('Pick a Video:', videos)  # Select box for video selection

    # Filter data for the selected video
    agg_filtered = df_agg[df_agg['Video title'] == video_select]  # Filter based on selected video title
    agg_sub_filtered = df_agg_sub[df_agg_sub['Video Title'] == video_select]  # Filter for subscriber data
    
    # Apply transformation and sorting to the subscriber data
    agg_sub_filtered['Country'] = agg_sub_filtered['Country Code'].apply(audience_simple)  # Map country codes to audience
    agg_sub_filtered.sort_values('Is Subscribed', inplace=True)  # Sort by subscription status

    # Plot subscriber data by country using a horizontal bar chart
    fig = px.bar(agg_sub_filtered, x='Views', y='Is Subscribed', color='Country', orientation='h')  # Horizontal bar chart
    st.plotly_chart(fig)  # Display the chart

    # Filter and sort time series data for the selected video
    agg_time_filtered = df_time_diff[df_time_diff['Video Title'] == video_select]  # Filter by selected video
    first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0, 30)]  # Filter for first 30 days
    first_30 = first_30.sort_values('days_published')  # Sort by days since published

    # Initialize the second figure for view comparison
    fig2 = go.Figure()

    # Add percentile traces for 20th, 50th, and 80th percentiles
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['20pct_views'],
                              mode='lines', name='20th percentile', line=dict(color='purple', dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['median_views'],
                              mode='lines', name='50th percentile', line=dict(color='black', dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['80pct_views'],
                              mode='lines', name='80th percentile', line=dict(color='royalblue', dash='dash')))

    # Add the current video trace for the first 30 days
    fig2.add_trace(go.Scatter(x=first_30['days_published'], y=first_30['Views'].cumsum(),
                              mode='lines', name='Current Video', line=dict(color='firebrick', width=8)))

    # Update layout for the second figure
    fig2.update_layout(title='View Comparison - First 30 Days',
                       xaxis_title='Days Since Published',
                       yaxis_title='Cumulative Views')

    # Display the second chart
    st.plotly_chart(fig2)
