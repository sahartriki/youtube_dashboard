
# YouTube Dashboard

This project provides an interactive dashboard using **Streamlit** that helps analyze YouTube metrics such as views, likes, comments, subscribers, and engagement. It processes and visualizes aggregated data, enabling content creators to better understand the performance of their videos.

## Table of Contents
- [Features](#features)
- [Technologies](#technologies)
- [Setup](#setup)
- [Usage](#usage)
- [Dataset](#dataset)
- [Contributing](#contributing)

## Features
- **Aggregate Metrics**: Display aggregated video data like views, likes, engagement ratio, etc., for the last 6 and 12 months.
- **Video Performance**: Analyze key performance indicators (KPIs) for individual videos.
- **Top Videos**: Display top 3 videos based on views for the last 12 months.
- **Visualizations**: Show views, likes, and engagement trends over time using Plotly and Seaborn.

## Technologies
This project uses the following Python libraries:
- **Streamlit**: for building interactive web apps.
- **Pandas**: for data manipulation and analysis.
- **Plotly**: for interactive visualizations.
- **Matplotlib & Seaborn**: for plotting graphs.
- **Pydeck**: for geospatial visualization.
- **Dateutil**: for flexible date parsing.
- **Base64**: for encoding images to display in the app.

## Setup
To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/youtube_dashboard.git
   cd youtube_dashboard
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

5. **Add your data**: Make sure to add your YouTube dataset CSV files (such as `Aggregated_Metrics_By_Video.csv`, etc.) in the `data` folder.

## Usage
Once the app is running, you'll have access to two main sections:

### 1. **Aggregate Metrics**
   - Displays overall video performance based on views, likes, comments, engagement ratios, and other metrics.
   - Key metrics comparison between the last 6 and 12 months.

### 2. **Individual Video Analysis**
   - Detailed analysis of individual video performance.
   - Displays top videos over the past year, along with engagement data.

## Dataset
The project assumes access to data files from YouTube's analytics exports:
- `Aggregated_Metrics_By_Video.csv`
- `Aggregated_Metrics_By_Country_And_Subscriber_Status.csv`
- `Video_Performance_Over_Time.csv`

Make sure these files are placed in a `data/` folder in the root directory.

## Contributing
If you'd like to contribute to the project:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request.

## Demo : Some Examples

![Demo Picture](https://github.com/sahartriki/youtube_dashboard/blob/master/img/1.png)
![Demo Picture](https://github.com/sahartriki/youtube_dashboard/blob/master/img/2.png)
![Demo Picture](https://github.com/sahartriki/youtube_dashboard/blob/master/img/3.png)

