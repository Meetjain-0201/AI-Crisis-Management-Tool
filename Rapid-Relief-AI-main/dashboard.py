import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from datetime import datetime, timedelta
import time

try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    db = client["resource_allocation"]
    synthetic_collection = db["gan_data"]  # Ensure this matches your collection name
    print("Connected to MongoDB successfully!")
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}")
    synthetic_collection = None  # Set to None if connection fails

# City coordinates
CITY_COORDINATES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946}
}

# Map styles
MAP_STYLES = {
    "Basic": "carto-positron",
    "Dark": "carto-darkmatter",
    "Streets": "open-street-map",
    "Satellite": "white-bg"
}

# Color mappings for legend
COLOR_MAPPINGS = {
    'severity': {
        'red': 'Critical (>70)',
        'orange': 'High (50-70)',
        'yellow': 'Moderate (30-50)',
        'green': 'Low (<30)'
    },
    'food': {
        'red': 'Critical (<2 days)',
        'orange': 'Low (2-4 days)',
        'green': 'Adequate (>4 days)'
    },
    'medical': {
        'red': 'Critical (<2 days)',
        'orange': 'Low (2-4 days)',
        'green': 'Adequate (>4 days)'
    },
    'water': {
        'red': 'Critical (<2 days)',
        'orange': 'Low (2-4 days)',
        'green': 'Adequate (>4 days)'
    },
    'roads': {
        'red': 'Severe (>3 blocks)',
        'orange': 'Moderate (2-3 blocks)',
        'green': 'Minor (0-1 blocks)'
    }
}

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'previous_data' not in st.session_state:
    st.session_state.previous_data = None
if 'selected_map_view' not in st.session_state:
    st.session_state.selected_map_view = 'severity'
if 'map_style' not in st.session_state:
    st.session_state.map_style = 'Basic'

def load_data():
    """Load data from MongoDB and validate structure"""
    if synthetic_collection is None:
        st.error("MongoDB connection is not established.")
        return pd.DataFrame()
    
    try:
        data = list(synthetic_collection.find({}, {'_id': 0}))
        df = pd.DataFrame(data)
        
        # Validate required columns
        required_columns = [
            'region_name', 'severity_score', 'road_block_status', 
            'population_density', 'warehouse_stock_status', 'resource_needs'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column in data: {col}")
                return pd.DataFrame()  # Return empty DataFrame if validation fails
        
        return df
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return pd.DataFrame()

def get_marker_properties(row, view_type):
    """Get marker properties based on selected view"""
    if view_type == 'severity':
        score = row['severity_score']
        if score > 70:
            return 'rgba(255,0,0,0.6)', f"Severity: {score:.1f}"
        elif score > 50:
            return 'rgba(255,165,0,0.6)', f"Severity: {score:.1f}"
        elif score > 30:
            return 'rgba(255,255,0,0.6)', f"Severity: {score:.1f}"
        return 'rgba(0,255,0,0.6)', f"Severity: {score:.1f}"
    
    elif view_type in ['food', 'medical', 'water']:
        stock = row['warehouse_stock_status'][view_type]
        need = row['resource_needs'][view_type]
        days_left = stock / need if need > 0 else float('inf')
        if days_left < 2:
            return 'rgba(255,0,0,0.6)', f"{view_type.capitalize()}: {days_left:.1f} days left"
        elif days_left < 4:
            return 'rgba(255,165,0,0.6)', f"{view_type.capitalize()}: {days_left:.1f} days left"
        return 'rgba(0,255,0,0.6)', f"{view_type.capitalize()}: {days_left:.1f} days left"
    
    elif view_type == 'roads':
        blocks = row['road_block_status']
        if blocks > 3:
            return 'rgba(255,0,0,0.6)', f"Blocked roads: {blocks}"
        elif blocks > 1:
            return 'rgba(255,165,0,0.6)', f"Blocked roads: {blocks}"
        return 'rgba(0,255,0,0.6)', f"Blocked roads: {blocks}"

# Added caching and error handling
@st.cache_data(ttl=10, show_spinner=False)
def load_data():
    try:
        data = list(synthetic_collection.find({}, {'_id': 0}).sort('timestamp', -1).limit(5))
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return pd.DataFrame()

# Added time range selector
def time_filtered_data(df):
    default_end = datetime.now()
    default_start = default_end - timedelta(hours=24)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", default_start)
        start_time = st.time_input("Start time", default_start.time())
    with col2:
        end_date = st.date_input("End date", default_end)
        end_time = st.time_input("End time", default_end.time())
    
    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)
    
    return df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
    
# Add this function at the top of the file, after the imports
def load_historical_data(hours=24):
    """Load historical data from MongoDB for the specified time window"""
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Query for historical data
        history_data = list(synthetic_collection.find({
            "timestamp": {
                "$gte": start_time,
                "$lte": end_time
            }
        }, {'_id': 0}))
        
        # Convert to DataFrame and add coordinates
        df = pd.DataFrame(history_data)
        if not df.empty:
            df['lat'] = df['region_name'].map(lambda x: CITY_COORDINATES[x]['lat'])
            df['lon'] = df['region_name'].map(lambda x: CITY_COORDINATES[x]['lon'])
        
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return pd.DataFrame()

def create_map(df):
    """Create an interactive map with resource status indicators"""
    if df.empty or 'region_name' not in df.columns:
        st.warning("No valid data available for map visualization.")
        return go.Figure()  # Return an empty figure
    
    fig = go.Figure()
    
    # Add historical trajectory layer if data exists
    try:
        history_df = load_historical_data(hours=6)  # Last 6 hours of data
        if not history_df.empty and 'region_name' in history_df.columns:
            for city in df['region_name'].unique():
                city_df = history_df[history_df['region_name'] == city]
                if not city_df.empty:
                    fig.add_trace(go.Scattermapbox(
                        lat=city_df['lat'],
                        lon=city_df['lon'],
                        mode='lines+markers',
                        marker=dict(size=8, color='rgba(100,100,100,0.5)'),
                        line=dict(width=1, color='gray'),
                        name=f"{city} Trend",
                        hoverinfo='none',
                        showlegend=True
                    ))
    except Exception as e:
        st.warning(f"Could not add historical data: {e}")
    
    # Add legend traces
    legend_colors = COLOR_MAPPINGS[st.session_state.selected_map_view]
    for color, label in legend_colors.items():
        rgba_color = color.replace('red', 'rgba(255,0,0,0.6)')\
                         .replace('orange', 'rgba(255,165,0,0.6)')\
                         .replace('yellow', 'rgba(255,255,0,0.6)')\
                         .replace('green', 'rgba(0,255,0,0.6)')
        
        fig.add_trace(go.Scattermapbox(
            lat=[None],
            lon=[None],
            mode='markers',
            marker=dict(size=20, color=rgba_color),
            name=label,
            showlegend=True
        ))
    
    # Add city markers
    for _, row in df.iterrows():
        if 'region_name' not in row or row['region_name'] not in CITY_COORDINATES:
            continue  # Skip invalid rows
        
        city_name = row['region_name']
        coords = CITY_COORDINATES[city_name]
        
        color, status_text = get_marker_properties(row, st.session_state.selected_map_view)
        
        hover_text = f"<b>{city_name}</b><br>{status_text}<br>"
        
        fig.add_trace(go.Scattermapbox(
            lat=[coords['lat']],
            lon=[coords['lon']],
            mode='markers',
            marker=dict(
                size=20,
                color=color,
                showscale=False
            ),
            text=[hover_text],
            hoverinfo='text',
            hoverlabel=dict(
                bgcolor='white',
                font=dict(color='black')
            ),
            name=city_name,
            showlegend=False
        ))
    
    fig.update_layout(
        mapbox=dict(
            style=MAP_STYLES[st.session_state.map_style],
            center=dict(lat=20.5937, lon=78.9629),  # Center of India
            zoom=4
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            font=dict(color='black'),
            x=0,
            y=1
        )
    )
    
    return fig

def create_resource_chart(df):
    """Create resource comparison chart"""
    # Prepare data for the chart
    chart_data = []
    
    for _, row in df.iterrows():
        region = row['region_name']
        stocks = row['warehouse_stock_status']
        needs = row['resource_needs']
        
        for resource in stocks.keys():
            chart_data.extend([
                {'Region': region, 'Type': f'Available, {resource}', 'Units': stocks[resource]},
                {'Region': region, 'Type': f'Needed, {resource}', 'Units': needs[resource]}
            ])
    
    chart_df = pd.DataFrame(chart_data)
    
    fig = px.bar(
        chart_df,
        x='Region',
        y='Units',
        color='Type',
        barmode='group',
        title='Resource Availability vs Needs'
    )
    
    return fig

def calculate_resource_recommendations(df):
    """Calculate resource allocation recommendations"""
    recommendations = []
    
    for _, row in df.iterrows():
        urgent_resources = []
        stocks = row['warehouse_stock_status']
        needs = row['resource_needs']
        
        for resource in stocks.keys():
            days_left = stocks[resource] / needs[resource] if needs[resource] > 0 else float('inf')
            if days_left < 3:
                urgent_resources.append(
                    f"{resource} (Stock: {stocks[resource]}, "
                    f"Need: {needs[resource]}, "
                    f"{days_left:.1f} days left)"
                )
        
        priority = "CRITICAL" if row['severity_score'] > 70 else \
                  "HIGH" if row['severity_score'] > 50 else \
                  "MODERATE" if row['severity_score'] > 30 else "LOW"
        
        action = "Urgent attention needed" if priority in ["CRITICAL", "HIGH"] else \
                "Monitor closely" if priority == "MODERATE" else "Situation stable"
        
        recommendations.append({
            "region": row['region_name'],
            "priority": priority,
            "action": action,
            "urgent_resources": urgent_resources
        })
    
    return sorted(recommendations, key=lambda x: ["LOW", "MODERATE", "HIGH", "CRITICAL"].index(x["priority"]))

def main():
    st.set_page_config(layout="wide", page_title="Real-time Resource Allocation Dashboard")
    
    st.title("🌐 Real-time Disaster Resource Management Dashboard")
    
    # Add map controls in the sidebar
    st.sidebar.header("Map Controls")
    st.session_state.map_style = st.sidebar.selectbox(
        "Select Map Style",
        options=list(MAP_STYLES.keys())
    )
    
    st.session_state.selected_map_view = st.sidebar.radio(
        "Select Map View",
        options=['severity', 'food', 'water', 'medical', 'roads'],
        format_func=lambda x: x.capitalize()
    )
    
    # Create a container for the main content
    main_container = st.container()
    
    with main_container:
        # Load new data
        df = load_data()
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Regions Monitored", len(df))
        with col2:
            current_severity = df['severity_score'].mean()
            if st.session_state.previous_data is not None:
                previous_severity = st.session_state.previous_data['severity_score'].mean()
                delta = current_severity - previous_severity
            else:
                delta = None
            st.metric("Average Severity", f"{current_severity:.1f}", 
                      delta=f"{delta:.1f}" if delta else None)
        with col3:
            blocked_roads = df['road_block_status'].sum()
            st.metric("Blocked Roads", blocked_roads)
        with col4:
            total_population = df['population_density'].sum()
            st.metric("Total Population", f"{total_population:,}")

        # Add the map
        st.subheader(f"📍 Real-time {st.session_state.selected_map_view.capitalize()} Status Map")
        fig_map = create_map(df)
        st.plotly_chart(fig_map, use_container_width=True)

        # Create two columns for main visualizations
        col_left, col_right = st.columns([2, 1])

        with col_left:
            # Severity Chart
            fig_severity = px.bar(
                df,
                x='region_name',
                y='severity_score',
                color='severity_score',
                color_continuous_scale='RdYlGn_r',
                title='Regional Severity Scores'
            )
            st.plotly_chart(fig_severity, use_container_width=True)

            # Resource Status
            fig_resources = create_resource_chart(df)
            st.plotly_chart(fig_resources, use_container_width=True)

        with col_right:
            # Critical Recommendations
            st.subheader("📊 Situation Analysis")
            recommendations = calculate_resource_recommendations(df)
            
            for rec in recommendations:
                color = {
                    "CRITICAL": "red",
                    "HIGH": "orange",
                    "MODERATE": "blue",
                    "LOW": "green"
                }[rec["priority"]]
                
                st.markdown(f"""
                <div style='padding: 10px; border-left: 5px solid {color}; margin-bottom: 10px;'>
                    <h4 style='color: {color};'>{rec['region']}</h4>
                    <p><strong>Priority:</strong> {rec['priority']}</p>
                    <p><strong>Action:</strong> {rec['action']}</p>
                    <p><strong>Urgent Resources:</strong> {', '.join(rec['urgent_resources']) if rec['urgent_resources'] else 'None'}</p>
                </div>
                """, unsafe_allow_html=True)

        # Add timestamp
        st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Update previous data
        st.session_state.previous_data = df

        # Trigger a rerun every 3 seconds
        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    main()
