import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class TemporalPatternMiner:
    """
    Novel implementation for temporal pattern mining in social networks.
    Identifies recurring patterns and predicts optimal dissemination times.
    """
    
    def __init__(self, temporal_graphs, interactions_data):
        self.temporal_graphs = temporal_graphs
        self.interactions_data = interactions_data
        self.hourly_patterns = {}
        self.daily_patterns = {}
        self.network_states = {}
        self.pattern_clusters = {}
        
    def _parse_timestamp(self, timestamp):
        """Safely parse various timestamp formats including your specific format"""
        if isinstance(timestamp, datetime):
            return timestamp
        
        timestamp_str = str(timestamp).strip()
        
        # Handle your specific format: 2024-11-13_19-21
        if '_' in timestamp_str and timestamp_str.count('-') >= 3:
            try:
                # Split by underscore to get date and time parts
                parts = timestamp_str.split('_')
                if len(parts) == 2:
                    date_part, time_part = parts
                    
                    # Parse date part (YYYY-MM-DD)
                    date_components = date_part.split('-')
                    if len(date_components) == 3:
                        year, month, day = date_components
                        
                        # Parse time part (HH-MM)
                        time_components = time_part.split('-')
                        if len(time_components) == 2:
                            hour, minute = time_components
                            
                            # Create datetime object
                            return datetime(int(year), int(month), int(day), int(hour), int(minute))
            except Exception as e:
                print(f"Error parsing custom format {timestamp_str}: {e}")
        
        # Try standard pandas parsing as fallback
        try:
            return pd.to_datetime(timestamp_str)
        except Exception as e:
            print(f"Warning: Could not parse timestamp: {timestamp_str}, Error: {e}")
            return None
        
    def extract_temporal_features(self):
        """Extract comprehensive temporal features from network snapshots"""
        temporal_features = []
        
        print(f"Processing {len(self.temporal_graphs)} temporal snapshots...")
        
        for timestamp, graph in self.temporal_graphs.items():
            # Parse timestamp safely
            dt = self._parse_timestamp(timestamp)
            if dt is None:
                print(f"Skipping timestamp: {timestamp}")
                continue
                
            # Basic network metrics
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            density = nx.density(graph) if n_nodes > 1 else 0
            
            # Centrality-based features
            if n_nodes > 0:
                try:
                    pagerank = nx.pagerank(graph)
                    centralization = self._calculate_centralization(pagerank)
                    
                    # Degree distribution features
                    degrees = [d for n, d in graph.degree()]
                    degree_variance = np.var(degrees) if degrees else 0
                    degree_skewness = stats.skew(degrees) if len(degrees) > 2 else 0
                    
                    # Clustering features
                    clustering_coeffs = list(nx.clustering(graph).values())
                    avg_clustering = np.mean(clustering_coeffs) if clustering_coeffs else 0
                    
                    # Connectivity features
                    if graph.is_directed():
                        # Convert to undirected for connectivity analysis
                        undirected = graph.to_undirected()
                        n_components = nx.number_connected_components(undirected)
                        largest_component_size = len(max(nx.connected_components(undirected), key=len)) if n_components > 0 else 0
                    else:
                        n_components = nx.number_connected_components(graph)
                        largest_component_size = len(max(nx.connected_components(graph), key=len)) if n_components > 0 else 0
                        
                except Exception as e:
                    print(f"Error calculating network metrics for {timestamp}: {e}")
                    centralization = 0
                    degree_variance = 0
                    degree_skewness = 0
                    avg_clustering = 0
                    n_components = 0
                    largest_component_size = 0
            else:
                centralization = 0
                degree_variance = 0
                degree_skewness = 0
                avg_clustering = 0
                n_components = 0
                largest_component_size = 0
            
            # Temporal context features
            hour = dt.hour
            day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
            is_weekend = day_of_week >= 5
            
            # Activity momentum (compare with previous periods)
            activity_growth = 0
            if len(temporal_features) > 0:
                prev_edges = temporal_features[-1]['n_edges']
                activity_growth = (n_edges - prev_edges) / (prev_edges + 1)
            
            feature_dict = {
                'timestamp': timestamp,
                'datetime': dt,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'density': density,
                'centralization': centralization,
                'degree_variance': degree_variance,
                'degree_skewness': degree_skewness,
                'avg_clustering': avg_clustering,
                'n_components': n_components,
                'largest_component_size': largest_component_size,
                'activity_growth': activity_growth,
                'network_efficiency': largest_component_size / n_nodes if n_nodes > 0 else 0
            }
            
            temporal_features.append(feature_dict)
        
        df = pd.DataFrame(temporal_features)
        
        # Add dissemination score
        if not df.empty:
            # Normalize edges for score calculation
            max_edges = df['n_edges'].max() if df['n_edges'].max() > 0 else 1
            
            df['dissemination_score'] = (
                df['density'] * 0.3 +
                df['avg_clustering'] * 0.25 +
                (1 - df['centralization']) * 0.2 +  # Lower centralization = better spread
                (df['n_edges'] / max_edges) * 0.25  # Normalized activity
            )
        
        return df
    
    def _calculate_centralization(self, centrality_dict):
        """Calculate network centralization index"""
        if not centrality_dict:
            return 0
        
        values = list(centrality_dict.values())
        if len(values) <= 1:
            return 0
            
        max_centrality = max(values)
        sum_diff = sum(max_centrality - c for c in values)
        n = len(values)
        
        # Freeman's centralization formula
        max_possible_sum = (n - 1) * (n - 2) if n > 2 else 1
        return sum_diff / max_possible_sum if max_possible_sum > 0 else 0
    
    def identify_hourly_patterns(self, features_df):
        """Identify recurring hourly patterns in network formation"""
        print("🔄 Analyzing hourly network formation patterns...")
        
        if features_df.empty:
            print("⚠️ No temporal features available")
            return {}
        
        # Group by hour and calculate pattern metrics
        hourly_stats = features_df.groupby('hour').agg({
            'n_nodes': ['mean', 'std'],
            'n_edges': ['mean', 'std'],
            'density': ['mean', 'std'],
            'centralization': ['mean', 'std'],
            'activity_growth': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        hourly_stats.columns = ['_'.join(col) for col in hourly_stats.columns]
        
        # Identify peak activity hours
        peak_activity_metric = hourly_stats['n_edges_mean']
        peak_hours = peak_activity_metric.nlargest(3).index.tolist()
        
        # Identify optimal network formation hours (high density + low centralization)
        formation_score = (hourly_stats['density_mean'] * 0.6 + 
                          (1 - hourly_stats['centralization_mean']) * 0.4)
        optimal_formation_hours = formation_score.nlargest(3).index.tolist()
        
        self.hourly_patterns = {
            'stats': hourly_stats,
            'peak_activity_hours': peak_hours,
            'optimal_formation_hours': optimal_formation_hours,
            'formation_scores': formation_score
        }
        
        return self.hourly_patterns
    
    def identify_daily_patterns(self, features_df):
        """Identify weekly patterns in network behavior"""
        print("🔄 Analyzing daily/weekly network patterns...")
        
        if features_df.empty:
            print("⚠️ No temporal features available")
            return {}
        
        # Group by day of week
        daily_stats = features_df.groupby('day_of_week').agg({
            'n_nodes': ['mean', 'std'],
            'n_edges': ['mean', 'std'],
            'density': ['mean', 'std'],
            'centralization': ['mean', 'std'],
            'avg_clustering': ['mean', 'std']
        }).round(4)
        
        daily_stats.columns = ['_'.join(col) for col in daily_stats.columns]
        
        # Compare weekday vs weekend patterns
        weekday_data = features_df[features_df['is_weekend'] == False]
        weekend_data = features_df[features_df['is_weekend'] == True]
        
        weekday_pattern = {
            'avg_nodes': weekday_data['n_nodes'].mean() if not weekday_data.empty else 0,
            'avg_edges': weekday_data['n_edges'].mean() if not weekday_data.empty else 0,
            'avg_density': weekday_data['density'].mean() if not weekday_data.empty else 0,
            'avg_centralization': weekday_data['centralization'].mean() if not weekday_data.empty else 0
        }
        
        weekend_pattern = {
            'avg_nodes': weekend_data['n_nodes'].mean() if not weekend_data.empty else 0,
            'avg_edges': weekend_data['n_edges'].mean() if not weekend_data.empty else 0,
            'avg_density': weekend_data['density'].mean() if not weekend_data.empty else 0,
            'avg_centralization': weekend_data['centralization'].mean() if not weekend_data.empty else 0
        }
        
        self.daily_patterns = {
            'daily_stats': daily_stats,
            'weekday_pattern': weekday_pattern,
            'weekend_pattern': weekend_pattern,
            'day_names': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        }
        
        return self.daily_patterns
    
    def cluster_network_states(self, features_df, n_clusters=5):
        """Cluster different network states for pattern recognition"""
        print(f"🔄 Clustering network states into {n_clusters} patterns...")
        
        if features_df.empty or len(features_df) < n_clusters:
            print(f"⚠️ Insufficient data for clustering (need at least {n_clusters} samples)")
            return {}
        
        # Select features for clustering
        clustering_features = [
            'density', 'centralization', 'degree_variance', 
            'avg_clustering', 'network_efficiency', 'activity_growth'
        ]
        
        # Prepare data
        cluster_data = features_df[clustering_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(features_df)), random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Analyze clusters
        features_df_copy = features_df.copy()
        features_df_copy['cluster'] = cluster_labels
        
        cluster_analysis = {}
        for cluster_id in range(max(cluster_labels) + 1):
            cluster_mask = features_df_copy['cluster'] == cluster_id
            cluster_data = features_df_copy[cluster_mask]
            
            # Temporal distribution of this cluster
            hourly_dist = cluster_data['hour'].value_counts().sort_index()
            daily_dist = cluster_data['day_of_week'].value_counts().sort_index()
            
            # Characteristic features
            cluster_profile = cluster_data[clustering_features].mean()
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'profile': cluster_profile,
                'hourly_distribution': hourly_dist,
                'daily_distribution': daily_dist,
                'peak_hours': hourly_dist.nlargest(3).index.tolist() if not hourly_dist.empty else [],
                'peak_days': daily_dist.nlargest(3).index.tolist() if not daily_dist.empty else []
            }
        
        self.pattern_clusters = {
            'cluster_labels': cluster_labels,
            'cluster_analysis': cluster_analysis,
            'scaler': scaler,
            'kmeans': kmeans
        }
        
        return self.pattern_clusters
    
    def predict_optimal_dissemination_times(self, features_df):
        """Predict optimal times for information dissemination"""
        print("🔄 Predicting optimal information dissemination times...")
        
        if features_df.empty:
            print("⚠️ No temporal features available")
            return {}
        
        # Find optimal times by hour
        hourly_dissemination = features_df.groupby('hour')['dissemination_score'].agg(['mean', 'std', 'count'])
        optimal_hours = hourly_dissemination['mean'].nlargest(5).index.tolist()
        
        # Find optimal days
        daily_dissemination = features_df.groupby('day_of_week')['dissemination_score'].agg(['mean', 'std', 'count'])
        optimal_days = daily_dissemination['mean'].nlargest(3).index.tolist()
        
        # Combine for best time slots
        best_combinations = []
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for hour in optimal_hours[:3]:
            for day in optimal_days:
                combo_data = features_df[(features_df['hour'] == hour) & (features_df['day_of_week'] == day)]
                if not combo_data.empty:
                    avg_score = combo_data['dissemination_score'].mean()
                    best_combinations.append({
                        'hour': hour,
                        'day': day,
                        'day_name': day_names[day],
                        'score': avg_score,
                        'sample_size': len(combo_data)
                    })
        
        best_combinations = sorted(best_combinations, key=lambda x: x['score'], reverse=True)
        
        # Seasonal patterns (if data spans multiple weeks)
        date_range = features_df['datetime'].max() - features_df['datetime'].min()
        if date_range.days >= 14:  # At least 2 weeks of data
            features_df['week_of_year'] = features_df['datetime'].dt.isocalendar().week
            weekly_patterns = features_df.groupby('week_of_year')['dissemination_score'].mean()
        else:
            weekly_patterns = None
        
        predictions = {
            'optimal_hours': optimal_hours,
            'optimal_days': optimal_days,
            'optimal_day_names': [day_names[d] for d in optimal_days],
            'best_time_combinations': best_combinations[:5],
            'hourly_effectiveness': hourly_dissemination,
            'daily_effectiveness': daily_dissemination,
            'weekly_patterns': weekly_patterns
        }
        
        return predictions
    
    def detect_recurring_patterns(self, features_df):
        """Detect recurring temporal patterns using autocorrelation"""
        print("🔄 Detecting recurring temporal patterns...")
        
        if features_df.empty or len(features_df) < 24:
            print("⚠️ Insufficient data for pattern detection")
            return {}
        
        # Sort by datetime
        features_df = features_df.sort_values('datetime')
        
        # Calculate autocorrelation for key metrics
        metrics = ['n_edges', 'density', 'centralization']
        patterns = {}
        
        for metric in metrics:
            series = features_df[metric].values
            if len(series) > 24:  # Need sufficient data
                # Calculate autocorrelation
                autocorr = []
                for i in range(1, min(48, len(series)//2)):
                    try:
                        correlation = np.corrcoef(series[:-i], series[i:])[0, 1]
                        if not np.isnan(correlation):
                            autocorr.append(correlation)
                        else:
                            autocorr.append(0)
                    except:
                        autocorr.append(0)
                
                # Find significant lags (peaks in autocorrelation)
                autocorr = np.array(autocorr)
                significant_lags = []
                
                for i in range(1, len(autocorr)-1):
                    if (autocorr[i] > autocorr[i-1] and 
                        autocorr[i] > autocorr[i+1] and 
                        autocorr[i] > 0.3):  # Threshold for significance
                        significant_lags.append(i+1)
                
                patterns[metric] = {
                    'autocorrelation': autocorr,
                    'significant_lags': significant_lags,
                    'strongest_pattern': np.argmax(autocorr) + 1 if len(autocorr) > 0 else None
                }
            else:
                patterns[metric] = {
                    'autocorrelation': [],
                    'significant_lags': [],
                    'strongest_pattern': None
                }
        
        return patterns
    
    def generate_pattern_report(self, features_df):
        """Generate comprehensive temporal pattern analysis report"""
        print("📊 Generating comprehensive temporal pattern report...")
        
        if features_df.empty:
            print("⚠️ No temporal features available for analysis")
            return {
                'analysis_period': {},
                'hourly_patterns': {},
                'daily_patterns': {},
                'network_clusters': {},
                'dissemination_predictions': {},
                'recurring_patterns': {},
                'key_insights': ['No temporal data available for analysis']
            }
        
        # Run all analyses
        hourly_patterns = self.identify_hourly_patterns(features_df)
        daily_patterns = self.identify_daily_patterns(features_df)
        clusters = self.cluster_network_states(features_df)
        predictions = self.predict_optimal_dissemination_times(features_df)
        recurring = self.detect_recurring_patterns(features_df)
        
        report = {
            'analysis_period': {
                'start': features_df['datetime'].min(),
                'end': features_df['datetime'].max(),
                'duration_days': (features_df['datetime'].max() - features_df['datetime'].min()).days,
                'total_snapshots': len(features_df)
            },
            'hourly_patterns': hourly_patterns,
            'daily_patterns': daily_patterns,
            'network_clusters': clusters,
            'dissemination_predictions': predictions,
            'recurring_patterns': recurring,
            'key_insights': self._generate_key_insights(features_df, hourly_patterns, daily_patterns, predictions)
        }
        
        return report
    
    def _generate_key_insights(self, features_df, hourly_patterns, daily_patterns, predictions):
        """Generate key insights from pattern analysis"""
        insights = []
        
        if features_df.empty:
            return ['No temporal data available for insights']
        
        # Peak activity insights
        if hourly_patterns and hourly_patterns.get('peak_activity_hours'):
            peak_hours = hourly_patterns['peak_activity_hours']
            insights.append(f"🔥 Peak network activity occurs at hours: {peak_hours}")
        
        # Formation insights
        if hourly_patterns and hourly_patterns.get('optimal_formation_hours'):
            formation_hours = hourly_patterns['optimal_formation_hours']
            insights.append(f"🌟 Optimal network formation hours: {formation_hours}")
        
        # Daily insights
        if daily_patterns and daily_patterns.get('weekday_pattern') and daily_patterns.get('weekend_pattern'):
            weekday_avg = daily_patterns['weekday_pattern']['avg_edges']
            weekend_avg = daily_patterns['weekend_pattern']['avg_edges']
            
            if weekday_avg > weekend_avg:
                insights.append("📅 Weekdays show higher network activity than weekends")
            else:
                insights.append("📅 Weekends show higher network activity than weekdays")
        
        # Dissemination insights
        if predictions and predictions.get('best_time_combinations'):
            best_times = predictions['best_time_combinations'][:3]
            if best_times:
                best_time_str = ", ".join([f"{t['day_name']} {t['hour']:02d}:00" for t in best_times])
                insights.append(f"🎯 Best dissemination times: {best_time_str}")
        
        # Network structure insights
        avg_centralization = features_df['centralization'].mean()
        if avg_centralization > 0.7:
            insights.append("🏗️ Network shows high centralization (hub-dominated structure)")
        elif avg_centralization < 0.3:
            insights.append("🏗️ Network shows low centralization (distributed structure)")
        else:
            insights.append("🏗️ Network shows moderate centralization (balanced structure)")
        
        return insights
    
    def visualize_patterns(self, features_df, save_path=None):
        """Create comprehensive visualizations of temporal patterns"""
        if features_df.empty:
            print("⚠️ No temporal features available for visualization")
            return None
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Temporal Network Pattern Analysis', fontsize=16, fontweight='bold')
        
        # 1. Hourly activity patterns
        hourly_activity = features_df.groupby('hour')['n_edges'].mean()
        axes[0, 0].bar(hourly_activity.index, hourly_activity.values, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Average Network Activity by Hour')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Average Interactions')
        
        # 2. Daily activity patterns
        daily_activity = features_df.groupby('day_of_week')['n_edges'].mean()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 1].bar(range(7), daily_activity.values, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Average Network Activity by Day')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Average Interactions')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(day_names)
        
        # 3. Network density over time
        axes[0, 2].plot(features_df['datetime'], features_df['density'], alpha=0.7, color='green')
        axes[0, 2].set_title('Network Density Over Time')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Network Density')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Dissemination score heatmap
        try:
            heatmap_data = features_df.pivot_table(
                values='dissemination_score', 
                index='hour', 
                columns='day_of_week', 
                aggfunc='mean'
            )
            sns.heatmap(heatmap_data, ax=axes[1, 0], cmap='YlOrRd', annot=True, fmt='.2f')
            axes[1, 0].set_title('Dissemination Effectiveness Heatmap')
            axes[1, 0].set_xlabel('Day of Week')
            axes[1, 0].set_ylabel('Hour of Day')
        except:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data\nfor heatmap', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Dissemination Effectiveness Heatmap')
        
        # 5. Network centralization patterns
        hourly_central = features_df.groupby('hour')['centralization'].mean()
        axes[1, 1].plot(hourly_central.index, hourly_central.values, marker='o', color='purple')
        axes[1, 1].set_title('Network Centralization by Hour')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Average Centralization')
        
        # 6. Activity growth patterns
        axes[1, 2].plot(features_df['datetime'], features_df['activity_growth'], alpha=0.6, color='orange')
        axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 2].set_title('Network Activity Growth Over Time')
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Activity Growth Rate')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig