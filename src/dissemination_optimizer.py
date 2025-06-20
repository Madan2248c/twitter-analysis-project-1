import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class InformationDisseminationOptimizer:
    """
    Advanced optimizer for finding optimal information dissemination times
    """
    
    def __init__(self, temporal_features_df, pattern_report):
        self.temporal_features_df = temporal_features_df
        self.pattern_report = pattern_report
        self.optimal_times = {}
        
    def calculate_comprehensive_dissemination_score(self):
        """Calculate enhanced dissemination effectiveness score"""
        df = self.temporal_features_df.copy()
        
        if df.empty:
            return df
            
        # Normalize all metrics to 0-1 scale
        metrics = ['density', 'avg_clustering', 'network_efficiency', 'n_edges']
        normalized_df = df.copy()
        
        for metric in metrics:
            if df[metric].max() > 0:
                normalized_df[f'{metric}_norm'] = df[metric] / df[metric].max()
            else:
                normalized_df[f'{metric}_norm'] = 0
        
        # Invert centralization (lower is better for dissemination)
        if df['centralization'].max() > 0:
            normalized_df['centralization_inv'] = 1 - (df['centralization'] / df['centralization'].max())
        else:
            normalized_df['centralization_inv'] = 1
            
        # Enhanced dissemination score with weighted factors
        df['enhanced_dissemination_score'] = (
            normalized_df['density_norm'] * 0.25 +           # Network connectivity
            normalized_df['avg_clustering_norm'] * 0.20 +    # Local clustering
            normalized_df['centralization_inv'] * 0.25 +     # Decentralization
            normalized_df['network_efficiency_norm'] * 0.15 + # Network efficiency
            normalized_df['n_edges_norm'] * 0.15            # Activity level
        )
        
        return df
    
    def find_optimal_dissemination_windows(self):
        """Find optimal time windows for information dissemination"""
        df = self.calculate_comprehensive_dissemination_score()
        
        if df.empty:
            return {}
        
        results = {
            'hourly_analysis': self._analyze_hourly_patterns(df),
            'daily_analysis': self._analyze_daily_patterns(df),
            'combined_analysis': self._analyze_combined_patterns(df),
            'time_windows': self._identify_optimal_windows(df),
            'audience_specific': self._analyze_audience_patterns(df)
        }
        
        return results
    
    def _analyze_hourly_patterns(self, df):
        """Detailed hourly pattern analysis"""
        hourly_stats = df.groupby('hour').agg({
            'enhanced_dissemination_score': ['mean', 'std', 'count'],
            'n_nodes': 'mean',
            'n_edges': 'mean',
            'density': 'mean'
        }).round(4)
        
        hourly_stats.columns = ['_'.join(col) for col in hourly_stats.columns]
        
        # Calculate confidence intervals
        hourly_stats['confidence_interval'] = (
            1.96 * hourly_stats['enhanced_dissemination_score_std'] / 
            np.sqrt(hourly_stats['enhanced_dissemination_score_count'])
        )
        
        # Rank hours by effectiveness
        hourly_stats['effectiveness_rank'] = hourly_stats['enhanced_dissemination_score_mean'].rank(ascending=False)
        
        # Categorize hours - FIXED: Handle single hour case
        if len(hourly_stats) > 1:
            top_threshold = hourly_stats['enhanced_dissemination_score_mean'].quantile(0.75)
            hourly_stats['category'] = hourly_stats['enhanced_dissemination_score_mean'].apply(
                lambda x: 'Excellent' if x >= top_threshold else 
                         'Good' if x >= hourly_stats['enhanced_dissemination_score_mean'].median() else 'Poor'
            )
        else:
            # If only one hour, mark it as the only available option
            hourly_stats['category'] = 'Only Available Hour'
        
        return hourly_stats.sort_values('enhanced_dissemination_score_mean', ascending=False)
    
    def _analyze_daily_patterns(self, df):
        """Detailed daily pattern analysis"""
        daily_stats = df.groupby('day_of_week').agg({
            'enhanced_dissemination_score': ['mean', 'std', 'count'],
            'n_nodes': 'mean',
            'n_edges': 'mean',
            'density': 'mean'
        }).round(4)
        
        daily_stats.columns = ['_'.join(col) for col in daily_stats.columns]
        
        # Add day names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_stats['day_name'] = [day_names[i] for i in daily_stats.index]
        
        # Categorize weekdays vs weekends
        daily_stats['day_type'] = daily_stats.index.map(
            lambda x: 'Weekend' if x >= 5 else 'Weekday'
        )
        
        return daily_stats.sort_values('enhanced_dissemination_score_mean', ascending=False)
    
    def _analyze_combined_patterns(self, df):
        """Analyze hour-day combinations"""
        combined_stats = df.groupby(['hour', 'day_of_week']).agg({
            'enhanced_dissemination_score': ['mean', 'count'],
            'n_edges': 'mean'
        }).round(4)
        
        combined_stats.columns = ['_'.join(col) for col in combined_stats.columns]
        
        # Filter combinations with sufficient data
        min_observations = 1  # REDUCED: Since you have limited data
        reliable_combinations = combined_stats[
            combined_stats['enhanced_dissemination_score_count'] >= min_observations
        ]
        
        if not reliable_combinations.empty:
            # Add day names and create readable time slots
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            reliable_combinations = reliable_combinations.reset_index()
            reliable_combinations['day_name'] = reliable_combinations['day_of_week'].map(
                lambda x: day_names[x]
            )
            reliable_combinations['time_slot'] = reliable_combinations.apply(
                lambda row: f"{row['day_name']} {row['hour']:02d}:00", axis=1
            )
            
            return reliable_combinations.sort_values('enhanced_dissemination_score_mean', ascending=False)
        
        return pd.DataFrame()
    
    def _identify_optimal_windows(self, df):
        """Identify continuous optimal time windows"""
        if df.empty:
            return []
            
        # Find periods of sustained high performance
        threshold = df['enhanced_dissemination_score'].quantile(0.60)  # LOWERED: More inclusive
        df_sorted = df.sort_values('datetime')
        
        windows = []
        current_window = []
        
        for _, row in df_sorted.iterrows():
            if row['enhanced_dissemination_score'] >= threshold:
                current_window.append(row)
            else:
                if len(current_window) >= 1:  # REDUCED: Minimum window size
                    windows.append({
                        'start_time': current_window[0]['datetime'],
                        'end_time': current_window[-1]['datetime'],
                        'duration_hours': (current_window[-1]['datetime'] - current_window[0]['datetime']).total_seconds() / 3600,
                        'avg_score': np.mean([w['enhanced_dissemination_score'] for w in current_window]),
                        'num_periods': len(current_window)
                    })
                current_window = []
        
        # Check last window
        if len(current_window) >= 1:  # REDUCED: Allow single-period windows
            windows.append({
                'start_time': current_window[0]['datetime'],
                'end_time': current_window[-1]['datetime'],
                'duration_hours': (current_window[-1]['datetime'] - current_window[0]['datetime']).total_seconds() / 3600,
                'avg_score': np.mean([w['enhanced_dissemination_score'] for w in current_window]),
                'num_periods': len(current_window)
            })
        
        return sorted(windows, key=lambda x: x['avg_score'], reverse=True)
    
    def _analyze_audience_patterns(self, df):
        """Analyze patterns for different audience types"""
        if df.empty:
            return {}
            
        # High activity periods (mass audience)
        high_activity_threshold = df['n_edges'].quantile(0.60)  # LOWERED: More inclusive
        mass_audience_times = df[df['n_edges'] >= high_activity_threshold]
        
        # High density periods (engaged audience)
        high_density_threshold = df['density'].quantile(0.60)  # LOWERED: More inclusive
        engaged_audience_times = df[df['density'] >= high_density_threshold]
        
        # Balanced periods (optimal for broad dissemination)
        balanced_times = df[
            (df['enhanced_dissemination_score'] >= df['enhanced_dissemination_score'].quantile(0.60)) &
            (df['n_edges'] >= df['n_edges'].median())
        ]
        
        return {
            'mass_audience': self._summarize_audience_pattern(mass_audience_times, 'Mass Audience'),
            'engaged_audience': self._summarize_audience_pattern(engaged_audience_times, 'Engaged Audience'),
            'balanced_dissemination': self._summarize_audience_pattern(balanced_times, 'Balanced Dissemination')
        }
    
    def _summarize_audience_pattern(self, df_subset, audience_type):
        """Summarize patterns for specific audience type"""
        if df_subset.empty:
            return {'audience_type': audience_type, 'optimal_hours': [], 'optimal_days': []}
            
        optimal_hours = df_subset.groupby('hour')['enhanced_dissemination_score'].mean().nlargest(3).index.tolist()
        optimal_days = df_subset.groupby('day_of_week')['enhanced_dissemination_score'].mean().nlargest(3).index.tolist()
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        optimal_day_names = [day_names[d] for d in optimal_days]
        
        return {
            'audience_type': audience_type,
            'optimal_hours': optimal_hours,
            'optimal_days': optimal_day_names,
            'avg_score': df_subset['enhanced_dissemination_score'].mean(),
            'sample_size': len(df_subset)
        }
    
    def generate_recommendations(self):
        """Generate actionable recommendations - FIXED"""
        results = self.find_optimal_dissemination_windows()
        
        recommendations = {
            'top_time_slots': [],
            'hourly_recommendations': {},
            'daily_recommendations': {},
            'audience_specific_recommendations': {},
            'strategic_insights': []
        }
        
        # Top time slots
        if not results['combined_analysis'].empty:
            top_slots = results['combined_analysis'].head(5)
            for _, slot in top_slots.iterrows():
                recommendations['top_time_slots'].append({
                    'time_slot': slot['time_slot'],
                    'effectiveness_score': slot['enhanced_dissemination_score_mean'],
                    'confidence': 'High' if slot['enhanced_dissemination_score_count'] >= 2 else 'Medium',
                    'expected_reach': slot['n_edges_mean']
                })
        
        # Hourly recommendations - FIXED
        hourly_analysis = results['hourly_analysis']
        if not hourly_analysis.empty:
            if len(hourly_analysis) > 1:
                # Multiple hours available
                best_hours = hourly_analysis.head(3).index.tolist()
                worst_hours = hourly_analysis.tail(3).index.tolist()
                peak_hour = hourly_analysis.index[0]
            else:
                # Only one hour available
                best_hours = hourly_analysis.index.tolist()
                worst_hours = ["Limited data - only 19:00 available"]
                peak_hour = hourly_analysis.index[0]
                
            recommendations['hourly_recommendations'] = {
                'best_hours': best_hours,
                'worst_hours': worst_hours,
                'peak_effectiveness_hour': peak_hour,
                'data_limitation': len(hourly_analysis) == 1
            }
        
        # Daily recommendations
        daily_analysis = results['daily_analysis']
        if not daily_analysis.empty:
            recommendations['daily_recommendations'] = {
                'best_days': daily_analysis.head(3)['day_name'].tolist(),
                'worst_days': daily_analysis.tail(3)['day_name'].tolist() if len(daily_analysis) > 3 else ["Limited data available"],
                'weekday_vs_weekend': self._compare_weekday_weekend(daily_analysis)
            }
        
        # Audience-specific recommendations
        recommendations['audience_specific_recommendations'] = results['audience_specific']
        
        # Strategic insights - ENHANCED
        recommendations['strategic_insights'] = self._generate_strategic_insights(results, hourly_analysis)
        
        return recommendations
    
    def _compare_weekday_weekend(self, daily_analysis):
        """Compare weekday vs weekend performance"""
        if 'day_type' not in daily_analysis.columns:
            return {'note': 'Insufficient data for weekday/weekend comparison'}
            
        weekday_data = daily_analysis[daily_analysis['day_type'] == 'Weekday']
        weekend_data = daily_analysis[daily_analysis['day_type'] == 'Weekend']
        
        if weekday_data.empty or weekend_data.empty:
            return {'note': 'Insufficient data for weekday/weekend comparison'}
            
        weekday_avg = weekday_data['enhanced_dissemination_score_mean'].mean()
        weekend_avg = weekend_data['enhanced_dissemination_score_mean'].mean()
        
        return {
            'weekday_average': weekday_avg,
            'weekend_average': weekend_avg,
            'better_period': 'Weekdays' if weekday_avg > weekend_avg else 'Weekends',
            'difference': abs(weekday_avg - weekend_avg)
        }
    
    def _generate_strategic_insights(self, results, hourly_analysis):
        """Generate strategic insights from the analysis - ENHANCED"""
        insights = []
        
        # Data limitation insights
        if not hourly_analysis.empty and len(hourly_analysis) == 1:
            insights.append("⚠️ Limited time coverage: Only 19:00 hour data available. Consider collecting data across more hours for comprehensive analysis.")
        
        # Time window insights
        if results['time_windows']:
            best_window = results['time_windows'][0]
            insights.append(f"🎯 Primary dissemination window identified: {best_window['start_time'].strftime('%Y-%m-%d %H:%M')} to {best_window['end_time'].strftime('%Y-%m-%d %H:%M')}")
        
        # Consistency insights
        if not hourly_analysis.empty:
            if len(hourly_analysis) == 1:
                insights.append("📊 19:00 shows consistent performance across all observed days")
            else:
                most_consistent = hourly_analysis.loc[hourly_analysis['enhanced_dissemination_score_std'].idxmin()]
                insights.append(f"📈 Most consistent hour for dissemination: {most_consistent.name}:00 (low variability)")
        
        # Daily pattern insights
        daily_analysis = results['daily_analysis']
        if not daily_analysis.empty and len(daily_analysis) > 1:
            best_day = daily_analysis.iloc[0]['day_name']
            insights.append(f"📅 {best_day} consistently shows highest dissemination effectiveness")
        
        # Audience insights
        audience_patterns = results['audience_specific']
        if audience_patterns:
            insights.append("👥 Different audience types show distinct optimal timing patterns - consider segmented content strategy")
        
        # Network structure insights
        if not self.temporal_features_df.empty:
            avg_density = self.temporal_features_df['density'].mean()
            if avg_density > 0.1:
                insights.append("🔗 Network shows good connectivity for information spread")
            else:
                insights.append("🔗 Network has sparse connectivity - consider targeted influencer engagement")
        
        return insights
    
    def create_dissemination_dashboard(self, save_path=None):
        """Create comprehensive visualization dashboard"""
        results = self.find_optimal_dissemination_windows()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Information Dissemination Optimization Dashboard\n(Based on Available Data)', fontsize=16, fontweight='bold')
        
        # 1. Hourly effectiveness
        hourly = results['hourly_analysis']
        if not hourly.empty:
            bars = axes[0, 0].bar(hourly.index, hourly['enhanced_dissemination_score_mean'], 
                          alpha=0.7, color='skyblue', edgecolor='navy')
            
            # Add data limitation warning if only one hour
            if len(hourly) == 1:
                axes[0, 0].text(0.5, 0.9, '⚠️ Limited to 19:00 hour only', 
                               transform=axes[0, 0].transAxes, ha='center', 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            axes[0, 0].set_title('Dissemination Effectiveness by Hour')
            axes[0, 0].set_xlabel('Hour of Day')
            axes[0, 0].set_ylabel('Effectiveness Score')
            
            # Highlight the available hour
            for i, (hour, score) in enumerate(zip(hourly.index, hourly['enhanced_dissemination_score_mean'])):
                bars[i].set_color('gold')
                axes[0, 0].text(hour, score + 0.01, f'{score:.3f}', 
                               ha='center', va='bottom', fontweight='bold')
        
        # 2. Daily effectiveness
        daily = results['daily_analysis']
        if not daily.empty:
            colors = ['lightcoral' if day_type == 'Weekend' else 'lightblue' 
                     for day_type in daily['day_type']]
            bars = axes[0, 1].bar(range(len(daily)), daily['enhanced_dissemination_score_mean'], 
                          alpha=0.7, color=colors, edgecolor='navy')
            axes[0, 1].set_title('Dissemination Effectiveness by Day')
            axes[0, 1].set_xlabel('Day of Week')
            axes[0, 1].set_ylabel('Effectiveness Score')
            axes[0, 1].set_xticks(range(len(daily)))
            axes[0, 1].set_xticklabels([name[:3] for name in daily['day_name']], rotation=45)
            
            # Highlight best performing days
            for i, (day, score) in enumerate(zip(daily['day_name'], daily['enhanced_dissemination_score_mean'])):
                if i < 3:  # Top 3 days
                    bars[i].set_color('gold')
        
        # 3. Heatmap of hour-day combinations
        if not results['combined_analysis'].empty and len(results['combined_analysis']) > 1:
            try:
                pivot_data = results['combined_analysis'].pivot(
                    index='hour', columns='day_of_week', values='enhanced_dissemination_score_mean'
                )
                sns.heatmap(pivot_data, ax=axes[0, 2], cmap='YlOrRd', annot=True, fmt='.3f')
                axes[0, 2].set_title('Effectiveness Heatmap (Hour vs Day)')
                axes[0, 2].set_xlabel('Day of Week')
                axes[0, 2].set_ylabel('Hour of Day')
            except:
                axes[0, 2].text(0.5, 0.5, 'Insufficient data\nfor heatmap\n(Only 19:00 available)', 
                               ha='center', va='center', transform=axes[0, 2].transAxes,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                axes[0, 2].set_title('Hour-Day Effectiveness Heatmap')
        else:
            axes[0, 2].text(0.5, 0.5, 'Limited time data\nNo heatmap possible', 
                           ha='center', va='center', transform=axes[0, 2].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[0, 2].set_title('Hour-Day Effectiveness Heatmap')
        
        # 4. Time windows
        if results['time_windows']:
            window_scores = [w['avg_score'] for w in results['time_windows']]
            window_labels = [f"Window {i+1}" for i in range(len(window_scores))]
            axes[1, 0].bar(window_labels, window_scores, alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('Optimal Time Windows')
            axes[1, 0].set_ylabel('Average Effectiveness Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No optimal\ntime windows\nidentified', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Optimal Time Windows')
        
        # 5. Audience-specific patterns
        audience_data = results['audience_specific']
        if audience_data:
            audience_names = list(audience_data.keys())
            audience_scores = [audience_data[name]['avg_score'] 
                             for name in audience_names if 'avg_score' in audience_data[name]]
            if audience_scores:
                axes[1, 1].bar(audience_names, audience_scores, alpha=0.7, color='orange')
                axes[1, 1].set_title('Effectiveness by Audience Type')
                axes[1, 1].set_ylabel('Average Effectiveness Score')
                axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No audience\npatterns available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Effectiveness by Audience Type')
        
        # 6. Temporal evolution
        if not self.temporal_features_df.empty:
            df_enhanced = self.calculate_comprehensive_dissemination_score()
            axes[1, 2].plot(df_enhanced['datetime'], df_enhanced['enhanced_dissemination_score'], 
                           alpha=0.7, color='purple', linewidth=2, marker='o')
            axes[1, 2].set_title('Dissemination Effectiveness Over Time')
            axes[1, 2].set_xlabel('Time')
            axes[1, 2].set_ylabel('Effectiveness Score')
            axes[1, 2].tick_params(axis='x', rotation=45)
            
            # Add annotation for data limitation
            axes[1, 2].text(0.02, 0.98, 'Data limited to 19:00 hour', 
                           transform=axes[1, 2].transAxes, va='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig