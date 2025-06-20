import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class TwitterVisualization:
    """
    Comprehensive visualization module for Twitter social network analysis.
    Creates plots for temporal sentiment, network structure, and influence tracking.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualization settings."""
        self.figsize = figsize
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Color schemes
        self.sentiment_colors = {
            'positive': '#2ECC71',
            'negative': '#E74C3C', 
            'neutral': '#95A5A6'
        }
        
        self.centrality_colors = {
            'pagerank': '#3498DB',
            'betweenness': '#E67E22',
            'in_degree': '#9B59B6',
            'out_degree': '#1ABC9C'
        }
    
    def plot_temporal_sentiment(self, temporal_sentiment_df: pd.DataFrame, 
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot sentiment evolution over time."""
        if temporal_sentiment_df.empty:
            print("No temporal sentiment data to plot.")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Temporal Sentiment Analysis', fontsize=16, fontweight='bold')
        
        # Convert time_period to datetime if needed
        if 'time_period' in temporal_sentiment_df.columns:
            temporal_sentiment_df['time_period'] = pd.to_datetime(temporal_sentiment_df['time_period'])
        
        # 1. Sentiment scores over time
        ax1 = axes[0, 0]
        ax1.plot(temporal_sentiment_df['time_period'], temporal_sentiment_df['avg_compound'], 
                marker='o', linewidth=2, markersize=4, color='#34495E', label='Compound Score')
        ax1.fill_between(temporal_sentiment_df['time_period'], temporal_sentiment_df['avg_compound'], 
                        alpha=0.3, color='#34495E')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_title('Average Sentiment Score Over Time')
        ax1.set_ylabel('Compound Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sentiment distribution percentages
        ax2 = axes[0, 1]
        ax2.plot(temporal_sentiment_df['time_period'], temporal_sentiment_df['positive_pct'], 
                marker='o', color=self.sentiment_colors['positive'], label='Positive %')
        ax2.plot(temporal_sentiment_df['time_period'], temporal_sentiment_df['negative_pct'], 
                marker='s', color=self.sentiment_colors['negative'], label='Negative %')
        ax2.plot(temporal_sentiment_df['time_period'], temporal_sentiment_df['neutral_pct'], 
                marker='^', color=self.sentiment_colors['neutral'], label='Neutral %')
        ax2.set_title('Sentiment Distribution Over Time')
        ax2.set_ylabel('Percentage')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Tweet volume and sentiment volatility
        ax3 = axes[1, 0]
        bars = ax3.bar(temporal_sentiment_df['time_period'], temporal_sentiment_df['total_tweets'], 
                      alpha=0.7, color='#3498DB', label='Tweet Volume')
        ax3.set_title('Tweet Volume Over Time')
        ax3.set_ylabel('Number of Tweets')
        ax3.legend()
        
        # Add volatility line on secondary axis
        ax3_twin = ax3.twinx()
        ax3_twin.plot(temporal_sentiment_df['time_period'], temporal_sentiment_df['sentiment_volatility'], 
                     color='red', marker='o', linewidth=2, label='Sentiment Volatility')
        ax3_twin.set_ylabel('Sentiment Volatility', color='red')
        ax3_twin.legend(loc='upper right')
        
        # 4. Dominant sentiment heatmap-style plot
        ax4 = axes[1, 1]
        sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        sentiment_numeric = temporal_sentiment_df['dominant_sentiment'].map(sentiment_mapping)
        
        colors = [self.sentiment_colors[sent] for sent in temporal_sentiment_df['dominant_sentiment']]
        ax4.scatter(temporal_sentiment_df['time_period'], [1]*len(temporal_sentiment_df), 
                   c=colors, s=100, alpha=0.8)
        ax4.set_title('Dominant Sentiment by Period')
        ax4.set_ylim(0.5, 1.5)
        ax4.set_yticks([])
        
        # Add legend
        for sentiment, color in self.sentiment_colors.items():
            ax4.scatter([], [], c=color, s=100, label=sentiment.capitalize())
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved temporal sentiment plot to {save_path}")
        
        return fig
    
    def plot_user_sentiment_patterns(self, user_patterns_df: pd.DataFrame, 
                                    top_n: int = 20, save_path: Optional[str] = None) -> plt.Figure:
        """Plot user sentiment patterns and consistency."""
        if user_patterns_df.empty:
            print("No user sentiment patterns to plot.")
            return None
        
        # Sort by total tweets and take top users
        top_users = user_patterns_df.nlargest(top_n, 'total_tweets')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('User Sentiment Patterns Analysis', fontsize=16, fontweight='bold')
        
        # 1. User sentiment consistency vs volatility
        ax1 = axes[0, 0]
        scatter = ax1.scatter(top_users['consistency_score'], top_users['sentiment_volatility'], 
                             c=top_users['avg_compound'], cmap='RdYlGn', 
                             s=top_users['total_tweets']*5, alpha=0.7)
        ax1.set_xlabel('Sentiment Consistency')
        ax1.set_ylabel('Sentiment Volatility')
        ax1.set_title('User Sentiment Consistency vs Volatility')
        plt.colorbar(scatter, ax=ax1, label='Avg Compound Score')
        
        # 2. Dominant sentiment distribution
        ax2 = axes[0, 1]
        sentiment_counts = user_patterns_df['dominant_sentiment'].value_counts()
        colors = [self.sentiment_colors[sent] for sent in sentiment_counts.index]
        wedges, texts, autotexts = ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('User Dominant Sentiment Distribution')
        
        # 3. Top users by sentiment score
        ax3 = axes[1, 0]
        top_by_sentiment = user_patterns_df.nlargest(15, 'avg_compound')
        bars = ax3.barh(range(len(top_by_sentiment)), top_by_sentiment['avg_compound'], 
                       color=[self.sentiment_colors[sent] for sent in top_by_sentiment['dominant_sentiment']])
        ax3.set_yticks(range(len(top_by_sentiment)))
        ax3.set_yticklabels([f"@{username}" for username in top_by_sentiment['username']], fontsize=8)
        ax3.set_xlabel('Average Sentiment Score')
        ax3.set_title('Most Positive Users')
        ax3.grid(True, alpha=0.3)
        
        # 4. User activity vs sentiment relationship
        ax4 = axes[1, 1]
        ax4.scatter(user_patterns_df['total_tweets'], user_patterns_df['avg_compound'], 
                   c=[self.sentiment_colors[sent] for sent in user_patterns_df['dominant_sentiment']], 
                   alpha=0.6, s=50)
        ax4.set_xlabel('Total Tweets')
        ax4.set_ylabel('Average Sentiment')
        ax4.set_title('User Activity vs Sentiment')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add legend
        for sentiment, color in self.sentiment_colors.items():
            ax4.scatter([], [], c=color, s=50, label=sentiment.capitalize())
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved user sentiment patterns plot to {save_path}")
        
        return fig
    
    def plot_network_graph(self, graph: nx.DiGraph, layout: str = 'spring', 
                          node_size_metric: str = 'pagerank', 
                          top_nodes: int = 50, save_path: Optional[str] = None) -> plt.Figure:
        """Plot network graph with nodes sized by centrality metrics."""
        if graph is None or graph.number_of_nodes() == 0:
            print("No graph data to plot.")
            return None
        
        # Calculate centrality metrics if not present
        if not any(node_size_metric in data for _, data in graph.nodes(data=True)):
            try:
                if node_size_metric == 'pagerank':
                    centrality = nx.pagerank(graph, weight='weight')
                elif node_size_metric == 'betweenness':
                    centrality = nx.betweenness_centrality(graph, weight='weight')
                elif node_size_metric == 'in_degree':
                    centrality = dict(graph.in_degree())
                elif node_size_metric == 'out_degree':
                    centrality = dict(graph.out_degree())
                else:
                    centrality = nx.pagerank(graph, weight='weight')
                
                nx.set_node_attributes(graph, centrality, node_size_metric)
            except:
                print(f"Could not calculate {node_size_metric} centrality")
                return None
        
        # Filter to top nodes for better visualization
        if graph.number_of_nodes() > top_nodes:
            node_metrics = nx.get_node_attributes(graph, node_size_metric)
            top_node_ids = sorted(node_metrics.keys(), key=lambda x: node_metrics[x], reverse=True)[:top_nodes]
            subgraph = graph.subgraph(top_node_ids)
        else:
            subgraph = graph
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(subgraph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(subgraph)
        else:
            pos = nx.spring_layout(subgraph)
        
        # Get node attributes
        node_sizes = []
        node_colors = []
        
        for node in subgraph.nodes():
            # Node size based on centrality metric
            metric_value = subgraph.nodes[node].get(node_size_metric, 0)
            node_sizes.append(max(50, metric_value * 1000))  # Scale for visibility
            
            # Node color based on follower count or other attribute
            followers = subgraph.nodes[node].get('followers_count', 0)
            node_colors.append(followers)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, edge_color='gray', 
                              width=0.5, arrows=True, arrowsize=10)
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                                      node_color=node_colors, cmap='viridis', 
                                      alpha=0.8, ax=ax)
        
        # Add colorbar for node colors
        if nodes:
            plt.colorbar(nodes, ax=ax, label='Followers Count')
        
        # Add labels for top nodes
        top_n_labels = 10
        node_metrics = nx.get_node_attributes(subgraph, node_size_metric)
        top_labeled_nodes = sorted(node_metrics.keys(), key=lambda x: node_metrics[x], reverse=True)[:top_n_labels]
        
        labels = {}
        for node in top_labeled_nodes:
            username = subgraph.nodes[node].get('username', str(node))
            labels[node] = f"@{username}"
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, font_weight='bold')
        
        ax.set_title(f'Social Network Graph (sized by {node_size_metric})', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add network statistics
        stats_text = f"Nodes: {subgraph.number_of_nodes()}\nEdges: {subgraph.number_of_edges()}\nDensity: {nx.density(subgraph):.4f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved network graph to {save_path}")
        
        return fig
    
    def plot_centrality_rankings(self, centrality_metrics: Dict[str, Dict], 
                                top_k: int = 15, save_path: Optional[str] = None) -> plt.Figure:
        """Plot top users by different centrality metrics."""
        if not centrality_metrics:
            print("No centrality metrics to plot.")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(centrality_metrics, orient='index')
        
        # Create 2x2 grid but only use 3 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('User Influence Rankings by Centrality Metrics', fontsize=16, fontweight='bold')
        
        # Only use 3 metrics - REMOVED betweenness_centrality
        metrics = ['pagerank', 'in_degree_centrality', 'out_degree_centrality']
        titles = ['PageRank (Overall Influence)',
                 'In-Degree (Popular Users)', 
                 'Out-Degree (Active Users)']
        
        # Plot positions: (0,0), (0,1), (1,0) - skip (1,1)
        positions = [(0, 0), (0, 1), (1, 0)]
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = positions[i]
            ax = axes[row, col]
            
            if metric in df.columns:
                top_users = df.nlargest(top_k, metric)
                
                bars = ax.barh(range(len(top_users)), top_users[metric], 
                              color=self.centrality_colors.get(metric.split('_')[0], '#3498DB'))
                
                # Add user labels (use index as user_id for now)
                ax.set_yticks(range(len(top_users)))
                ax.set_yticklabels([f"User_{user_id[:8]}" for user_id in top_users.index], fontsize=8)
                ax.set_xlabel(f'{metric.replace("_", " ").title()} Score')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for j, (idx, row) in enumerate(top_users.iterrows()):
                    ax.text(row[metric], j, f'{row[metric]:.4f}', 
                           va='center', ha='left', fontsize=8)
            else:
                ax.text(0.5, 0.5, f'{metric} not available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
        
        # Hide the unused fourth subplot (bottom right)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved centrality rankings plot to {save_path}")
        
        return fig
    
    def plot_temporal_influence(self, temporal_influence_df: pd.DataFrame, 
                               top_users: int = 10, save_path: Optional[str] = None) -> plt.Figure:
        """Plot how user influence changes over time."""
        if temporal_influence_df.empty:
            print("No temporal influence data to plot.")
            return None
        
        # Get top influential users overall
        user_influence = temporal_influence_df.groupby('user_id')['pagerank'].mean().sort_values(ascending=False)
        top_user_ids = user_influence.head(top_users).index
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Temporal Influence Analysis', fontsize=16, fontweight='bold')
        
        # 1. PageRank evolution for top users
        ax1 = axes[0]
        for i, user_id in enumerate(top_user_ids):
            user_data = temporal_influence_df[temporal_influence_df['user_id'] == user_id]
            user_data = user_data.sort_values('timestamp')
            
            ax1.plot(pd.to_datetime(user_data['timestamp']), user_data['pagerank'], 
                    marker='o', linewidth=2, markersize=4, label=f'User_{user_id[:8]}')
        
        ax1.set_title('PageRank Evolution for Top Users')
        ax1.set_ylabel('PageRank Score')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Average influence metrics over time
        ax2 = axes[1]
        temporal_avg = temporal_influence_df.groupby('timestamp').agg({
            'pagerank': 'mean',
            'in_degree_centrality': 'mean',
            'out_degree_centrality': 'mean',
            'total_interactions': 'mean'
        }).reset_index()
        
        temporal_avg['timestamp'] = pd.to_datetime(temporal_avg['timestamp'])
        
        ax2_twin = ax2.twinx()
        
        ax2.plot(temporal_avg['timestamp'], temporal_avg['pagerank'], 
                marker='o', color='blue', label='Avg PageRank')
        ax2.plot(temporal_avg['timestamp'], temporal_avg['in_degree_centrality'], 
                marker='s', color='green', label='Avg In-Degree')
        ax2.plot(temporal_avg['timestamp'], temporal_avg['out_degree_centrality'], 
                marker='^', color='red', label='Avg Out-Degree')
        
        ax2_twin.plot(temporal_avg['timestamp'], temporal_avg['total_interactions'], 
                     marker='D', color='orange', label='Avg Interactions', linewidth=2)
        
        ax2.set_title('Average Network Metrics Over Time')
        ax2.set_ylabel('Centrality Scores')
        ax2_twin.set_ylabel('Total Interactions', color='orange')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved temporal influence plot to {save_path}")
        
        return fig
    
    def plot_community_structure(self, graph: nx.DiGraph, communities: Dict[str, int], 
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot network graph colored by community membership."""
        if not communities or graph is None:
            print("No community data to plot.")
            return None
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Filter graph to nodes with community assignments
        community_nodes = [node for node in graph.nodes() if node in communities]
        subgraph = graph.subgraph(community_nodes)
        
        if subgraph.number_of_nodes() == 0:
            print("No nodes with community assignments.")
            return None
        
        # Create layout
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Get unique communities and assign colors
        unique_communities = list(set(communities.values()))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
        community_colors = dict(zip(unique_communities, colors))
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, edge_color='gray', width=0.5)
        
        # Draw nodes colored by community
        for community_id in unique_communities:
            community_nodes = [node for node, comm in communities.items() 
                             if comm == community_id and node in subgraph.nodes()]
            
            if community_nodes:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=community_nodes,
                                     node_color=[community_colors[community_id]], 
                                     node_size=100, alpha=0.8, 
                                     label=f'Community {community_id}')
        
        ax.set_title('Network Community Structure', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add community statistics
        community_sizes = pd.Series(communities.values()).value_counts().sort_index()
        stats_text = f"Communities: {len(unique_communities)}\nLargest: {community_sizes.max()} nodes\nSmallest: {community_sizes.min()} nodes"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved community structure plot to {save_path}")
        
        return fig
    
    def create_interactive_sentiment_dashboard(self, temporal_sentiment_df: pd.DataFrame, 
                                             sentiment_df: pd.DataFrame, 
                                             save_path: Optional[str] = None):
        """Create an interactive dashboard using Plotly."""
        if temporal_sentiment_df.empty and sentiment_df.empty:
            print("No data for interactive dashboard.")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Over Time', 'Sentiment Distribution', 
                           'Tweet Volume', 'Sentiment vs Followers'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        if not temporal_sentiment_df.empty:
            # Temporal sentiment
            fig.add_trace(
                go.Scatter(x=temporal_sentiment_df['time_period'], 
                          y=temporal_sentiment_df['avg_compound'],
                          mode='lines+markers', name='Avg Sentiment',
                          line=dict(color='blue', width=3)),
                row=1, col=1
            )
            
            # Tweet volume with volatility
            fig.add_trace(
                go.Bar(x=temporal_sentiment_df['time_period'], 
                      y=temporal_sentiment_df['total_tweets'],
                      name='Tweet Volume', marker_color='lightblue'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=temporal_sentiment_df['time_period'], 
                          y=temporal_sentiment_df['sentiment_volatility'],
                          mode='lines+markers', name='Volatility',
                          line=dict(color='red', width=2), yaxis='y2'),
                row=2, col=1, secondary_y=True
            )
        
        if not sentiment_df.empty:
            # Sentiment distribution pie chart
            sentiment_counts = sentiment_df['sentiment_label'].value_counts()
            fig.add_trace(
                go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values,
                      name="Sentiment Distribution"),
                row=1, col=2
            )
            
            # Sentiment vs followers (if available)
            if 'followers_count' in sentiment_df.columns:
                fig.add_trace(
                    go.Scatter(x=sentiment_df['followers_count'], 
                              y=sentiment_df['compound'],
                              mode='markers', name='Sentiment vs Followers',
                              marker=dict(size=8, opacity=0.6)),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Sentiment Analysis Dashboard",
            title_x=0.5,
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Saved interactive dashboard to {save_path}")
        
        return fig

if __name__ == "__main__":
    # Example usage
    from preprocess import TwitterDataPreprocessor
    from build_graph import TemporalGraphBuilder
    from sentiment_analysis import TwitterSentimentAnalyzer
    
    # Load and process data
    preprocessor = TwitterDataPreprocessor("../data/tweetdata-en.csv")
    users, interactions = preprocessor.process_data(filter_business=True)
    
    # Build graphs
    graph_builder = TemporalGraphBuilder(users, interactions)
    full_graph = graph_builder.create_full_graph()
    temporal_graphs = graph_builder.create_temporal_snapshots('daily')
    centrality_metrics = graph_builder.calculate_centrality_metrics()
    communities = graph_builder.detect_communities()
    
    # Analyze sentiment
    business_tweets = preprocessor.filter_business_related()
    sentiment_analyzer = TwitterSentimentAnalyzer()
    sentiment_df = sentiment_analyzer.analyze_dataset_sentiment(business_tweets)
    temporal_sentiment = sentiment_analyzer.analyze_temporal_sentiment(sentiment_df)
    user_patterns = sentiment_analyzer.analyze_user_sentiment_patterns(sentiment_df)
    
    # Create visualizations
    viz = TwitterVisualization()
    
    # Plot temporal sentiment
    viz.plot_temporal_sentiment(temporal_sentiment, save_path="../plots/temporal_sentiment.png")
    
    # Plot user sentiment patterns
    viz.plot_user_sentiment_patterns(user_patterns, save_path="../plots/user_sentiment_patterns.png")
    
    # Plot network graph
    viz.plot_network_graph(full_graph, node_size_metric='pagerank', 
                          save_path="../plots/network_graph.png")
    
    # Plot centrality rankings
    viz.plot_centrality_rankings(centrality_metrics, save_path="../plots/centrality_rankings.png")
    
    # Plot community structure
    viz.plot_community_structure(full_graph, communities, 
                                save_path="../plots/community_structure.png")
    
    # Create interactive dashboard
    viz.create_interactive_sentiment_dashboard(temporal_sentiment, sentiment_df, 
                                             save_path="../plots/interactive_dashboard.html")
    
    plt.show()