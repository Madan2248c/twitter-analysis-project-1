import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import pickle

class TemporalGraphBuilder:
    """
    Builds temporal social graphs from Twitter interaction data.
    Creates snapshots of the network at different time intervals for analysis.
    """
    
    def __init__(self, users: Dict, interactions: List[Dict]):
        """Initialize with preprocessed users and interactions data."""
        self.users = users
        self.interactions = interactions
        self.temporal_graphs = {}
        self.full_graph = None
        self.time_intervals = []
        
    def create_full_graph(self) -> nx.DiGraph:
        """Create a complete directed graph with all interactions."""
        G = nx.DiGraph()
        
        # Add all users as nodes with attributes
        for user_id, user_info in self.users.items():
            G.add_node(user_id, **user_info)
        
        # Track multiple interactions between same users
        edge_weights = defaultdict(float)
        edge_types = defaultdict(list)
        edge_timestamps = defaultdict(list)
        
        # Process all interactions
        for interaction in self.interactions:
            source = interaction['source_user_id']
            target = interaction['target_user_id']
            
            if source and target and source in self.users and target in self.users:
                edge_key = (source, target)
                edge_weights[edge_key] += interaction['weight']
                edge_types[edge_key].append(interaction['edge_type'])
                edge_timestamps[edge_key].append(interaction['timestamp'])
        
        # Add edges with aggregated attributes
        for (source, target), weight in edge_weights.items():
            types = edge_types[(source, target)]
            timestamps = edge_timestamps[(source, target)]
            
            # Calculate edge attributes
            edge_attrs = {
                'weight': weight,
                'interaction_count': len(types),
                'edge_types': list(set(types)),
                'first_interaction': min(timestamps),
                'last_interaction': max(timestamps),
                'interaction_span': (max(timestamps) - min(timestamps)).days,
                'dominant_type': max(set(types), key=types.count)
            }
            
            G.add_edge(source, target, **edge_attrs)
        
        self.full_graph = G
        print(f"Created full graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def create_temporal_snapshots(self, interval: str = 'daily') -> Dict[str, nx.DiGraph]:
        """Create temporal snapshots of the network."""
        if not self.interactions:
            return {}
        
        # Get time range
        timestamps = [interaction['timestamp'] for interaction in self.interactions if interaction['timestamp']]
        if not timestamps:
            return {}
        
        start_time = min(timestamps)
        end_time = max(timestamps)
        
        # Determine interval delta
        if interval == 'daily':
            delta = timedelta(days=1)
        elif interval == 'weekly':
            delta = timedelta(weeks=1)
        elif interval == 'hourly':
            delta = timedelta(hours=1)
        else:
            delta = timedelta(days=1)
        
        # Create time intervals
        current_time = start_time
        intervals = []
        while current_time <= end_time:
            intervals.append(current_time)
            current_time += delta
        
        self.time_intervals = intervals
        
        # Create snapshot for each interval
        for i, interval_start in enumerate(intervals[:-1]):
            interval_end = intervals[i + 1]
            interval_key = interval_start.strftime('%Y-%m-%d_%H-%M')
            
            # Filter interactions for this time window
            interval_interactions = [
                interaction for interaction in self.interactions
                if interaction['timestamp'] and 
                interval_start <= interaction['timestamp'] < interval_end
            ]
            
            if interval_interactions:
                snapshot = self._create_snapshot_graph(interval_interactions, interval_start, interval_end)
                self.temporal_graphs[interval_key] = snapshot
        
        print(f"Created {len(self.temporal_graphs)} temporal snapshots")
        return self.temporal_graphs
    
    def _create_snapshot_graph(self, interactions: List[Dict], start_time: datetime, end_time: datetime) -> nx.DiGraph:
        """Create a graph snapshot for a specific time interval."""
        G = nx.DiGraph()
        
        # Get active users in this interval
        active_users = set()
        for interaction in interactions:
            active_users.add(interaction['source_user_id'])
            active_users.add(interaction['target_user_id'])
        
        # Add active users as nodes
        for user_id in active_users:
            if user_id in self.users:
                G.add_node(user_id, **self.users[user_id])
        
        # Track interactions
        edge_weights = defaultdict(float)
        edge_types = defaultdict(list)
        
        # Process interactions in this interval
        for interaction in interactions:
            source = interaction['source_user_id']
            target = interaction['target_user_id']
            
            if source and target and source in active_users and target in active_users:
                edge_key = (source, target)
                edge_weights[edge_key] += interaction['weight']
                edge_types[edge_key].append(interaction['edge_type'])
        
        # Add edges
        for (source, target), weight in edge_weights.items():
            types = edge_types[(source, target)]
            G.add_edge(source, target, 
                      weight=weight,
                      interaction_count=len(types),
                      edge_types=list(set(types)),
                      dominant_type=max(set(types), key=types.count))
        
        # Add graph metadata
        G.graph['start_time'] = start_time
        G.graph['end_time'] = end_time
        G.graph['interval'] = (end_time - start_time).total_seconds() / 3600  # hours
        
        return G
    
    def calculate_centrality_metrics(self, graph: nx.DiGraph = None) -> Dict[str, Dict]:
        """Calculate various centrality metrics for nodes."""
        if graph is None:
            graph = self.full_graph
        
        if graph is None or graph.number_of_nodes() == 0:
            return {}
        
        centrality_metrics = {}
        
        # In-degree centrality (mentions, replies received)
        in_degree_centrality = nx.in_degree_centrality(graph)
        
        # Out-degree centrality (mentions, replies sent)
        out_degree_centrality = nx.out_degree_centrality(graph)
        
        # PageRank (influence measure)
        try:
            pagerank = nx.pagerank(graph, weight='weight', max_iter=100)
        except:
            pagerank = nx.pagerank(graph, max_iter=100)
        
        # Betweenness centrality (bridge users)
        try:
            betweenness = nx.betweenness_centrality(graph, weight='weight')
        except:
            betweenness = {}
        
        # Combine metrics
        for node in graph.nodes():
            centrality_metrics[node] = {
                'in_degree_centrality': in_degree_centrality.get(node, 0),
                'out_degree_centrality': out_degree_centrality.get(node, 0),
                'pagerank': pagerank.get(node, 0),
                'betweenness_centrality': betweenness.get(node, 0),
                'in_degree': graph.in_degree(node),
                'out_degree': graph.out_degree(node),
                'total_degree': graph.degree(node)
            }
        
        return centrality_metrics
    
    def detect_communities(self, graph: nx.DiGraph = None) -> Dict[str, int]:
        """Detect communities using modularity-based algorithms."""
        if graph is None:
            graph = self.full_graph
        
        if graph is None or graph.number_of_nodes() == 0:
            return {}
        
        # Convert to undirected for community detection
        undirected_graph = graph.to_undirected()
        
        try:
            # Use greedy modularity communities
            communities = nx.community.greedy_modularity_communities(undirected_graph, weight='weight')
            
            # Create node to community mapping
            node_communities = {}
            for i, community in enumerate(communities):
                for node in community:
                    node_communities[node] = i
            
            print(f"Detected {len(communities)} communities")
            return node_communities
        
        except Exception as e:
            print(f"Community detection failed: {e}")
            return {}
    
    def get_influential_users(self, top_k: int = 10, metric: str = 'pagerank') -> List[Tuple[str, float]]:
        """Get top influential users based on centrality metric."""
        if self.full_graph is None:
            return []
        
        centrality_metrics = self.calculate_centrality_metrics()
        
        if not centrality_metrics:
            return []
        
        # Sort users by specified metric
        users_by_metric = [
            (user_id, metrics[metric]) 
            for user_id, metrics in centrality_metrics.items()
        ]
        users_by_metric.sort(key=lambda x: x[1], reverse=True)
        
        return users_by_metric[:top_k]
    
    def analyze_temporal_influence(self) -> pd.DataFrame:
        """Analyze how user influence changes over time."""
        if not self.temporal_graphs:
            return pd.DataFrame()
        
        influence_data = []
        
        for time_key, graph in self.temporal_graphs.items():
            centrality_metrics = self.calculate_centrality_metrics(graph)
            
            for user_id, metrics in centrality_metrics.items():
                user_info = self.users.get(user_id, {})
                influence_data.append({
                    'timestamp': time_key,
                    'user_id': user_id,
                    'username': user_info.get('username', ''),
                    'name': user_info.get('name', ''),
                    'followers_count': user_info.get('followers_count', 0),
                    'pagerank': metrics['pagerank'],
                    'in_degree_centrality': metrics['in_degree_centrality'],
                    'out_degree_centrality': metrics['out_degree_centrality'],
                    'betweenness_centrality': metrics['betweenness_centrality'],
                    'total_interactions': metrics['total_degree']
                })
        
        return pd.DataFrame(influence_data)
    
    def get_network_statistics(self, graph: nx.DiGraph = None) -> Dict:
        """Calculate basic network statistics."""
        if graph is None:
            graph = self.full_graph
        
        if graph is None:
            return {}
        
        stats = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'average_clustering': nx.average_clustering(graph.to_undirected()),
        }
        
        # Try to calculate additional metrics if graph is not too large
        if graph.number_of_nodes() < 1000:
            try:
                stats['average_shortest_path_length'] = nx.average_shortest_path_length(graph)
            except:
                stats['average_shortest_path_length'] = None
        
        # Degree statistics
        degrees = [d for n, d in graph.degree()]
        if degrees:
            stats['average_degree'] = np.mean(degrees)
            stats['max_degree'] = max(degrees)
            stats['min_degree'] = min(degrees)
        
        return stats
    
    def save_graphs(self, output_dir: str):
        """Save graphs to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full graph
        if self.full_graph:
            nx.write_gpickle(self.full_graph, f"{output_dir}/full_graph.gpickle")
        
        # Save temporal graphs
        if self.temporal_graphs:
            with open(f"{output_dir}/temporal_graphs.pickle", 'wb') as f:
                pickle.dump(self.temporal_graphs, f)
        
        print(f"Saved graphs to {output_dir}")
    
    def load_graphs(self, output_dir: str):
        """Load graphs from files."""
        import os
        
        # Load full graph
        full_graph_path = f"{output_dir}/full_graph.gpickle"
        if os.path.exists(full_graph_path):
            self.full_graph = nx.read_gpickle(full_graph_path)
        
        # Load temporal graphs
        temporal_graphs_path = f"{output_dir}/temporal_graphs.pickle"
        if os.path.exists(temporal_graphs_path):
            with open(temporal_graphs_path, 'rb') as f:
                self.temporal_graphs = pickle.load(f)
        
        print(f"Loaded graphs from {output_dir}")

if __name__ == "__main__":
    # Example usage
    from preprocess import TwitterDataPreprocessor
    
    # Load preprocessed data
    preprocessor = TwitterDataPreprocessor("../data/tweetdata-en.csv")
    users, interactions = preprocessor.process_data(filter_business=True)
    
    # Build graphs
    graph_builder = TemporalGraphBuilder(users, interactions)
    
    # Create full graph
    full_graph = graph_builder.create_full_graph()
    
    # Create temporal snapshots
    temporal_graphs = graph_builder.create_temporal_snapshots(interval='daily')
    
    # Calculate centrality metrics
    centrality_metrics = graph_builder.calculate_centrality_metrics()
    
    # Get influential users
    top_users = graph_builder.get_influential_users(top_k=10, metric='pagerank')
    print("\nTop 10 influential users (PageRank):")
    for i, (user_id, score) in enumerate(top_users, 1):
        user_info = users.get(user_id, {})
        print(f"{i}. {user_info.get('name', 'Unknown')} (@{user_info.get('username', user_id)}) - Score: {score:.4f}")
    
    # Detect communities
    communities = graph_builder.detect_communities()
    
    # Get network statistics
    stats = graph_builder.get_network_statistics()
    print(f"\nNetwork Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save graphs
    graph_builder.save_graphs("../data/graphs")