"""
Network analysis module for ReviewGuard
Detects coordinated fake review rings using graph analysis
"""
import pandas as pd
import numpy as np
import networkx as nx
from config.settings import (
    PROCESSED_DATA_PATH, 
    NETWORK_FEATURES_PATH,
    MIN_COMMUNITY_SIZE,
    MAX_COMMUNITY_SIZE
)
from src.utils import setup_logger, print_section_header

logger = setup_logger(__name__)


def build_reviewer_network(input_path=PROCESSED_DATA_PATH, output_path=NETWORK_FEATURES_PATH):
    """
    Build reviewer network and extract network-based features
    
    Network creation:
    - Nodes: Reviewers
    - Edges: Reviewers who reviewed the same product
    - Edge weights: Number of common products reviewed
    
    Features extracted:
    - Degree centrality: How connected a reviewer is
    - Clustering coefficient: Tendency to form tight clusters
    - Betweenness centrality: Bridge between communities
    - PageRank: Importance in network
    
    Args:
        input_path: Path to processed reviews CSV
        output_path: Path to save network features CSV
    
    Returns:
        pd.DataFrame: Network features for each reviewer
    """
    print_section_header("NETWORK ANALYSIS")
    logger.info(f"Loading processed reviews from {input_path}")
    
    try:
        # Load data
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df):,} reviews from {df['reviewerID'].nunique():,} reviewers")
        
        # Build graph
        logger.info("Building reviewer network graph...")
        G = nx.Graph()
        
        # Add all reviewers as nodes
        reviewers = df["reviewerID"].unique()
        G.add_nodes_from(reviewers)
        logger.info(f"Added {len(reviewers):,} reviewer nodes")
        
        # Create edges between reviewers who reviewed same products
        # Edge weight = number of common products
        logger.info("Creating edges based on co-reviewed products...")
        
        edge_weights = {}
        product_groups = df.groupby("productID")["reviewerID"].apply(list)
        
        for product_id, reviewer_list in product_groups.items():
            # For each pair of reviewers who reviewed this product
            for i in range(len(reviewer_list)):
                for j in range(i + 1, len(reviewer_list)):
                    reviewer1, reviewer2 = reviewer_list[i], reviewer_list[j]
                    edge = tuple(sorted([reviewer1, reviewer2]))
                    edge_weights[edge] = edge_weights.get(edge, 0) + 1
        
        # Add weighted edges to graph
        for (node1, node2), weight in edge_weights.items():
            G.add_edge(node1, node2, weight=weight)
        
        logger.info(f"Created {G.number_of_edges():,} edges")
        
        # Calculate network metrics
        logger.info("Calculating network metrics...")
        
        # Degree centrality (normalized)
        degree_centrality = nx.degree_centrality(G)
        
        # Clustering coefficient
        clustering = nx.clustering(G)
        
        # Betweenness centrality (computationally expensive for large graphs)
        # Sample if graph is too large
        if G.number_of_nodes() > 5000:
            logger.info("Large graph detected. Sampling for betweenness centrality...")
            sample_nodes = np.random.choice(list(G.nodes()), size=5000, replace=False)
            betweenness = nx.betweenness_centrality(G, k=5000)
        else:
            betweenness = nx.betweenness_centrality(G)
        
        # PageRank
        pagerank = nx.pagerank(G, max_iter=100)
        
        # Create features dataframe
        network_features = []
        for node in G.nodes():
            network_features.append({
                'reviewerID': node,
                'degree_centrality': degree_centrality.get(node, 0),
                'clustering_coefficient': clustering.get(node, 0),
                'betweenness_centrality': betweenness.get(node, 0),
                'pagerank': pagerank.get(node, 0),
                'num_connections': G.degree(node),
            })
        
        network_df = pd.DataFrame(network_features)
        logger.info(f"Extracted network features for {len(network_df):,} reviewers")
        
        # Detect communities (potential review rings)
        logger.info("Detecting reviewer communities...")
        try:
            communities = nx.community.greedy_modularity_communities(G)
            logger.info(f"Found {len(communities)} communities")
            
            # Analyze communities
            suspicious_communities = []
            for i, community in enumerate(communities):
                size = len(community)
                if MIN_COMMUNITY_SIZE <= size <= MAX_COMMUNITY_SIZE:
                    suspicious_communities.append(community)
                    
                    # Calculate community statistics
                    subgraph = G.subgraph(community)
                    density = nx.density(subgraph)
                    logger.info(
                        f"  Suspicious community {i+1}: "
                        f"{size} members, density: {density:.3f}"
                    )
            
            # Mark reviewers in suspicious communities
            suspicious_reviewer_ids = set()
            for community in suspicious_communities:
                suspicious_reviewer_ids.update(community)
            
            network_df['in_suspicious_community'] = network_df['reviewerID'].isin(
                suspicious_reviewer_ids
            ).astype(int)
            
            logger.info(
                f"Flagged {len(suspicious_reviewer_ids):,} reviewers in "
                f"{len(suspicious_communities)} suspicious communities"
            )
            
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            network_df['in_suspicious_community'] = 0
        
        # Calculate additional derived features
        # High clustering + high connections = suspicious
        network_df['network_suspicion_score'] = (
            network_df['clustering_coefficient'] * 
            np.log1p(network_df['num_connections'])
        )
        
        # Normalize network suspicion score
        max_score = network_df['network_suspicion_score'].max()
        if max_score > 0:
            network_df['network_suspicion_score'] = (
                network_df['network_suspicion_score'] / max_score
            )
        
        # Save network features
        network_df.to_csv(output_path, index=False)
        logger.info(f"Saved network features to {output_path}")
        
        # Print summary
        print(f"\n✅ Network analysis completed successfully")
        print(f"   Graph statistics:")
        print(f"     - Nodes (reviewers): {G.number_of_nodes():,}")
        print(f"     - Edges (connections): {G.number_of_edges():,}")
        print(f"     - Avg degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        print(f"     - Density: {nx.density(G):.6f}")
        print(f"     - Communities found: {len(communities) if 'communities' in locals() else 'N/A'}")
        print(f"     - Suspicious communities: {len(suspicious_communities) if 'suspicious_communities' in locals() else 0}")
        print(f"\n   Network features:")
        print(f"     - Avg clustering coefficient: {network_df['clustering_coefficient'].mean():.4f}")
        print(f"     - Max connections: {network_df['num_connections'].max()}")
        print(f"     - High suspicion score (>0.5): {(network_df['network_suspicion_score'] > 0.5).sum():,}")
        
        return network_df
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        raise
    except Exception as e:
        logger.error(f"Error during network analysis: {e}")
        raise


if __name__ == "__main__":
    build_reviewer_network()