"""
Graph manager for mixed-type graph (bidirectional and directed edges).
"""

from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import networkx as nx

from .schemas import EdgeType, EdgeSummary


class GraphManager:
    """Manages the mixed-type graph structure."""
    
    def __init__(self):
        """Initialize the graph manager."""
        self.nodes: Dict[str, Dict] = {}  # agent_id -> node_data
        self.edges: Dict[Tuple[str, str], EdgeType] = {}  # (source, target) -> edge_type
        self.edge_weights: Dict[Tuple[str, str], float] = {}  # (source, target) -> weight
        
        # Adjacency lists for fast lookup
        self.outgoing: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> set of targets
        self.incoming: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> set of sources
        
        # For bidirectional edges, maintain both directions
        self.bidirectional_pairs: Set[Tuple[str, str]] = set()
    
    def add_node(self, agent_id: str, role: str, **metadata) -> None:
        """Add a node to the graph."""
        self.nodes[agent_id] = {
            "role": role,
            "metadata": metadata,
        }
    
    def remove_node(self, agent_id: str) -> None:
        """Remove a node and all its edges."""
        if agent_id not in self.nodes:
            return
        
        # Remove all edges involving this node
        edges_to_remove = []
        for (source, target) in self.edges.keys():
            if source == agent_id or target == agent_id:
                edges_to_remove.append((source, target))
        
        for edge in edges_to_remove:
            self.remove_edge(edge[0], edge[1])
        
        # Remove node
        del self.nodes[agent_id]
        self.outgoing.pop(agent_id, None)
        self.incoming.pop(agent_id, None)
    
    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType = EdgeType.DIRECTED,
        weight: float = 1.0,
    ) -> None:
        """Add an edge to the graph."""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"One or both nodes not in graph: {source}, {target}")
        
        edge_type = EdgeType.from_value(edge_type)

        if edge_type == EdgeType.BIDIRECTIONAL:
            # For bidirectional, add both directions
            self.edges[(source, target)] = EdgeType.BIDIRECTIONAL
            self.edges[(target, source)] = EdgeType.BIDIRECTIONAL
            self.outgoing[source].add(target)
            self.outgoing[target].add(source)
            self.incoming[source].add(target)
            self.incoming[target].add(source)
            self.bidirectional_pairs.add((min(source, target), max(source, target)))
            self.edge_weights[(source, target)] = weight
            self.edge_weights[(target, source)] = weight
        else:
            # For one-way semantic edges, add only one direction and preserve type.
            self.edges[(source, target)] = edge_type
            self.outgoing[source].add(target)
            self.incoming[target].add(source)
            self.edge_weights[(source, target)] = weight
    
    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from the graph."""
        if (source, target) not in self.edges:
            return
        
        edge_type = self.edges[(source, target)]
        
        if edge_type == EdgeType.BIDIRECTIONAL:
            # Remove both directions
            del self.edges[(source, target)]
            del self.edges[(target, source)]
            self.outgoing[source].discard(target)
            self.outgoing[target].discard(source)
            self.incoming[source].discard(target)
            self.incoming[target].discard(source)
            self.bidirectional_pairs.discard((min(source, target), max(source, target)))
            self.edge_weights.pop((source, target), None)
            self.edge_weights.pop((target, source), None)
        else:
            # Remove only one direction
            del self.edges[(source, target)]
            self.outgoing[source].discard(target)
            self.incoming[target].discard(source)
            self.edge_weights.pop((source, target), None)
    
    def change_edge_type(self, source: str, target: str, new_type: EdgeType) -> None:
        """Change the type of an existing edge."""
        if (source, target) not in self.edges:
            raise ValueError(f"Edge ({source}, {target}) does not exist")
        
        old_type = self.edges[(source, target)]
        if old_type == new_type:
            return
        
        weight = self.edge_weights.get((source, target), 1.0)
        self.remove_edge(source, target)
        self.add_edge(source, target, new_type, weight)
    
    def get_neighbors(self, agent_id: str, direction: str = "both") -> List[str]:
        """Get neighbors of an agent.
        
        Args:
            agent_id: The agent ID
            direction: "in", "out", or "both"
        """
        if agent_id not in self.nodes:
            return []
        
        if direction == "in":
            return list(self.incoming.get(agent_id, set()))
        elif direction == "out":
            return list(self.outgoing.get(agent_id, set()))
        else:  # both
            return list(self.incoming.get(agent_id, set()) | self.outgoing.get(agent_id, set()))
    
    def get_edge_type(self, source: str, target: str) -> Optional[EdgeType]:
        """Get the type of an edge."""
        return self.edges.get((source, target))
    
    def get_all_nodes(self) -> List[str]:
        """Get all node IDs."""
        return list(self.nodes.keys())
    
    def get_all_edges(self) -> List[Tuple[str, str, EdgeType]]:
        """Get all edges as (source, target, type) tuples."""
        seen = set()
        edges = []
        for (source, target), edge_type in self.edges.items():
            if edge_type == EdgeType.BIDIRECTIONAL:
                pair = (min(source, target), max(source, target), edge_type.value)
                if pair in seen:
                    continue
                seen.add(pair)
            edges.append((source, target, edge_type))
        return edges
    
    def get_node_degree(self, agent_id: str) -> int:
        """Get the degree of a node."""
        return len(self.get_neighbors(agent_id))
    
    def get_in_degree(self, agent_id: str) -> int:
        """Get the in-degree of a node."""
        return len(self.incoming.get(agent_id, set()))
    
    def get_out_degree(self, agent_id: str) -> int:
        """Get the out-degree of a node."""
        return len(self.outgoing.get(agent_id, set()))
    
    def is_connected(self) -> bool:
        """Check if the graph is connected (weakly for directed graphs)."""
        if not self.nodes:
            return True
        
        # Build networkx graph for connectivity check
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes.keys())
        for (source, target) in self.edges.keys():
            G.add_edge(source, target)
        
        return nx.is_weakly_connected(G)
    
    def get_connected_components(self) -> List[Set[str]]:
        """Get connected components."""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes.keys())
        for (source, target) in self.edges.keys():
            G.add_edge(source, target)
        
        return [set(c) for c in nx.weakly_connected_components(G)]
    
    def get_communities(self) -> List[Set[str]]:
        """Detect communities using modularity optimization."""
        if not self.nodes:
            return []
        
        # Convert to undirected for community detection
        G = nx.Graph()
        G.add_nodes_from(self.nodes.keys())
        for (source, target) in self.edges.keys():
            G.add_edge(source, target)
        
        try:
            from networkx.algorithms import community
            communities = list(community.greedy_modularity_communities(G))
            return [set(c) for c in communities]
        except:
            # Fallback to connected components
            return self.get_connected_components()
    
    def get_modularity(self) -> float:
        """Calculate graph modularity."""
        if not self.nodes or not self.edges:
            return 0.0
        
        G = nx.Graph()
        G.add_nodes_from(self.nodes.keys())
        for (source, target) in self.edges.keys():
            G.add_edge(source, target)
        
        try:
            from networkx.algorithms import community
            communities = list(community.greedy_modularity_communities(G))
            return community.modularity(G, communities)
        except:
            return 0.0
    
    def get_density(self) -> float:
        """Calculate graph density."""
        if not self.nodes:
            return 0.0
        
        n = len(self.nodes)
        if n <= 1:
            return 0.0
        
        # Count unique edges (treating bidirectional as one edge)
        unique_edges = set()
        for (source, target) in self.edges.keys():
            pair = (min(source, target), max(source, target))
            unique_edges.add(pair)
        
        max_edges = n * (n - 1) / 2
        return len(unique_edges) / max_edges if max_edges > 0 else 0.0
    
    def get_bottleneck_nodes(self, threshold: float = 0.5) -> List[str]:
        """Identify bottleneck nodes (high betweenness centrality)."""
        if not self.nodes:
            return []
        
        G = nx.Graph()
        G.add_nodes_from(self.nodes.keys())
        for (source, target) in self.edges.keys():
            G.add_edge(source, target)
        
        try:
            centrality = nx.betweenness_centrality(G)
            mean_centrality = sum(centrality.values()) / len(centrality) if centrality else 0
            threshold_value = mean_centrality * threshold
            return [node for node, c in centrality.items() if c > threshold_value]
        except:
            return []
    
    def get_bridge_edges(self) -> List[Tuple[str, str]]:
        """Get bridge edges (edges whose removal disconnects the graph)."""
        if not self.nodes:
            return []
        
        G = nx.Graph()
        G.add_nodes_from(self.nodes.keys())
        for (source, target) in self.edges.keys():
            G.add_edge(source, target)
        
        try:
            return list(nx.bridges(G))
        except:
            return []
    
    def get_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Get shortest path between two nodes."""
        if source not in self.nodes or target not in self.nodes:
            return None
        
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes.keys())
        for (s, t) in self.edges.keys():
            G.add_edge(s, t)
        
        try:
            return nx.shortest_path(G, source, target)
        except nx.NetworkXNoPath:
            return None
    
    def to_dict(self) -> Dict:
        """Convert graph to dictionary representation."""
        return {
            "nodes": list(self.nodes.keys()),
            "edges": [
                {
                    "source": s,
                    "target": t,
                    "type": self.edges[(s, t)].value,
                    "weight": self.edge_weights.get((s, t), 1.0),
                }
                for s, t in self.edges.keys()
            ],
        }
    
    def from_dict(self, data: Dict) -> None:
        """Load graph from dictionary representation."""
        self.nodes.clear()
        self.edges.clear()
        self.edge_weights.clear()
        self.outgoing.clear()
        self.incoming.clear()
        self.bidirectional_pairs.clear()
        
        for node_id in data.get("nodes", []):
            self.add_node(node_id, "unknown")
        
        for edge_data in data.get("edges", []):
            edge_type = EdgeType(edge_data["type"])
            weight = edge_data.get("weight", 1.0)
            self.add_edge(edge_data["source"], edge_data["target"], edge_type, weight)
