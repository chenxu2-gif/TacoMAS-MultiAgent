"""
System summarizer for generating compressed summaries for meta LLM.
"""

from typing import Dict, List, Optional
import statistics

from .schemas import (
    AgentState,
    EdgeSummary,
    NodeSummary,
    SubgraphSummary,
    SystemSummary,
    EdgeType,
)
from .graph_manager import GraphManager
from .population_manager import PopulationManager


class SystemSummarizer:
    """Generates system summaries for meta LLM decision making."""
    
    def __init__(
        self,
        graph_manager: GraphManager,
        population_manager: PopulationManager,
    ):
        """Initialize the summarizer."""
        self.graph = graph_manager
        self.population = population_manager
    
    def summarize(
        self,
        slow_update_step: int,
        fast_time_window: tuple,
        task_objective: str,
        current_answer_quality: float = 0.0,
    ) -> SystemSummary:
        """Generate a complete system summary.
        
        Args:
            slow_update_step: Current slow time step
            fast_time_window: (start_step, end_step) of fast time window
            task_objective: Description of the task
            current_answer_quality: Quality of current system answer
        
        Returns:
            Complete system summary
        """
        # Generate node summaries
        node_summaries = self._summarize_nodes()
        
        # Generate edge summaries
        edge_summaries = self._summarize_edges()
        
        # Generate subgraph summaries
        subgraph_summaries = self._summarize_subgraphs()
        
        # Calculate global statistics
        global_stats = self._calculate_global_stats(node_summaries)
        
        # Determine system state
        is_stable = self._is_system_stable(node_summaries)
        has_bottleneck = len(self.graph.get_bottleneck_nodes()) > 0
        learning_efficiency = self._calculate_learning_efficiency(node_summaries)
        
        graph_connectivity_score = self._calculate_graph_connectivity_score(edge_summaries)

        return SystemSummary(
            slow_update_step=slow_update_step,
            fast_time_window=fast_time_window,
            task_objective=task_objective,
            mean_score=global_stats["mean_score"],
            score_trend=global_stats["score_trend"],
            graph_density=self.graph.get_density(),
            modularity=self.graph.get_modularity(),
            bottleneck_count=len(self.graph.get_bottleneck_nodes()),
            diversity_index=self._calculate_diversity_index(),
            current_answer_quality=current_answer_quality,
            graph_connectivity_score=graph_connectivity_score,
            node_summaries=node_summaries,
            edge_summaries=edge_summaries,
            subgraph_summaries=subgraph_summaries,
            is_stable=is_stable,
            has_bottleneck=has_bottleneck,
            learning_efficiency=learning_efficiency,
            total_agents=self.population.get_population_size(),
            total_edges=len(self.graph.get_all_edges()),
        )
    
    def _summarize_nodes(self) -> List[NodeSummary]:
        """Generate summaries for all nodes."""
        summaries = []
        
        for agent in self.population.get_all_agents():
            summary = NodeSummary(
                agent_id=agent.agent_id,
                role=agent.role,
                recent_scores=agent.recent_scores[-10:],  # Last 10 scores
                score_trend=agent.score_trend,
                latest_output=agent.output[:200] if agent.output else "",  # Truncate
                current_improvement_direction=agent.improvement_direction,
                neighbor_externality=agent.neighbor_externality,
                redundancy_cluster=self._get_redundancy_cluster(agent.agent_id),
                memory_summary=agent.memory_summary[:150],  # Truncate
                failure_modes=agent.failure_modes[:3],  # Top 3 failure modes
                bridge_value=agent.bridge_value,
                community_id=agent.community_id,
                last_meta_score=agent.meta_score,
                workflow_correction=agent.workflow_correction[:200],
                next_round_workflow=agent.next_round_workflow[:200],
            )
            summaries.append(summary)
        
        return summaries
    
    def _summarize_edges(self) -> List[EdgeSummary]:
        """Generate summaries for all edges."""
        summaries = []

        for source, target, edge_type in self.graph.get_all_edges():
            # Calculate edge quality metrics
            quality = self._calculate_edge_quality(source, target)
            redundancy = self._calculate_edge_redundancy(source, target)
            conflict_rate = self._calculate_conflict_rate(source, target)
            bridge_edges = self.graph.get_bridge_edges()
            is_bridge = (source, target) in bridge_edges or (target, source) in bridge_edges
            
            summary = EdgeSummary(
                source=source,
                target=target,
                edge_type=edge_type,
                semantic_group=edge_type.semantic_group(),
                importance_weight=edge_type.importance_weight(),
                interaction_quality=quality,
                redundancy=redundancy,
                conflict_rate=conflict_rate,
                bridge_flag=is_bridge,
            )
            summaries.append(summary)
        
        return summaries
    
    def _summarize_subgraphs(self) -> List[SubgraphSummary]:
        """Generate summaries for subgraphs/communities."""
        summaries = []
        communities = self.graph.get_communities()
        
        for i, community in enumerate(communities):
            agents_in_community = [
                self.population.get_agent(aid) for aid in community
                if self.population.get_agent(aid) is not None
            ]
            
            if not agents_in_community:
                continue
            
            # Calculate community metrics
            scores = [
                s for a in agents_in_community
                for s in a.recent_scores
            ]
            quality_trend = (
                statistics.mean(scores[-5:]) - statistics.mean(scores[:5])
                if len(scores) >= 10 else 0.0
            )
            
            # Calculate internal density
            internal_edges = 0
            for aid1 in community:
                for aid2 in community:
                    if aid1 != aid2 and (aid1, aid2) in self.graph.edges:
                        internal_edges += 1
            
            max_internal_edges = len(community) * (len(community) - 1)
            internal_density = (
                internal_edges / max_internal_edges
                if max_internal_edges > 0 else 0.0
            )
            
            # Count external bridges
            external_bridges = 0
            for aid in community:
                for neighbor in self.graph.get_neighbors(aid):
                    if neighbor not in community:
                        external_bridges += 1
            
            # Identify bottleneck nodes
            bottleneck_nodes = [
                aid for aid in community
                if aid in self.graph.get_bottleneck_nodes()
            ]
            
            # Determine convergence status
            if quality_trend < -0.1:
                convergence_status = "stuck"
            elif quality_trend > 0.1:
                convergence_status = "learning"
            else:
                convergence_status = "converged"
            
            summary = SubgraphSummary(
                community_id=i,
                agent_ids=list(community),
                convergence_status=convergence_status,
                quality_trend=quality_trend,
                internal_density=internal_density,
                external_bridges=external_bridges,
                bottleneck_nodes=bottleneck_nodes,
            )
            summaries.append(summary)
        
        return summaries
    
    def _calculate_global_stats(self, node_summaries: List[NodeSummary]) -> Dict:
        """Calculate global statistics."""
        all_scores = []
        for summary in node_summaries:
            all_scores.extend(summary.recent_scores)
        
        if not all_scores:
            return {
                "mean_score": 0.0,
                "score_trend": 0.0,
            }
        
        mean_score = statistics.mean(all_scores)
        
        # Calculate trend
        if len(all_scores) >= 2:
            recent_mean = statistics.mean(all_scores[-len(all_scores)//2:])
            old_mean = statistics.mean(all_scores[:len(all_scores)//2])
            score_trend = recent_mean - old_mean
        else:
            score_trend = 0.0
        
        return {
            "mean_score": mean_score,
            "score_trend": score_trend,
        }
    
    def _calculate_diversity_index(self) -> float:
        """Calculate population diversity index."""
        role_dist = self.population.get_role_distribution()
        
        if not role_dist:
            return 0.0
        
        # Shannon entropy
        total = sum(role_dist.values())
        entropy = 0.0
        for count in role_dist.values():
            if count > 0:
                p = count / total
                entropy -= p * (p ** 0.5)  # Simplified entropy
        
        return entropy
    
    def _is_system_stable(self, node_summaries: List[NodeSummary]) -> bool:
        """Check if the system is stable."""
        if not node_summaries:
            return True
        
        # System is stable if most agents have low score trend
        stable_count = sum(
            1 for s in node_summaries
            if abs(s.score_trend) < 0.1
        )
        
        return stable_count / len(node_summaries) > 0.7
    
    def _calculate_learning_efficiency(self, node_summaries: List[NodeSummary]) -> float:
        """Calculate recent learning efficiency."""
        if not node_summaries:
            return 0.0
        
        positive_trends = sum(
            1 for s in node_summaries
            if s.score_trend > 0.05
        )
        
        return positive_trends / len(node_summaries)

    def _calculate_graph_connectivity_score(self, edge_summaries: List[EdgeSummary]) -> float:
        """Blend density, semantic edge quality, modularity, and bottlenecks into a [0,1] score."""
        density = self.graph.get_density()
        modularity = self.graph.get_modularity()
        bottlenecks = len(self.graph.get_bottleneck_nodes())
        total_agents = max(1, self.population.get_population_size())
        bottleneck_ratio = min(1.0, bottlenecks / total_agents)

        if edge_summaries:
            total_weight = sum(max(0.0, e.importance_weight) for e in edge_summaries) or 1.0
            avg_edge_quality = sum(
                e.interaction_quality * max(0.0, e.importance_weight)
                for e in edge_summaries
            ) / total_weight
            mainline_ratio = sum(
                1 for e in edge_summaries if e.edge_type.is_mainline()
            ) / len(edge_summaries)
            auxiliary_ratio = sum(
                1 for e in edge_summaries if e.edge_type.is_auxiliary()
            ) / len(edge_summaries)
            auxiliary_balance = max(0.0, 1.0 - auxiliary_ratio)
        else:
            avg_edge_quality = 0.0
            mainline_ratio = 0.0
            auxiliary_balance = 0.0

        # Modularity closer to 0 => more connected; high absolute modularity => fragmented.
        modularity_score = 1.0 - min(1.0, abs(modularity))
        bottleneck_score = 1.0 - bottleneck_ratio

        score = (
            density
            + avg_edge_quality
            + modularity_score
            + bottleneck_score
            + mainline_ratio
            + auxiliary_balance
        ) / 6.0
        return max(0.0, min(1.0, score))
    
    def _get_redundancy_cluster(self, agent_id: str) -> Optional[str]:
        """Get the redundancy cluster for an agent."""
        agent = self.population.get_agent(agent_id)
        if not agent:
            return None
        
        # Find agents with same role
        same_role_agents = self.population.get_agents_by_role(agent.role)
        
        if len(same_role_agents) > 1:
            return f"{agent.role}_cluster_{len(same_role_agents)}"
        
        return None
    
    def _calculate_edge_quality(self, source: str, target: str) -> float:
        """Calculate interaction quality for an edge."""
        source_agent = self.population.get_agent(source)
        target_agent = self.population.get_agent(target)
        
        if not source_agent or not target_agent:
            return 0.0
        
        # Simple heuristic: average of recent scores
        source_score = (
            statistics.mean(source_agent.recent_scores)
            if source_agent.recent_scores else 0.0
        )
        target_score = (
            statistics.mean(target_agent.recent_scores)
            if target_agent.recent_scores else 0.0
        )
        
        return (source_score + target_score) / 2
    
    def _calculate_edge_redundancy(self, source: str, target: str) -> float:
        """Calculate information redundancy for an edge."""
        source_agent = self.population.get_agent(source)
        target_agent = self.population.get_agent(target)
        
        if not source_agent or not target_agent:
            return 0.0
        
        # Simple heuristic: same role = high redundancy
        if source_agent.role == target_agent.role:
            return 0.8
        
        return 0.2
    
    def _calculate_conflict_rate(self, source: str, target: str) -> float:
        """Calculate conflict rate for an edge."""
        # This would require tracking actual debate/conflict history
        # For now, return a placeholder
        return 0.0
    
    def to_dict(self) -> Dict:
        """Convert summary to dictionary for logging."""
        return {
            "timestamp": "...",
            "total_agents": self.population.get_population_size(),
            "total_edges": len(self.graph.get_all_edges()),
            "graph_density": self.graph.get_density(),
            "modularity": self.graph.get_modularity(),
            "role_distribution": self.population.get_role_distribution(),
        }
