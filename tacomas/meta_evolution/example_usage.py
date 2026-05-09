"""
Example usage of the Meta-LLM driven Birth-Death Graph Co-Evolution system.
"""

import logging
from tacomas.config.llm import LLMConfig
from tacomas.config.meta_evolution import MetaEvolutionConfig
from tacomas.meta_evolution import (
    EvolutionController,
    AgentState,
    EdgeType,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Basic example of using the evolution controller."""
    
    # 1. Configure LLM for meta LLM
    llm_config = LLMConfig(
        model="gpt-4-turbo",
        temperature=0.7,
        api_key="your-api-key",  # Set via environment variable
    )
    
    # 2. Configure meta evolution system
    evolution_config = MetaEvolutionConfig(
        enabled=True,
        meta_llm_model="gpt-4-turbo",
        meta_llm_temperature=0.7,
        
        # Time scales
        fast_steps_per_window=10,
        bd_check_interval=50,
        graph_rewire_interval=200,
        
        # Population control
        n_min=5,
        n_max=15,
        specialization_level="medium",
        
        # Evolution control
        max_birth_death_pairs=2,
        max_edge_edits=5,
        protected_nodes=["planner", "summarizer"],
        critical_roles=["planner", "proposer", "verifier"],
        
        # Logging
        log_decisions=True,
        log_dir="./meta_evolution_logs",
    )
    
    # 3. Initialize evolution controller
    controller = EvolutionController(evolution_config, llm_config)
    
    # 4. Initialize agents
    agents = [
        AgentState(
            agent_id="planner",
            role="planner",
            policy="Break down the task into subproblems",
            output="",
            memory_summary="",
            capability={"planning": 0.8, "reasoning": 0.7},
            recent_scores=[0.7, 0.75, 0.8],
            score_trend=0.05,
            improvement_direction="Focus on clearer decomposition",
            neighbor_externality=0.3,
            redundancy_score=0.0,
            failure_modes=["Incomplete decomposition"],
            bridge_value=0.8,
            community_id=0,
            creation_time=0,
        ),
        AgentState(
            agent_id="proposer",
            role="proposer",
            policy="Generate candidate solutions",
            output="",
            memory_summary="",
            capability={"creativity": 0.8, "reasoning": 0.6},
            recent_scores=[0.6, 0.65, 0.7],
            score_trend=0.05,
            improvement_direction="Increase solution diversity",
            neighbor_externality=0.2,
            redundancy_score=0.1,
            failure_modes=["Repetitive solutions"],
            bridge_value=0.5,
            community_id=0,
            creation_time=0,
        ),
        AgentState(
            agent_id="verifier",
            role="verifier",
            policy="Check solution validity",
            output="",
            memory_summary="",
            capability={"verification": 0.9, "reasoning": 0.8},
            recent_scores=[0.8, 0.82, 0.85],
            score_trend=0.025,
            improvement_direction="Reduce false positives",
            neighbor_externality=0.4,
            redundancy_score=0.0,
            failure_modes=["Missed edge cases"],
            bridge_value=0.6,
            community_id=0,
            creation_time=0,
        ),
    ]
    
    for agent in agents:
        controller.add_agent(agent)
    
    # 5. Initialize graph structure
    controller.add_edge("planner", "proposer", "directed")
    controller.add_edge("proposer", "verifier", "bidirectional")
    controller.add_edge("verifier", "planner", "directed")
    
    logger.info(f"Initialized system with {len(agents)} agents")
    logger.info(f"System state: {controller.get_system_state()}")
    
    # 6. Simulate fast-time steps
    for step in range(100):
        controller.step_fast_time()
        
        # Simulate agent updates
        for agent_id in controller.get_agent_ids():
            agent = controller.population.get_agent(agent_id)
            if agent:
                # Simulate score improvement
                new_score = agent.recent_scores[-1] + 0.01 if agent.recent_scores else 0.5
                agent.recent_scores.append(new_score)
                agent.score_trend = (new_score - agent.recent_scores[0]) / len(agent.recent_scores)
                controller.update_agent_state(agent)
        
        # Check if slow update should be triggered
        if controller.should_trigger_slow_update():
            logger.info(f"Triggering slow update at fast step {step}")
            controller.execute_slow_update(
                task_description="Solve complex reasoning tasks",
                current_answer_quality=0.75,
            )
    
    logger.info(f"Final system state: {controller.get_system_state()}")


def example_with_custom_config():
    """Example with custom configuration."""
    
    # Custom configuration for high-specialization system
    evolution_config = MetaEvolutionConfig(
        enabled=True,
        meta_llm_model="gpt-4-turbo",
        
        # Aggressive evolution
        fast_steps_per_window=5,
        bd_check_interval=30,
        graph_rewire_interval=100,
        
        # Larger population
        n_min=8,
        n_max=20,
        specialization_level="high",
        
        # More aggressive evolution
        max_birth_death_pairs=3,
        max_edge_edits=8,
        
        # Protect core roles
        protected_nodes=["planner", "summarizer", "verifier"],
        critical_roles=["planner", "proposer", "debater", "verifier", "summarizer"],
    )
    
    llm_config = LLMConfig(model="gpt-4-turbo")
    controller = EvolutionController(evolution_config, llm_config)
    
    logger.info(f"Created controller with custom config")
    logger.info(f"Population bounds: [{evolution_config.n_min}, {evolution_config.n_max}]")
    logger.info(f"Critical roles: {evolution_config.critical_roles}")


if __name__ == "__main__":
    logger.info("Starting Meta-LLM Evolution System Example")
    
    # Run basic example
    try:
        example_basic_usage()
    except Exception as e:
        logger.error(f"Error in basic usage example: {e}")
    
    # Show custom config example
    try:
        example_with_custom_config()
    except Exception as e:
        logger.error(f"Error in custom config example: {e}")
