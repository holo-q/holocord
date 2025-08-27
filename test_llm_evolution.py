#!/usr/bin/env python3
"""
🧬 Test LLM Self-Evolution System
Demonstrates how agents can mutate their own DNA based on performance
"""

import time
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)

from config.production_setup import initialize_production_system, create_production_agent
from evolution.evolution_scheduler import evolution_scheduler, start_agent_evolution
from evolution.mutation_engine import mutation_engine
from emotion_engine.monitoring import production_monitor
from emotion_engine.dynamics import dynamics_processor, EnvironmentalInput
from emotion_engine.reflection import MetaReflector, ReflectionContext
from config.optimal_production import OPTIMAL_CONFIG


def demonstrate_llm_self_evolution():
    """Demonstrate the complete LLM self-evolution system"""
    
    print("🧬 LLM Self-Evolution System Demonstration")
    print("=" * 50)
    
    # Initialize production system
    print("\n1️⃣  Initializing production system with optimal configuration...")
    init_status = initialize_production_system()
    print(f"   ✅ System initialized: {init_status['status']}")
    
    # Create test agents with different initial configurations
    print("\n2️⃣  Creating test agents with suboptimal configurations...")
    
    agents = {}
    
    # Agent 1: Poor social decay rate (too high)
    agent1_id = "evolving_agent_1"
    runtime1 = create_production_agent(agent1_id, "claude-3-sonnet")
    # Deliberately set a poor social decay rate
    runtime1.genome.core.base_stats.social_energy_base = 0.3  # Too low
    agents[agent1_id] = runtime1
    print(f"   Created {agent1_id} with low social energy base: 0.3 (optimal: {OPTIMAL_CONFIG.social_energy_base:.3f})")
    
    # Agent 2: Poor confidence settings
    agent2_id = "evolving_agent_2" 
    runtime2 = create_production_agent(agent2_id, "claude-3-sonnet")
    runtime2.genome.core.base_stats.confidence_base = 0.8  # Too high
    agents[agent2_id] = runtime2
    print(f"   Created {agent2_id} with high confidence base: 0.8 (optimal: {OPTIMAL_CONFIG.confidence_base:.3f})")
    
    # Agent 3: Poor response threshold
    agent3_id = "evolving_agent_3"
    runtime3 = create_production_agent(agent3_id, "claude-3-sonnet")
    runtime3.genome.core.speak_threshold = 0.9  # Too high (won't respond much)
    agents[agent3_id] = runtime3
    print(f"   Created {agent3_id} with high response threshold: 0.9 (optimal: {OPTIMAL_CONFIG.response_threshold:.3f})")
    
    # Start evolution system
    print("\n3️⃣  Starting evolution scheduler...")
    evolution_scheduler.config.mutation_check_interval = 60.0  # Check every minute for demo
    evolution_scheduler.config.max_concurrent_mutations = 3    # Allow all to mutate
    
    for agent_id, runtime in agents.items():
        evolution_scheduler.register_agent(agent_id, runtime)
    
    start_agent_evolution()
    print(f"   ✅ Evolution scheduler started with {len(agents)} agents")
    
    # Simulate agent activity to build performance history
    print("\n4️⃣  Simulating agent activity to establish performance baselines...")
    
    reflector = MetaReflector()
    
    for iteration in range(25):  # Simulate 25 iterations
        for agent_id, runtime in agents.items():
            
            # Create environmental input with some variety
            env_input = EnvironmentalInput(
                conversation_novelty=0.2 + (iteration % 5) * 0.15,
                topic_expertise_match=0.3 + (iteration % 4) * 0.2,
                conversation_harmony=0.5,
                direct_mention=(iteration % 6 == 0),
                message_count_recent=iteration % 8 + 1
            )
            
            # Update emotional state
            old_state = runtime.current_state
            new_state = dynamics_processor.update_state(runtime, env_input, delta_time=45.0)
            runtime.current_state = new_state
            
            # Simulate reflection decisions
            context = ReflectionContext(
                conversation_history=[{"content": f"message {iteration}", "author": "user"}],
                directly_mentioned=(iteration % 6 == 0),
                conversation_energy=0.4 + (iteration % 3) * 0.2
            )
            
            reflection_result = reflector.perform_reflection(runtime, context)
        
        # Show progress every 5 iterations
        if (iteration + 1) % 5 == 0:
            print(f"   Iteration {iteration + 1}/25 completed")
    
    # Calculate initial performance for all agents
    print("\n5️⃣  Initial performance assessment:")
    initial_performance = {}
    for agent_id in agents:
        performance = production_monitor.calculate_agent_performance(agent_id)
        if performance:
            initial_performance[agent_id] = performance
            print(f"   {agent_id}: fitness={performance.fitness_score:.3f}, stability={performance.stability_score:.3f}, response_rate={performance.response_rate:.3f}")
        else:
            print(f"   {agent_id}: insufficient data for performance calculation")
    
    # Trigger evolution checks for all agents
    print("\n6️⃣  Triggering LLM self-evolution for all agents...")
    
    evolution_results = {}
    for agent_id in agents:
        print(f"\n   🧬 Processing evolution for {agent_id}...")
        
        # Force evolution check
        if evolution_scheduler.force_evolution_check(agent_id):
            evolution_results[agent_id] = "evolution_triggered"
            print(f"      ✅ Evolution process started")
        else:
            evolution_results[agent_id] = "no_evolution_needed"
            print(f"      ⚪ No evolution needed at this time")
    
    # Wait for mutations to be applied
    print(f"\n   Waiting for mutations to be applied and evaluated...")
    time.sleep(5)  # Give mutations time to be applied
    
    # Show mutation details
    print("\n7️⃣  Mutation details:")
    for agent_id in agents:
        if agent_id in mutation_engine.mutation_history:
            history = mutation_engine.mutation_history[agent_id]
            if history.mutations:
                latest_mutation = history.mutations[-1]
                print(f"   {agent_id}:")
                print(f"      Parameter: {latest_mutation['parameter']}")
                print(f"      Change: {latest_mutation['old_value']:.4f} → {latest_mutation['new_value']:.4f}")
                print(f"      Type: {latest_mutation['mutation_type']}")
                print(f"      Reasoning: {latest_mutation['reasoning']}")
        else:
            print(f"   {agent_id}: No mutations applied")
    
    # Simulate more activity to evaluate mutation effects
    print("\n8️⃣  Simulating post-mutation activity...")
    
    for iteration in range(15):  # Shorter post-mutation simulation
        for agent_id, runtime in agents.items():
            # Similar simulation as before
            env_input = EnvironmentalInput(
                conversation_novelty=0.3 + (iteration % 4) * 0.15,
                topic_expertise_match=0.4 + (iteration % 3) * 0.2,
                conversation_harmony=0.6,
                direct_mention=(iteration % 5 == 0),
                message_count_recent=iteration % 6 + 2
            )
            
            old_state = runtime.current_state
            new_state = dynamics_processor.update_state(runtime, env_input, delta_time=45.0)
            runtime.current_state = new_state
            
            context = ReflectionContext(
                conversation_history=[{"content": f"post-mutation message {iteration}", "author": "user"}],
                directly_mentioned=(iteration % 5 == 0),
                conversation_energy=0.5 + (iteration % 2) * 0.2
            )
            
            reflection_result = reflector.perform_reflection(runtime, context)
    
    # Calculate final performance
    print("\n9️⃣  Post-evolution performance assessment:")
    final_performance = {}
    
    for agent_id in agents:
        performance = production_monitor.calculate_agent_performance(agent_id)
        if performance:
            final_performance[agent_id] = performance
            
            # Compare with initial if available
            if agent_id in initial_performance:
                initial = initial_performance[agent_id]
                fitness_change = performance.fitness_score - initial.fitness_score
                stability_change = performance.stability_score - initial.stability_score
                response_change = performance.response_rate - initial.response_rate
                
                fitness_indicator = "📈" if fitness_change > 0.02 else "📉" if fitness_change < -0.02 else "➡️"
                
                print(f"   {agent_id}: fitness={performance.fitness_score:.3f} {fitness_indicator} ({fitness_change:+.3f})")
                print(f"      stability={performance.stability_score:.3f} ({stability_change:+.3f}), response_rate={performance.response_rate:.3f} ({response_change:+.3f})")
            else:
                print(f"   {agent_id}: fitness={performance.fitness_score:.3f}, stability={performance.stability_score:.3f}")
    
    # Evolution system status
    print("\n🔟 Evolution system status:")
    evolution_status = evolution_scheduler.get_evolution_status()
    print(f"   Total mutation checks: {evolution_status['statistics']['total_mutation_checks']}")
    print(f"   Mutations applied: {evolution_status['statistics']['mutations_applied']}")
    print(f"   Agents evolved: {len(evolution_status['statistics']['agents_evolved'])}")
    print(f"   Currently mutating: {evolution_status['agents_in_mutation']}")
    
    # Summary
    print("\n📋 DEMONSTRATION SUMMARY:")
    print("-" * 30)
    
    successful_evolutions = 0
    for agent_id in agents:
        if agent_id in final_performance and agent_id in initial_performance:
            improvement = final_performance[agent_id].fitness_score - initial_performance[agent_id].fitness_score
            if improvement > 0.02:
                successful_evolutions += 1
                print(f"✅ {agent_id}: Successfully evolved (improvement: {improvement:+.3f})")
            elif improvement < -0.02:
                print(f"❌ {agent_id}: Evolution may have hurt performance (change: {improvement:+.3f})")
            else:
                print(f"⚪ {agent_id}: Neutral evolution result (change: {improvement:+.3f})")
        else:
            print(f"ℹ️  {agent_id}: Insufficient data for comparison")
    
    print(f"\nOverall success rate: {successful_evolutions}/{len(agents)} agents improved")
    
    # Cleanup
    evolution_scheduler.stop_evolution_cycles()
    print("\n🧬 Evolution demonstration complete!")
    
    return {
        'agents_created': len(agents),
        'mutations_applied': evolution_status['statistics']['mutations_applied'],
        'successful_evolutions': successful_evolutions,
        'initial_performance': initial_performance,
        'final_performance': final_performance
    }


def show_mutation_engine_capabilities():
    """Show the different types of mutations the engine can generate"""
    
    print("\n🧬 MUTATION ENGINE CAPABILITIES")
    print("=" * 40)
    
    print("\n1️⃣  Mutation Types:")
    print("   • FINE_TUNE: Small adjustments to parameters (±2-10%)")
    print("   • ADAPTATION: Targeted fixes based on performance issues")
    print("   • EXPLORATION: Random parameter space exploration when safe")
    print("   • REVERSION: Return to previously successful configurations")
    print("   • DRIFT: Gradual parameter evolution based on experience")
    
    print("\n2️⃣  Parameter Sensitivity Awareness:")
    for param, settings in mutation_engine.mutation_sensitivity.items():
        print(f"   • {param}: max_change=±{settings['max_change']:.3f}, risk_multiplier={settings['risk_multiplier']:.1f}")
    
    print("\n3️⃣  LLM Decision Process:")
    print("   • Agent evaluates mutation candidates based on:")
    print("     - Expected benefit vs. risk factor")
    print("     - Current performance vs. targets") 
    print("     - Mutation type preferences (fine-tune > adapt > explore)")
    print("     - Historical success patterns")
    print("   • Agent chooses mutations using reasoning and some randomness")
    print("   • Mutations are applied gradually with performance monitoring")
    
    print("\n4️⃣  Safety Features:")
    print("   • Parameter bounds checking (0.0-1.0 for most parameters)")
    print("   • Mutation cooldown periods (5+ minutes between changes)")
    print("   • Limited concurrent mutations across agents")
    print("   • Performance-based mutation frequency adjustment")
    print("   • Automatic reversion for severely harmful mutations")


if __name__ == "__main__":
    # Show capabilities first
    show_mutation_engine_capabilities()
    
    # Run demonstration
    print("\n" + "="*60)
    results = demonstrate_llm_self_evolution()
    
    print(f"\n🎉 FINAL RESULTS:")
    print(f"   Agents created: {results['agents_created']}")
    print(f"   Mutations applied: {results['mutations_applied']}")
    print(f"   Successful evolutions: {results['successful_evolutions']}")
    
    if results['successful_evolutions'] > 0:
        print(f"   ✅ LLM self-evolution system working correctly!")
    else:
        print(f"   ⚠️  No clear improvements detected (may need longer evaluation)")
    
    print("\n🧬 The LLMs are now in control of their own evolution!")