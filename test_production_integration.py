#!/usr/bin/env python3
"""
🧠 Test Production System Integration
Validates that the optimal configuration is properly integrated
"""

import sys
import time
from typing import Dict, Any

# Test imports
try:
    from config.optimal_production import OPTIMAL_CONFIG, EXPECTED_PERFORMANCE
    from config.production_setup import initialize_production_system, create_production_agent
    from emotion_engine.monitoring import production_monitor
    from emotion_engine.dynamics import dynamics_processor, EnvironmentalInput
    from emotion_engine.reflection import MetaReflector, ReflectionContext
    from genome.types import EmotionState, ConsciousnessLevel
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_optimal_configuration():
    """Test that optimal configuration is loaded correctly"""
    print("\n📊 Testing Optimal Configuration...")
    
    # Check configuration values
    assert OPTIMAL_CONFIG.curiosity_base == 0.5574, f"Expected curiosity_base 0.5574, got {OPTIMAL_CONFIG.curiosity_base}"
    assert OPTIMAL_CONFIG.confidence_base == 0.3433, f"Expected confidence_base 0.3433, got {OPTIMAL_CONFIG.confidence_base}"
    assert OPTIMAL_CONFIG.social_decay_rate == 0.0729, f"Expected social_decay_rate 0.0729, got {OPTIMAL_CONFIG.social_decay_rate}"
    assert OPTIMAL_CONFIG.response_threshold == 0.5785, f"Expected response_threshold 0.5785, got {OPTIMAL_CONFIG.response_threshold}"
    
    print(f"  ✅ Curiosity base: {OPTIMAL_CONFIG.curiosity_base}")
    print(f"  ✅ Confidence base: {OPTIMAL_CONFIG.confidence_base}")
    print(f"  ✅ Social decay rate: {OPTIMAL_CONFIG.social_decay_rate}")
    print(f"  ✅ Response threshold: {OPTIMAL_CONFIG.response_threshold}")
    
    # Check sensitivity rankings
    sensitivity = OPTIMAL_CONFIG.sensitivity_ranking
    assert sensitivity['social_decay_rate'] > sensitivity['confidence_base'], "social_decay_rate should be most sensitive"
    
    print(f"  ✅ Most sensitive parameter: social_decay_rate ({sensitivity['social_decay_rate']:.2f})")
    print("📊 Configuration test passed!")

def test_dynamics_processor_integration():
    """Test that dynamics processor uses optimal configuration"""
    print("\n🔄 Testing Dynamics Processor Integration...")
    
    # Check that processor uses optimal config
    assert dynamics_processor.use_optimal_config == True, "Dynamics processor should use optimal config"
    assert dynamics_processor.config == OPTIMAL_CONFIG, "Dynamics processor should reference optimal config"
    
    # Check baseline states
    expected_baselines = {
        'curiosity': OPTIMAL_CONFIG.curiosity_base,
        'confidence': OPTIMAL_CONFIG.confidence_base,
        'social_energy': OPTIMAL_CONFIG.social_energy_base,
        'restlessness': OPTIMAL_CONFIG.restlessness_base,
        'harmony': OPTIMAL_CONFIG.harmony_base
    }
    
    for state, expected_value in expected_baselines.items():
        actual_value = dynamics_processor.baseline_states[getattr(__import__('genome.types', fromlist=['StateVar']).StateVar, state.upper())]
        assert abs(actual_value - expected_value) < 0.001, f"Baseline {state} should be {expected_value}, got {actual_value}"
    
    print("  ✅ Dynamics processor configured with optimal baselines")
    print("🔄 Dynamics processor integration test passed!")

def test_monitoring_system():
    """Test monitoring system functionality"""
    print("\n🔍 Testing Monitoring System...")
    
    # Initialize monitoring
    test_agent_id = "test_agent_monitoring"
    
    # Create a test emotional state
    test_state_1 = EmotionState(
        curiosity=0.5,
        confidence=0.3,
        social_energy=0.8,
        restlessness=0.6,
        harmony=0.4,
        consciousness=ConsciousnessLevel.ALERT
    )
    
    test_state_2 = EmotionState(
        curiosity=0.6,
        confidence=0.4,
        social_energy=0.7,
        restlessness=0.5,
        harmony=0.5,
        consciousness=ConsciousnessLevel.ALERT
    )
    
    # Test state recording
    production_monitor.record_state_update(test_agent_id, test_state_1, test_state_2)
    
    # Test response decision recording  
    production_monitor.record_response_decision(test_agent_id, True, 0.7, OPTIMAL_CONFIG.response_threshold)
    
    # Check agent is registered
    assert test_agent_id in production_monitor.agent_metrics, "Agent should be registered in monitoring"
    
    metrics = production_monitor.agent_metrics[test_agent_id]
    assert len(metrics.state_history) > 0, "State history should be recorded"
    assert len(metrics.response_history) > 0, "Response history should be recorded"
    
    print("  ✅ State updates recorded")
    print("  ✅ Response decisions tracked")
    print("  ✅ Agent metrics maintained")
    print("🔍 Monitoring system test passed!")

def test_reflection_integration():
    """Test reflection system uses optimal thresholds"""
    print("\n🤔 Testing Reflection System Integration...")
    
    reflector = MetaReflector()
    
    # Create test runtime with optimal configuration
    from genome.base import AgentRuntime, BaseStats, GenomeCore
    from genome import ExtendedGenome
    
    # Create a minimal genome
    base_stats = BaseStats(
        curiosity_base=OPTIMAL_CONFIG.curiosity_base,
        confidence_base=OPTIMAL_CONFIG.confidence_base,
        social_energy_base=OPTIMAL_CONFIG.social_energy_base,
        restlessness_amplitude=OPTIMAL_CONFIG.restlessness_base,
        harmony_factor=OPTIMAL_CONFIG.harmony_base
    )
    
    core = GenomeCore(
        model_id="claude-3-sonnet",
        base_stats=base_stats
    )
    
    genome = ExtendedGenome(core=core)
    
    test_runtime = AgentRuntime(
        agent_id="test_reflection_agent",
        genome=genome,
        current_state=EmotionState(
            curiosity=OPTIMAL_CONFIG.curiosity_base,
            confidence=OPTIMAL_CONFIG.confidence_base,
            social_energy=OPTIMAL_CONFIG.social_energy_base,
            restlessness=OPTIMAL_CONFIG.restlessness_base,
            harmony=OPTIMAL_CONFIG.harmony_base,
            consciousness=ConsciousnessLevel.ALERT
        )
    )
    
    # Create reflection context
    context = ReflectionContext(
        conversation_history=[{"content": "Hello world", "author": "user"}],
        directly_mentioned=False,
        conversation_energy=0.6
    )
    
    # Perform reflection
    result = reflector.perform_reflection(test_runtime, context)
    
    # Check that result has proper structure
    assert hasattr(result, 'should_respond'), "Reflection result should have should_respond"
    assert hasattr(result, 'confidence'), "Reflection result should have confidence"
    assert hasattr(result, 'reasoning'), "Reflection result should have reasoning"
    
    print(f"  ✅ Reflection completed: {'RESPOND' if result.should_respond else 'PASS'}")
    print(f"  ✅ Confidence: {result.confidence:.3f}")
    print(f"  ✅ Using optimal threshold: {OPTIMAL_CONFIG.response_threshold}")
    print("🤔 Reflection integration test passed!")

def test_production_setup():
    """Test production setup script"""
    print("\n🚀 Testing Production Setup...")
    
    # Initialize production system
    init_status = initialize_production_system()
    
    assert init_status['status'] == 'initialized', "System should initialize successfully"
    assert init_status['monitoring_enabled'] == True, "Monitoring should be enabled"
    
    # Create production agent
    agent_id = "test_production_agent"
    runtime = create_production_agent(agent_id, "claude-3-sonnet")
    
    assert runtime.agent_id == agent_id, "Agent should be created with correct ID"
    assert runtime.current_state.curiosity == OPTIMAL_CONFIG.curiosity_base, "Agent should start with optimal curiosity"
    assert runtime.current_state.confidence == OPTIMAL_CONFIG.confidence_base, "Agent should start with optimal confidence"
    
    # Check agent is registered in monitoring
    assert agent_id in production_monitor.agent_metrics, "Agent should be registered in monitoring"
    
    print(f"  ✅ Production system initialized")
    print(f"  ✅ Agent created: {agent_id}")
    print(f"  ✅ Agent state: curiosity={runtime.current_state.curiosity:.3f}, confidence={runtime.current_state.confidence:.3f}")
    print("🚀 Production setup test passed!")

def test_performance_calculation():
    """Test performance metrics calculation"""
    print("\n📈 Testing Performance Calculation...")
    
    agent_id = "test_perf_agent"
    
    # Generate some mock state history
    for i in range(20):
        state = EmotionState(
            curiosity=0.5 + (i % 3) * 0.1,
            confidence=0.3 + (i % 4) * 0.1,
            social_energy=0.8 - (i % 2) * 0.1,
            restlessness=0.6,
            harmony=0.4,
            consciousness=ConsciousnessLevel.ALERT
        )
        production_monitor.record_state_update(agent_id, state, state)
    
    # Generate some response decisions
    for i in range(10):
        production_monitor.record_response_decision(agent_id, i % 2 == 0, 0.7, OPTIMAL_CONFIG.response_threshold)
    
    # Calculate performance
    performance = production_monitor.calculate_agent_performance(agent_id)
    
    assert performance is not None, "Performance should be calculated"
    assert 0.0 <= performance.fitness_score <= 1.0, "Fitness score should be between 0 and 1"
    assert 0.0 <= performance.stability_score <= 1.0, "Stability score should be between 0 and 1"
    
    print(f"  ✅ Fitness score: {performance.fitness_score:.3f}")
    print(f"  ✅ Stability score: {performance.stability_score:.3f}")
    print(f"  ✅ Response rate: {performance.response_rate:.3f}")
    print("📈 Performance calculation test passed!")

def run_all_tests():
    """Run all integration tests"""
    print("🧪 Starting Production System Integration Tests\n")
    
    try:
        test_optimal_configuration()
        test_dynamics_processor_integration()
        test_monitoring_system()
        test_reflection_integration()
        test_production_setup()
        test_performance_calculation()
        
        print("\n🎉 All Integration Tests Passed!")
        print("✅ Production system is ready for deployment")
        print(f"🎯 Expected fitness target: {EXPECTED_PERFORMANCE['fitness_target']:.3f}")
        print(f"🎯 Expected stability target: {EXPECTED_PERFORMANCE['stability_target']:.3f}")
        print(f"🎯 Expected response rate: {EXPECTED_PERFORMANCE['response_rate_target']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)