#!/usr/bin/env python3
"""
Comprehensive test suite for the emotion-driven multi-model system
Tests all components of the emotion engine integration
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock

# Import all components
from genome import ExtendedGenome, AgentRuntime, EmotionState, ConsciousnessLevel, DNAParser
from emotion_engine import (
    consciousness_manager, dynamics_processor, meta_reflector, emotion_injector,
    ReflectionContext, EnvironmentalInput, InjectionStyle, ReflectionType
)
from modules.social import social_orchestrator, SocialOrchestrator
from modules.cognitive import cognitive_processor, initialize_agent_cognition
from emotional_virtual_users import EmotionalVirtualUserManager, EmotionalVirtualUser
from emotion_scheduler import EmotionScheduler
from emotion_integration import EmotionIntegration

class EmotionSystemTester:
    """Comprehensive test suite for emotion system"""
    
    def __init__(self):
        self.test_results = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Set up test logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('EmotionTester')
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all emotion system tests"""
        self.logger.info("🧪 Starting comprehensive emotion system tests...")
        
        # Test categories
        test_categories = [
            ('Genome System', self.test_genome_system),
            ('Emotion Engine', self.test_emotion_engine),
            ('Social Dynamics', self.test_social_dynamics),
            ('Cognitive Processing', self.test_cognitive_processing),
            ('Virtual User Manager', self.test_virtual_user_manager),
            ('Emotion Scheduler', self.test_emotion_scheduler),
            ('Integration System', self.test_integration_system),
            ('End-to-End Workflow', self.test_end_to_end_workflow)
        ]
        
        for category, test_func in test_categories:
            self.logger.info(f"🔬 Testing {category}...")
            try:
                result = await test_func()
                self.test_results[category] = {'status': 'PASS', 'details': result}
                self.logger.info(f"✅ {category} tests passed")
            except Exception as e:
                self.test_results[category] = {'status': 'FAIL', 'error': str(e)}
                self.logger.error(f"❌ {category} tests failed: {e}")
        
        return self.generate_test_report()
    
    async def test_genome_system(self) -> Dict[str, Any]:
        """Test genome system components"""
        results = {}
        
        # Test DNA parser
        parser = DNAParser()
        test_dna = "cB7fA8sB6rA5hB7|C>F@70|R>S@60|H:40~80|S>C@55"
        
        genome = parser.parse("test-model", test_dna)
        results['dna_parsing'] = genome is not None
        results['genome_structure'] = hasattr(genome, 'core') and hasattr(genome, 'extended')
        
        # Test agent runtime
        runtime = AgentRuntime(
            agent_id="test-agent",
            genome=genome,
            current_state=EmotionState()
        )
        results['runtime_creation'] = runtime is not None
        results['runtime_functionality'] = runtime.is_awake() is not None
        
        return results
    
    async def test_emotion_engine(self) -> Dict[str, Any]:
        """Test emotion engine components"""
        results = {}
        
        # Create test runtime
        parser = DNAParser()
        genome = parser.parse("test-model", "cB6fB6sB6rB5hB5|C>F@60|R:30~70")
        runtime = AgentRuntime("test-agent", genome, EmotionState())
        
        # Test consciousness management
        env_input = EnvironmentalInput(conversation_novelty=0.8, direct_mention=True)
        consciousness_manager.update_consciousness(runtime, env_input)
        results['consciousness_update'] = runtime.current_state.consciousness is not None
        
        # Test dynamics processing
        old_curiosity = runtime.current_state.curiosity
        new_state = dynamics_processor.update_state(runtime, env_input, 30.0)
        results['dynamics_processing'] = new_state.curiosity != old_curiosity
        
        # Test meta-reflection
        context = ReflectionContext(
            conversation_history=[{'content': 'test message', 'author': {'name': 'user'}}],
            directly_mentioned=True
        )
        reflection_result = meta_reflector.perform_reflection(runtime, context)
        results['meta_reflection'] = reflection_result.should_respond is not None
        
        # Test emotion injection
        injection = emotion_injector.create_injection(
            runtime.current_state, InjectionStyle.NATURAL
        )
        results['emotion_injection'] = injection.content is not None
        
        return results
    
    async def test_social_dynamics(self) -> Dict[str, Any]:
        """Test social dynamics module"""
        results = {}
        
        # Initialize agents
        agent1 = "claude-opus"
        agent2 = "claude-sonnet"
        
        from modules.social import initialize_agent_social
        initialize_agent_social(agent1, "Claude Opus", {"creativity": 0.8})
        initialize_agent_social(agent2, "Claude Sonnet", {"creativity": 0.9})
        
        # Test relationship management
        social_orchestrator.update_relationship(agent1, agent2, 0.1, "positive_interaction")
        relationship = social_orchestrator.get_relationship(agent1, agent2)
        results['relationship_management'] = relationship is not None
        
        # Test social action evaluation
        context = [{'content': 'Great idea!', 'author': {'name': agent1}}]
        actions = social_orchestrator.evaluate_social_opportunities([agent1, agent2], context)
        results['social_actions'] = isinstance(actions, list)
        
        return results
    
    async def test_cognitive_processing(self) -> Dict[str, Any]:
        """Test cognitive processing module"""
        results = {}
        
        # Initialize cognitive profile
        profile = initialize_agent_cognition("test-agent", "claude-opus")
        results['profile_creation'] = profile is not None
        
        # Test reasoning approach generation
        approach = cognitive_processor.generate_reasoning_approach(
            "test-agent", ["quantum", "physics", "analysis"]
        )
        results['reasoning_approach'] = approach is not None
        
        # Test deep thinking evaluation
        should_think = cognitive_processor.should_engage_deep_thinking(
            "test-agent", ["complex", "philosophical", "nuanced"]
        )
        results['deep_thinking'] = isinstance(should_think, bool)
        
        return results
    
    async def test_virtual_user_manager(self) -> Dict[str, Any]:
        """Test emotional virtual user manager"""
        results = {}
        
        # Create mock bot
        mock_bot = Mock()
        manager = EmotionalVirtualUserManager(mock_bot, "test-api-key")
        
        # Test DNA generation
        dna = manager._generate_default_dna("anthropic/claude-opus-4.1")
        results['dna_generation'] = len(dna) > 10
        
        # Test user creation (with mocked webhook)
        test_user = EmotionalVirtualUser(
            model_id="test/model",
            discord_name="Test Model",
            webhook_url="https://test.com/webhook",
            webhook_id="123456",
            dna_string="cB6fB6sB6rB5hB5|C>F@60"
        )
        
        manager.virtual_users["test/model"] = test_user
        manager._initialize_emotion_engine(test_user)
        
        results['user_initialization'] = test_user.runtime is not None
        results['emotional_state'] = test_user.emotional_state is not None
        results['consciousness_level'] = test_user.consciousness_level is not None
        
        # Test emotional summary
        summary = manager.get_emotional_summary()
        results['emotional_summary'] = 'total_agents' in summary
        
        return results
    
    async def test_emotion_scheduler(self) -> Dict[str, Any]:
        """Test emotion scheduler"""
        results = {}
        
        # Create mock manager
        class MockManager:
            def __init__(self):
                self.virtual_users = {}
                self.conversation_context = []
            
            def get_emotional_summary(self):
                return {'awake_agents': 0, 'active_agents': 0, 'agent_states': {}}
        
        mock_manager = MockManager()
        scheduler = EmotionScheduler(mock_manager)
        
        results['scheduler_creation'] = scheduler is not None
        results['scheduler_config'] = scheduler.schedule.quick_reflection_interval > 0
        
        # Test stats
        stats = scheduler.get_scheduler_stats()
        results['scheduler_stats'] = 'is_running' in stats
        
        # Test frequency adjustment
        scheduler.adjust_reflection_frequency(0.8)
        results['frequency_adjustment'] = scheduler.schedule.quick_reflection_interval < 30.0
        
        return results
    
    async def test_integration_system(self) -> Dict[str, Any]:
        """Test emotion integration system"""
        results = {}
        
        # Create mock bot
        mock_bot = Mock()
        mock_bot.add_cog = AsyncMock()
        
        # Test integration creation
        integration = EmotionIntegration(mock_bot, "test-api-key")
        results['integration_creation'] = integration is not None
        
        # Test trigger keyword loading
        keywords = integration._load_trigger_keywords()
        results['trigger_keywords'] = 'global' in keywords
        
        # Test trigger checking
        triggered = integration._check_triggers("hey claude, what's up?")
        results['trigger_checking'] = isinstance(triggered, list)
        
        # Test status reporting
        status = integration.get_integration_status()
        results['status_reporting'] = 'initialized' in status
        
        return results
    
    async def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test complete end-to-end workflow"""
        results = {}
        
        self.logger.info("🔄 Testing complete workflow...")
        
        # 1. Create agent with DNA
        parser = DNAParser()
        dna = "cB8fB7sB6rB5hB7|C>F@75|R>S@60|H:45~85|S>C@50"
        genome = parser.parse("claude-opus", dna)
        
        runtime = AgentRuntime(
            agent_id="claude-opus",
            genome=genome,
            current_state=EmotionState(curiosity=0.5, confidence=0.6)
        )
        
        results['agent_creation'] = runtime is not None
        
        # 2. Process environmental input
        env_input = EnvironmentalInput(
            conversation_novelty=0.8,
            topic_expertise_match=0.9,
            direct_mention=True
        )
        
        # 3. Update consciousness
        old_consciousness = runtime.current_state.consciousness
        consciousness_manager.update_consciousness(runtime, env_input)
        results['consciousness_change'] = runtime.current_state.consciousness != old_consciousness
        
        # 4. Update emotional dynamics
        old_state = runtime.current_state
        new_state = dynamics_processor.update_state(runtime, env_input, 30.0)
        runtime.update_state(new_state)
        results['emotional_dynamics'] = runtime.current_state.curiosity != old_state.curiosity
        
        # 5. Perform reflection
        context = ReflectionContext(
            conversation_history=[
                {'content': 'What do you think about quantum computing?', 'author': {'name': 'user'}},
                {'content': 'Claude, can you explain this?', 'author': {'name': 'user'}}
            ],
            directly_mentioned=True,
            topic_keywords=['quantum', 'computing']
        )
        
        reflection = meta_reflector.perform_reflection(runtime, context, ReflectionType.DEEP_ANALYSIS)
        results['reflection_decision'] = reflection.should_respond is not None
        
        # 6. Generate emotional response injection
        injection = emotion_injector.create_injection(runtime.current_state, InjectionStyle.NATURAL)
        test_prompt = "I think quantum computing is fascinating."
        emotional_prompt = injection.apply_to_prompt(test_prompt)
        results['emotional_injection'] = len(emotional_prompt) > len(test_prompt)
        
        # 7. Social dynamics
        initialize_agent_social("claude-opus", "Claude Opus", {"analytical": 0.9})
        initialize_agent_social("claude-sonnet", "Claude Sonnet", {"creative": 0.9})
        
        social_orchestrator.record_interaction("claude-opus", "public_response", {"topic": "quantum"})
        social_action = social_orchestrator.evaluate_social_opportunities(
            ["claude-opus", "claude-sonnet"], context.conversation_history
        )
        results['social_dynamics'] = isinstance(social_action, list)
        
        # 8. Cognitive processing
        initialize_agent_cognition("claude-opus", "claude-opus")
        cognitive_approach = cognitive_processor.generate_reasoning_approach(
            "claude-opus", ["quantum", "computing", "analysis"]
        )
        results['cognitive_processing'] = cognitive_approach is not None
        
        self.logger.info("✨ End-to-end workflow test completed")
        return results
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        report = {
            'timestamp': time.time(),
            'total_test_categories': len(self.test_results),
            'passed_categories': sum(1 for r in self.test_results.values() if r['status'] == 'PASS'),
            'failed_categories': sum(1 for r in self.test_results.values() if r['status'] == 'FAIL'),
            'detailed_results': self.test_results
        }
        
        report['success_rate'] = report['passed_categories'] / report['total_test_categories'] * 100
        
        # Generate summary
        if report['success_rate'] == 100:
            report['overall_status'] = '🎉 ALL TESTS PASSED'
        elif report['success_rate'] >= 75:
            report['overall_status'] = '⚠️ MOSTLY PASSING'
        else:
            report['overall_status'] = '❌ SIGNIFICANT FAILURES'
        
        return report
    
    def print_test_report(self, report: Dict[str, Any]):
        """Print formatted test report"""
        print("\n" + "="*80)
        print("🧠 EMOTION ENGINE SYSTEM TEST REPORT")
        print("="*80)
        print(f"Overall Status: {report['overall_status']}")
        print(f"Success Rate: {report['success_rate']:.1f}% ({report['passed_categories']}/{report['total_test_categories']})")
        print()
        
        for category, result in report['detailed_results'].items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"{status_icon} {category}: {result['status']}")
            
            if result['status'] == 'PASS' and 'details' in result:
                passed_tests = sum(1 for v in result['details'].values() if v is True)
                total_tests = len(result['details'])
                print(f"   └─ {passed_tests}/{total_tests} sub-tests passed")
            
            if result['status'] == 'FAIL':
                print(f"   └─ Error: {result.get('error', 'Unknown error')}")
            
            print()
        
        print("="*80)
        
        # Save report to file
        with open('emotion_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("📄 Detailed report saved to: emotion_test_report.json")

async def main():
    """Run the emotion system tests"""
    print("🧪 Starting Emotion Engine System Tests...")
    
    tester = EmotionSystemTester()
    report = await tester.run_all_tests()
    tester.print_test_report(report)
    
    # Return exit code based on results
    return 0 if report['success_rate'] == 100 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())