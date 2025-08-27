#!/usr/bin/env python3
"""
Emotion Engine Simulation Framework
Validates system stability through comprehensive mock scenarios
"""

import asyncio
import time
import random
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Import our emotion system
from genome import DNAParser, AgentRuntime, EmotionState, ConsciousnessLevel
from emotion_engine import (
    consciousness_manager, dynamics_processor, meta_reflector,
    ReflectionContext, EnvironmentalInput, ReflectionType
)
from modules.social import social_orchestrator, initialize_agent_social
from modules.cognitive import cognitive_processor, initialize_agent_cognition

@dataclass
class SimulationAgent:
    """Agent wrapper for simulation"""
    agent_id: str
    display_name: str
    dna_string: str
    runtime: Optional[AgentRuntime] = None
    response_history: List[Dict] = field(default_factory=list)
    reflection_history: List[Dict] = field(default_factory=list)
    state_history: List[Dict] = field(default_factory=list)
    
@dataclass
class SimulationScenario:
    """Simulation scenario configuration"""
    name: str
    duration_minutes: float
    message_frequency: float  # messages per minute
    agents: List[SimulationAgent]
    conversation_patterns: List[str]
    environmental_factors: Dict[str, Any] = field(default_factory=dict)

class EmotionSimulation:
    """Main simulation engine for emotion system validation"""
    
    def __init__(self):
        self.setup_logging()
        self.parser = DNAParser()
        self.agents: Dict[str, SimulationAgent] = {}
        self.simulation_data = []
        self.stability_metrics = {}
        
    def setup_logging(self):
        """Setup simulation logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('simulation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EmotionSimulation')
    
    def create_test_agents(self) -> List[SimulationAgent]:
        """Create diverse test agents with different personalities"""
        test_agents = [
            SimulationAgent(
                agent_id="claude-opus-sim",
                display_name="Claude Opus",
                dna_string="c8f7s6r5h7"  # Ultra simple format
            ),
            SimulationAgent(
                agent_id="claude-sonnet-sim", 
                display_name="Claude Sonnet",
                dna_string="c7f8s7r6h6"  # Ultra simple format
            ),
            SimulationAgent(
                agent_id="gemini-sim",
                display_name="Gemini",
                dna_string="c6f7s5r6h5"  # Ultra simple format
            ),
            SimulationAgent(
                agent_id="deepseek-sim",
                display_name="DeepSeek",
                dna_string="c9f8s5r4h6"  # Ultra simple format
            )
        ]
        
        # Initialize agents
        for agent in test_agents:
            try:
                genome = self.parser.parse(agent.agent_id, agent.dna_string)
                agent.runtime = AgentRuntime(
                    agent_id=agent.agent_id,
                    genome=genome,
                    current_state=EmotionState(
                        curiosity=random.uniform(0.3, 0.7),
                        confidence=random.uniform(0.4, 0.8),
                        social_energy=random.uniform(0.5, 0.9),
                        restlessness=random.uniform(0.2, 0.6),
                        harmony=random.uniform(0.4, 0.7)
                    )
                )
                
                # Initialize social and cognitive profiles
                initialize_agent_social(agent.agent_id, agent.display_name)
                initialize_agent_cognition(agent.agent_id, agent.agent_id)
                
                self.agents[agent.agent_id] = agent
                self.logger.info(f"✅ Initialized agent: {agent.display_name}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {agent.agent_id}: {e}")
        
        return list(self.agents.values())
    
    async def run_stability_simulation(self, duration_hours: float = 2.0) -> Dict[str, Any]:
        """Run comprehensive stability simulation"""
        self.logger.info(f"🚀 Starting {duration_hours}h stability simulation...")
        
        agents = self.create_test_agents()
        if not agents:
            raise RuntimeError("No agents initialized successfully")
        
        # Simulation parameters
        total_steps = int(duration_hours * 60 * 2)  # 30-second steps
        step_duration = 30.0  # seconds
        
        stability_data = {
            'emotional_stability': [],
            'consciousness_transitions': [],
            'reflection_patterns': [],
            'social_dynamics': [],
            'system_performance': []
        }
        
        start_time = time.time()
        
        for step in range(total_steps):
            step_start = time.time()
            
            try:
                # Generate environmental stimuli
                env_factors = self._generate_environmental_stimuli(step, total_steps)
                
                # Update all agents
                step_data = await self._simulate_step(agents, env_factors, step_duration)
                
                # Collect stability metrics
                self._collect_stability_metrics(step_data, stability_data)
                
                # Log progress
                if step % 20 == 0:  # Every 10 minutes
                    elapsed = (time.time() - start_time) / 60
                    progress = (step / total_steps) * 100
                    self.logger.info(f"⏱️  Step {step}/{total_steps} ({progress:.1f}%) - {elapsed:.1f}min elapsed")
                
                # Small delay to prevent CPU overload
                await asyncio.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in simulation step {step}: {e}")
                continue
        
        # Analyze results
        results = self._analyze_stability_results(stability_data)
        results['simulation_duration'] = duration_hours
        results['total_steps'] = total_steps
        results['agents_tested'] = len(agents)
        
        self.logger.info(f"✅ Simulation completed! Overall stability: {results['overall_stability']:.2f}")
        return results
    
    def _generate_environmental_stimuli(self, step: int, total_steps: int) -> Dict[str, Any]:
        """Generate realistic environmental factors for simulation"""
        # Simulate conversation patterns
        activity_cycle = np.sin(2 * np.pi * step / (total_steps / 4))  # 4 cycles over simulation
        base_activity = 0.5 + 0.3 * activity_cycle
        
        # Add random spikes and lulls
        if random.random() < 0.05:  # 5% chance of activity spike
            base_activity = min(1.0, base_activity + random.uniform(0.2, 0.4))
        elif random.random() < 0.1:  # 10% chance of quiet period
            base_activity = max(0.1, base_activity - random.uniform(0.2, 0.3))
        
        # Generate conversation topics that affect novelty
        topics = ['technology', 'philosophy', 'creativity', 'science', 'casual', 'technical']
        current_topic = random.choice(topics)
        
        novelty_factors = {
            'technology': 0.6, 'philosophy': 0.9, 'creativity': 0.8,
            'science': 0.7, 'casual': 0.3, 'technical': 0.8
        }
        
        return {
            'conversation_activity': base_activity,
            'topic_novelty': novelty_factors.get(current_topic, 0.5),
            'social_pressure': random.uniform(0.2, 0.8),
            'topic': current_topic,
            'direct_mentions': random.random() < 0.1,  # 10% chance someone mentioned
            'time_factor': step / total_steps  # How far through simulation
        }
    
    async def _simulate_step(self, agents: List[SimulationAgent], 
                           env_factors: Dict[str, Any], delta_time: float) -> Dict[str, Any]:
        """Simulate one time step for all agents"""
        step_data = {
            'timestamp': time.time(),
            'environmental_factors': env_factors,
            'agent_states': {},
            'reflections': {},
            'consciousness_changes': {},
            'social_interactions': []
        }
        
        # Create shared conversation context
        conversation_context = self._generate_conversation_context(env_factors)
        
        # Update each agent
        for agent in agents:
            if not agent.runtime:
                continue
            
            try:
                # Create environmental input
                env_input = EnvironmentalInput(
                    conversation_novelty=env_factors['topic_novelty'],
                    topic_expertise_match=random.uniform(0.3, 0.9),
                    conversation_harmony=env_factors['conversation_activity'],
                    direct_mention=env_factors['direct_mentions'],
                    message_count_recent=int(env_factors['conversation_activity'] * 10)
                )
                
                # Store old state
                old_state = agent.runtime.current_state
                old_consciousness = old_state.consciousness
                
                # Update emotional dynamics
                new_state = dynamics_processor.update_state(agent.runtime, env_input, delta_time)
                agent.runtime.update_state(new_state)
                
                # Update consciousness
                consciousness_manager.update_consciousness(agent.runtime, env_input)
                new_consciousness = agent.runtime.current_state.consciousness
                
                # Perform reflection
                context = ReflectionContext.from_conversation(
                    conversation_context, agent.agent_id,
                    {a.agent_id: a.runtime.current_state for a in agents if a.runtime and a.agent_id != agent.agent_id}
                )
                
                reflection = meta_reflector.perform_reflection(agent.runtime, context)
                
                # Record data
                step_data['agent_states'][agent.agent_id] = {
                    'curiosity': new_state.curiosity,
                    'confidence': new_state.confidence,
                    'social_energy': new_state.social_energy,
                    'restlessness': new_state.restlessness,
                    'harmony': new_state.harmony,
                    'consciousness': new_state.consciousness.value
                }
                
                step_data['reflections'][agent.agent_id] = {
                    'should_respond': reflection.should_respond,
                    'confidence': reflection.confidence,
                    'reasoning': reflection.reasoning[:50]  # Truncate for storage
                }
                
                if old_consciousness != new_consciousness:
                    step_data['consciousness_changes'][agent.agent_id] = {
                        'from': old_consciousness.name,
                        'to': new_consciousness.name
                    }
                
                # Update agent history
                agent.state_history.append({
                    'timestamp': time.time(),
                    'state': step_data['agent_states'][agent.agent_id].copy()
                })
                
                agent.reflection_history.append({
                    'timestamp': time.time(),
                    'reflection': step_data['reflections'][agent.agent_id].copy()
                })
                
            except Exception as e:
                self.logger.error(f"Error updating agent {agent.agent_id}: {e}")
                continue
        
        # Apply cross-agent dynamics
        try:
            active_runtimes = {a.agent_id: a.runtime for a in agents if a.runtime and a.runtime.is_awake()}
            if len(active_runtimes) > 1:
                from emotion_engine.dynamics import cross_agent_dynamics
                updated_states = cross_agent_dynamics.apply_cross_agent_effects(active_runtimes)
                
                for agent_id, new_state in updated_states.items():
                    if agent_id in self.agents and self.agents[agent_id].runtime:
                        self.agents[agent_id].runtime.update_state(new_state)
        except Exception as e:
            self.logger.error(f"Error in cross-agent dynamics: {e}")
        
        return step_data
    
    def _generate_conversation_context(self, env_factors: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mock conversation context"""
        context = []
        
        # Generate messages based on activity level
        message_count = max(1, int(env_factors['conversation_activity'] * 5))
        
        for i in range(message_count):
            context.append({
                'content': f"Mock message about {env_factors['topic']} topic",
                'author': {'name': f'User{i+1}'},
                'timestamp': time.time() - (message_count - i) * 60  # Spread over last hour
            })
        
        return context
    
    def _collect_stability_metrics(self, step_data: Dict[str, Any], stability_data: Dict[str, List]):
        """Collect stability metrics from step data"""
        # Emotional stability (variance in emotional states)
        emotional_variances = []
        for agent_id, state in step_data['agent_states'].items():
            variance = np.var([state['curiosity'], state['confidence'], 
                             state['social_energy'], state['restlessness'], state['harmony']])
            emotional_variances.append(variance)
        
        stability_data['emotional_stability'].append({
            'timestamp': step_data['timestamp'],
            'mean_variance': np.mean(emotional_variances) if emotional_variances else 0,
            'agent_count': len(emotional_variances)
        })
        
        # Consciousness transitions
        transitions = len(step_data['consciousness_changes'])
        stability_data['consciousness_transitions'].append({
            'timestamp': step_data['timestamp'],
            'transition_count': transitions,
            'changes': step_data['consciousness_changes']
        })
        
        # Reflection patterns
        reflection_stats = {
            'positive_responses': sum(1 for r in step_data['reflections'].values() if r['should_respond']),
            'avg_confidence': np.mean([r['confidence'] for r in step_data['reflections'].values()]) if step_data['reflections'] else 0,
            'total_reflections': len(step_data['reflections'])
        }
        stability_data['reflection_patterns'].append({
            'timestamp': step_data['timestamp'],
            **reflection_stats
        })
    
    def _analyze_stability_results(self, stability_data: Dict[str, List]) -> Dict[str, Any]:
        """Analyze simulation results for stability metrics"""
        results = {}
        
        # Emotional stability analysis
        emotional_data = stability_data['emotional_stability']
        if emotional_data:
            variances = [d['mean_variance'] for d in emotional_data]
            results['emotional_stability'] = {
                'mean_variance': np.mean(variances),
                'variance_stability': np.std(variances),  # Lower = more stable
                'max_variance': np.max(variances),
                'min_variance': np.min(variances)
            }
        
        # Consciousness transition analysis
        transition_data = stability_data['consciousness_transitions']
        if transition_data:
            transition_counts = [d['transition_count'] for d in transition_data]
            results['consciousness_stability'] = {
                'avg_transitions_per_step': np.mean(transition_counts),
                'transition_stability': np.std(transition_counts),
                'total_transitions': sum(transition_counts)
            }
        
        # Reflection pattern analysis
        reflection_data = stability_data['reflection_patterns']
        if reflection_data:
            response_rates = [d['positive_responses'] / max(d['total_reflections'], 1) for d in reflection_data]
            confidences = [d['avg_confidence'] for d in reflection_data if d['avg_confidence'] > 0]
            
            results['reflection_stability'] = {
                'avg_response_rate': np.mean(response_rates),
                'response_rate_stability': np.std(response_rates),
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'confidence_stability': np.std(confidences) if confidences else 0
            }
        
        # Overall stability score (0-1, higher = more stable)
        stability_factors = []
        
        if 'emotional_stability' in results:
            # Lower variance = higher stability
            emotional_score = max(0, 1 - results['emotional_stability']['variance_stability'])
            stability_factors.append(emotional_score)
        
        if 'consciousness_stability' in results:
            # Moderate transitions are good, too many or too few are bad
            transition_score = max(0, 1 - abs(results['consciousness_stability']['avg_transitions_per_step'] - 0.1) / 0.1)
            stability_factors.append(transition_score)
        
        if 'reflection_stability' in results:
            # Stable response rates and high confidence are good
            reflection_score = (1 - results['reflection_stability']['response_rate_stability']) * results['reflection_stability']['avg_confidence']
            stability_factors.append(reflection_score)
        
        results['overall_stability'] = np.mean(stability_factors) if stability_factors else 0.0
        
        # Stability classification
        if results['overall_stability'] >= 0.8:
            results['stability_classification'] = 'EXCELLENT'
        elif results['overall_stability'] >= 0.6:
            results['stability_classification'] = 'GOOD'
        elif results['overall_stability'] >= 0.4:
            results['stability_classification'] = 'MODERATE'
        else:
            results['stability_classification'] = 'POOR'
        
        return results
    
    async def run_stress_test(self, intensity: str = "medium") -> Dict[str, Any]:
        """Run stress test with high activity and rapid changes"""
        self.logger.info(f"🔥 Starting {intensity} intensity stress test...")
        
        agents = self.create_test_agents()
        if not agents:
            raise RuntimeError("No agents initialized for stress test")
        
        # Stress test parameters
        intensity_configs = {
            'low': {'duration': 0.5, 'message_freq': 5, 'chaos_factor': 0.2},
            'medium': {'duration': 1.0, 'message_freq': 10, 'chaos_factor': 0.4},
            'high': {'duration': 1.5, 'message_freq': 20, 'chaos_factor': 0.6}
        }
        
        config = intensity_configs.get(intensity, intensity_configs['medium'])
        total_steps = int(config['duration'] * 60 * 4)  # 15-second steps for stress test
        
        stress_metrics = {
            'error_count': 0,
            'max_processing_time': 0,
            'memory_usage': [],
            'agent_failures': defaultdict(int),
            'recovery_times': []
        }
        
        for step in range(total_steps):
            step_start = time.time()
            
            try:
                # Generate chaotic environment
                env_factors = {
                    'conversation_activity': random.uniform(0.8, 1.0),
                    'topic_novelty': random.uniform(0.7, 1.0),
                    'social_pressure': random.uniform(0.6, 1.0),
                    'direct_mentions': random.random() < config['chaos_factor'],
                    'topic': random.choice(['urgent', 'complex', 'controversial', 'technical'])
                }
                
                # Rapid updates
                step_data = await self._simulate_step(agents, env_factors, 15.0)
                
                # Track performance
                processing_time = time.time() - step_start
                stress_metrics['max_processing_time'] = max(stress_metrics['max_processing_time'], processing_time)
                
                if step % 10 == 0:
                    progress = (step / total_steps) * 100
                    self.logger.info(f"🔥 Stress test {progress:.1f}% - Max processing: {stress_metrics['max_processing_time']:.3f}s")
                
            except Exception as e:
                stress_metrics['error_count'] += 1
                self.logger.error(f"Stress test error at step {step}: {e}")
        
        # Calculate stress test results
        results = {
            'intensity': intensity,
            'total_steps': total_steps,
            'error_rate': stress_metrics['error_count'] / total_steps,
            'max_processing_time': stress_metrics['max_processing_time'],
            'avg_processing_time': stress_metrics['max_processing_time'] / total_steps,
            'agents_survived': len([a for a in agents if a.runtime]),
            'stress_score': self._calculate_stress_score(stress_metrics, total_steps)
        }
        
        self.logger.info(f"🔥 Stress test completed! Score: {results['stress_score']:.2f}")
        return results
    
    def _calculate_stress_score(self, metrics: Dict, total_steps: int) -> float:
        """Calculate stress test score (0-1, higher = better performance under stress)"""
        error_penalty = metrics['error_count'] / total_steps
        performance_penalty = min(1.0, metrics['max_processing_time'] / 5.0)  # 5s max acceptable
        
        score = max(0.0, 1.0 - error_penalty - performance_penalty)
        return score
    
    async def run_edge_case_scenarios(self) -> Dict[str, Any]:
        """Test edge cases and boundary conditions"""
        self.logger.info("🧪 Running edge case scenario tests...")
        
        agents = self.create_test_agents()
        edge_case_results = {}
        
        # Test Case 1: All agents asleep
        self.logger.info("Testing: All agents forced to sleep")
        for agent in agents:
            if agent.runtime:
                agent.runtime.current_state.consciousness = ConsciousnessLevel.DEEP_SLEEP
        
        result1 = await self._test_scenario_stability("all_asleep", agents, duration_minutes=10)
        edge_case_results['all_asleep'] = result1
        
        # Test Case 2: Maximum stimulation
        self.logger.info("Testing: Maximum environmental stimulation")
        extreme_env = {
            'conversation_activity': 1.0,
            'topic_novelty': 1.0,
            'social_pressure': 1.0,
            'direct_mentions': True
        }
        result2 = await self._test_extreme_environment(agents, extreme_env, duration_minutes=15)
        edge_case_results['max_stimulation'] = result2
        
        # Test Case 3: Zero stimulation
        self.logger.info("Testing: Zero environmental stimulation")
        minimal_env = {
            'conversation_activity': 0.0,
            'topic_novelty': 0.0,
            'social_pressure': 0.0,
            'direct_mentions': False
        }
        result3 = await self._test_extreme_environment(agents, minimal_env, duration_minutes=15)
        edge_case_results['zero_stimulation'] = result3
        
        # Test Case 4: Rapid consciousness cycling
        self.logger.info("Testing: Rapid consciousness level changes")
        result4 = await self._test_consciousness_cycling(agents, duration_minutes=20)
        edge_case_results['consciousness_cycling'] = result4
        
        return edge_case_results
    
    async def _test_scenario_stability(self, scenario_name: str, agents: List[SimulationAgent], 
                                     duration_minutes: float) -> Dict[str, Any]:
        """Test stability of a specific scenario"""
        steps = int(duration_minutes * 2)  # 30-second steps
        errors = 0
        
        for step in range(steps):
            try:
                env_factors = self._generate_environmental_stimuli(step, steps)
                await self._simulate_step(agents, env_factors, 30.0)
            except Exception as e:
                errors += 1
                self.logger.error(f"Error in {scenario_name} step {step}: {e}")
        
        return {
            'scenario': scenario_name,
            'total_steps': steps,
            'error_count': errors,
            'success_rate': (steps - errors) / steps if steps > 0 else 0,
            'agents_functional': len([a for a in agents if a.runtime])
        }
    
    async def _test_extreme_environment(self, agents: List[SimulationAgent], 
                                      env_config: Dict[str, Any], duration_minutes: float) -> Dict[str, Any]:
        """Test system under extreme environmental conditions"""
        steps = int(duration_minutes * 2)
        state_changes = 0
        consciousness_changes = 0
        
        for step in range(steps):
            for agent in agents:
                if not agent.runtime:
                    continue
                
                old_state = agent.runtime.current_state
                old_consciousness = old_state.consciousness
                
                # Apply extreme environment
                env_input = EnvironmentalInput(
                    conversation_novelty=env_config['topic_novelty'],
                    conversation_harmony=env_config['conversation_activity'],
                    direct_mention=env_config['direct_mentions']
                )
                
                try:
                    new_state = dynamics_processor.update_state(agent.runtime, env_input, 30.0)
                    agent.runtime.update_state(new_state)
                    consciousness_manager.update_consciousness(agent.runtime, env_input)
                    
                    if new_state != old_state:
                        state_changes += 1
                    if agent.runtime.current_state.consciousness != old_consciousness:
                        consciousness_changes += 1
                        
                except Exception as e:
                    self.logger.error(f"Error in extreme environment test: {e}")
        
        return {
            'environment': env_config,
            'total_steps': steps,
            'state_changes': state_changes,
            'consciousness_changes': consciousness_changes,
            'change_rate': (state_changes + consciousness_changes) / (steps * len(agents))
        }
    
    async def _test_consciousness_cycling(self, agents: List[SimulationAgent], duration_minutes: float) -> Dict[str, Any]:
        """Test rapid consciousness level changes"""
        steps = int(duration_minutes * 4)  # 15-second steps for rapid cycling
        successful_transitions = 0
        failed_transitions = 0
        
        consciousness_levels = list(ConsciousnessLevel)
        
        for step in range(steps):
            for agent in agents:
                if not agent.runtime:
                    continue
                
                try:
                    # Force random consciousness transitions
                    old_level = agent.runtime.current_state.consciousness
                    new_level = random.choice(consciousness_levels)
                    
                    # Apply transition
                    agent.runtime.current_state.consciousness = new_level
                    
                    # Test if system handles the transition gracefully
                    env_input = EnvironmentalInput()
                    consciousness_manager.update_consciousness(agent.runtime, env_input)
                    
                    successful_transitions += 1
                    
                except Exception as e:
                    failed_transitions += 1
                    self.logger.error(f"Failed consciousness transition: {e}")
        
        return {
            'total_transitions': successful_transitions + failed_transitions,
            'successful_transitions': successful_transitions,
            'failed_transitions': failed_transitions,
            'success_rate': successful_transitions / (successful_transitions + failed_transitions) if (successful_transitions + failed_transitions) > 0 else 0
        }
    
    def generate_stability_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive stability report"""
        report = []
        report.append("=" * 80)
        report.append("🧠 EMOTION ENGINE STABILITY VALIDATION REPORT")
        report.append("=" * 80)
        
        if 'overall_stability' in results:
            stability = results['overall_stability']
            classification = results['stability_classification']
            report.append(f"Overall Stability Score: {stability:.3f} ({classification})")
            report.append("")
        
        if 'emotional_stability' in results:
            es = results['emotional_stability']
            report.append(f"📊 Emotional Stability:")
            report.append(f"  • Mean Variance: {es['mean_variance']:.4f}")
            report.append(f"  • Stability Factor: {es['variance_stability']:.4f}")
            report.append(f"  • Range: {es['min_variance']:.4f} - {es['max_variance']:.4f}")
            report.append("")
        
        if 'consciousness_stability' in results:
            cs = results['consciousness_stability']
            report.append(f"🧠 Consciousness Stability:")
            report.append(f"  • Avg Transitions/Step: {cs['avg_transitions_per_step']:.4f}")
            report.append(f"  • Transition Stability: {cs['transition_stability']:.4f}")
            report.append(f"  • Total Transitions: {cs['total_transitions']}")
            report.append("")
        
        if 'reflection_stability' in results:
            rs = results['reflection_stability']
            report.append(f"🤔 Reflection Stability:")
            report.append(f"  • Avg Response Rate: {rs['avg_response_rate']:.3f}")
            report.append(f"  • Response Stability: {rs['response_rate_stability']:.4f}")
            report.append(f"  • Avg Confidence: {rs['avg_confidence']:.3f}")
            report.append("")
        
        # Add recommendations
        report.append("📋 STABILITY ASSESSMENT:")
        if results.get('overall_stability', 0) >= 0.8:
            report.append("✅ SYSTEM IS HIGHLY STABLE - Ready for production")
        elif results.get('overall_stability', 0) >= 0.6:
            report.append("⚠️  SYSTEM IS MODERATELY STABLE - Minor tuning recommended")
        else:
            report.append("❌ SYSTEM NEEDS STABILITY IMPROVEMENTS")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filename: str = "stability_results.json"):
        """Save simulation results to file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"📄 Results saved to {filename}")

async def main():
    """Run comprehensive stability validation"""
    print("🧠 Starting Emotion Engine Stability Validation...")
    
    simulation = EmotionSimulation()
    
    try:
        # Run main stability simulation
        print("\n1️⃣ Running 2-hour stability simulation...")
        stability_results = await simulation.run_stability_simulation(duration_hours=2.0)
        
        # Run stress tests
        print("\n2️⃣ Running stress tests...")
        stress_low = await simulation.run_stress_test("low")
        stress_medium = await simulation.run_stress_test("medium") 
        stress_high = await simulation.run_stress_test("high")
        
        # Run edge case tests
        print("\n3️⃣ Running edge case scenarios...")
        edge_cases = await simulation.run_edge_case_scenarios()
        
        # Compile comprehensive results
        comprehensive_results = {
            'stability_simulation': stability_results,
            'stress_tests': {
                'low': stress_low,
                'medium': stress_medium,
                'high': stress_high
            },
            'edge_cases': edge_cases,
            'validation_timestamp': time.time(),
            'validation_summary': {
                'overall_stability': stability_results.get('overall_stability', 0),
                'stress_performance': np.mean([stress_low['stress_score'], stress_medium['stress_score'], stress_high['stress_score']]),
                'edge_case_resilience': np.mean([result['success_rate'] for result in edge_cases.values() if 'success_rate' in result])
            }
        }
        
        # Generate and display report
        report = simulation.generate_stability_report(stability_results)
        print("\n" + report)
        
        # Additional summary
        print("\n🎯 COMPREHENSIVE VALIDATION SUMMARY:")
        print(f"  Stability Score: {comprehensive_results['validation_summary']['overall_stability']:.3f}")
        print(f"  Stress Performance: {comprehensive_results['validation_summary']['stress_performance']:.3f}")
        print(f"  Edge Case Resilience: {comprehensive_results['validation_summary']['edge_case_resilience']:.3f}")
        
        # Save results
        simulation.save_results(comprehensive_results)
        
        # Final verdict
        overall_score = np.mean(list(comprehensive_results['validation_summary'].values()))
        if overall_score >= 0.8:
            print("\n🎉 VALIDATION PASSED - System is production-ready!")
        elif overall_score >= 0.6:
            print("\n⚠️  VALIDATION PARTIAL - System is functional with minor issues")
        else:
            print("\n❌ VALIDATION FAILED - System needs significant improvements")
        
        return comprehensive_results
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        simulation.logger.error(f"Validation error: {e}")
        return None

if __name__ == "__main__":
    results = asyncio.run(main())