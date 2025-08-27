#!/usr/bin/env python3
"""
Validation of Optimal Configuration from Hyperparameter Sweep
Tests system stability and performance with discovered optimal parameters
"""

import asyncio
import time
import random
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

@dataclass
class OptimalConfig:
    """Optimal configuration discovered by hyperparameter sweep"""
    curiosity_base: float = 0.5574
    confidence_base: float = 0.3433
    social_energy_base: float = 0.7805
    restlessness_base: float = 0.5642
    harmony_base: float = 0.4639
    novelty_sensitivity: float = 0.8349
    social_decay_rate: float = 0.0729
    response_threshold: float = 0.5785
    confidence_weight: float = 0.7716
    novelty_weight: float = 0.8712

class OptimalEmotionAgent:
    """Emotion agent using optimal configuration"""
    
    def __init__(self, agent_id: str, config: OptimalConfig):
        self.agent_id = agent_id
        self.config = config
        self.reset_state()
        self.history = []
        self.response_history = []
        
    def reset_state(self):
        """Reset to initial optimal state"""
        self.curiosity = self.config.curiosity_base
        self.confidence = self.config.confidence_base
        self.social_energy = self.config.social_energy_base
        self.restlessness = self.config.restlessness_base
        self.harmony = self.config.harmony_base
        self.consciousness_level = 2
        self.awake = True
    
    def update(self, env_activity: float, env_novelty: float, direct_mention: bool, delta_time: float = 30.0):
        """Update agent state with optimal dynamics"""
        dt = delta_time / 60.0  # Convert to minutes
        
        # Curiosity dynamics (optimized)
        curiosity_change = env_novelty * self.config.novelty_sensitivity * dt * 0.02
        self.curiosity = np.clip(self.curiosity + curiosity_change - 0.005 * dt, 0.0, 1.0)
        
        # Confidence dynamics (optimized)
        if direct_mention or env_activity > 0.7:
            confidence_change = 0.05 * dt
        else:
            confidence_change = -0.01 * dt
        self.confidence = np.clip(self.confidence + confidence_change, 0.0, 1.0)
        
        # Social energy dynamics (optimized)
        if env_activity > 0.8:
            social_change = -self.config.social_decay_rate * dt
        elif env_activity < 0.3:
            social_change = 0.02 * dt
        else:
            social_change = 0.0
        self.social_energy = np.clip(self.social_energy + social_change, 0.0, 1.0)
        
        # Restlessness dynamics (optimized)
        if env_activity < 0.2 or env_activity > 0.9:
            restlessness_change = 0.03 * dt
        else:
            restlessness_change = -0.01 * dt
        self.restlessness = np.clip(self.restlessness + restlessness_change, 0.0, 1.0)
        
        # Harmony dynamics (optimized)
        harmony_target = 1.0 - (env_activity - 0.5) ** 2
        harmony_change = (harmony_target - self.harmony) * 0.1 * dt
        self.harmony = np.clip(self.harmony + harmony_change, 0.0, 1.0)
        
        # Consciousness level (optimized thresholds)
        activation = (self.curiosity + self.confidence + self.social_energy) / 3.0
        if activation > 0.8:
            self.consciousness_level = 4  # Hyperfocus
        elif activation > 0.6:
            self.consciousness_level = 3  # Alert
        elif activation > 0.3:
            self.consciousness_level = 2  # Normal
        elif activation > 0.1:
            self.consciousness_level = 1  # Drowsy
        else:
            self.consciousness_level = 0  # Sleep
        
        self.awake = self.consciousness_level >= 2
        
        # Record state
        self.history.append({
            'timestamp': time.time(),
            'curiosity': self.curiosity,
            'confidence': self.confidence,
            'social_energy': self.social_energy,
            'restlessness': self.restlessness,
            'harmony': self.harmony,
            'consciousness': self.consciousness_level,
            'awake': self.awake,
            'env_activity': env_activity,
            'env_novelty': env_novelty,
            'direct_mention': direct_mention
        })
    
    def should_respond(self, env_activity: float, env_novelty: float, direct_mention: bool) -> Tuple[bool, float]:
        """Determine if agent should respond using optimal weights"""
        if not self.awake:
            return False, 0.0
        
        # Optimal decision function
        decision_score = (
            self.curiosity * self.config.novelty_weight +
            self.confidence * self.config.confidence_weight +
            self.social_energy * 0.8 +
            (0.6 if direct_mention else 0.0) +
            env_novelty * 0.3
        ) / 3.5
        
        # Consciousness level multiplier
        consciousness_multipliers = {0: 0.1, 1: 0.3, 2: 1.0, 3: 1.2, 4: 1.5}
        decision_score *= consciousness_multipliers.get(self.consciousness_level, 1.0)
        
        would_respond = decision_score > self.config.response_threshold
        confidence = min(1.0, decision_score)
        
        self.response_history.append({
            'timestamp': time.time(),
            'would_respond': would_respond,
            'decision_score': decision_score,
            'confidence': confidence
        })
        
        return would_respond, confidence
    
    def get_current_state_summary(self) -> Dict[str, float]:
        """Get current state summary"""
        return {
            'curiosity': self.curiosity,
            'confidence': self.confidence,
            'social_energy': self.social_energy,
            'restlessness': self.restlessness,
            'harmony': self.harmony,
            'consciousness': self.consciousness_level,
            'overall_activation': (self.curiosity + self.confidence + self.social_energy) / 3.0,
            'emotional_balance': 1.0 - np.std([self.curiosity, self.confidence, self.social_energy, self.restlessness, self.harmony]),
            'awake': float(self.awake)
        }

class SystemStabilityValidator:
    """Validates system stability with optimal configuration"""
    
    def __init__(self):
        self.setup_logging()
        self.optimal_config = OptimalConfig()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('StabilityValidator')
    
    async def run_long_term_stability_test(self, duration_hours: float = 6.0) -> Dict[str, Any]:
        """Run long-term stability test with optimal configuration"""
        self.logger.info(f"🔬 Running {duration_hours}h stability test with optimal configuration...")
        
        # Create multiple agents with optimal config
        agents = [
            OptimalEmotionAgent(f"agent_{i}", self.optimal_config)
            for i in range(4)
        ]
        
        # Simulation parameters
        total_steps = int(duration_hours * 60 * 2)  # 30-second steps
        step_duration = 30.0
        
        stability_metrics = {
            'emotional_variance': [],
            'response_patterns': [],
            'consciousness_transitions': [],
            'system_health': []
        }
        
        start_time = time.time()
        
        for step in range(total_steps):
            # Generate realistic environmental patterns
            env_factors = self._generate_realistic_environment(step, total_steps)
            
            step_data = {
                'step': step,
                'timestamp': time.time(),
                'agents': []
            }
            
            # Update all agents
            for agent in agents:
                agent.update(
                    env_factors['activity'],
                    env_factors['novelty'],
                    env_factors['direct_mention'],
                    step_duration
                )
                
                would_respond, response_confidence = agent.should_respond(
                    env_factors['activity'],
                    env_factors['novelty'],
                    env_factors['direct_mention']
                )
                
                step_data['agents'].append({
                    'agent_id': agent.agent_id,
                    'state': agent.get_current_state_summary(),
                    'would_respond': would_respond,
                    'response_confidence': response_confidence
                })
            
            # Collect stability metrics
            self._collect_stability_metrics(step_data, stability_metrics)
            
            # Progress logging
            if step % 120 == 0:  # Every hour
                elapsed_hours = (time.time() - start_time) / 3600
                progress = (step / total_steps) * 100
                self.logger.info(f"⏱️  {progress:.1f}% complete - {elapsed_hours:.2f}h elapsed")
                
                # Log current system health
                avg_health = np.mean([data['system_health'][-1] if data['system_health'] else 0 
                                    for data in [stability_metrics]])
                self.logger.info(f"📊 System health: {avg_health:.3f}")
        
        # Analyze results
        results = self._analyze_long_term_results(stability_metrics, agents)
        results['test_duration_hours'] = duration_hours
        results['total_steps'] = total_steps
        results['agents_tested'] = len(agents)
        
        self.logger.info(f"✅ Long-term stability test completed!")
        self.logger.info(f"📊 Overall system stability: {results['overall_stability']:.3f}")
        
        return results
    
    def _generate_realistic_environment(self, step: int, total_steps: int) -> Dict[str, Any]:
        """Generate realistic environmental patterns"""
        # Multi-layered activity patterns
        # Daily cycle (24h equivalent in simulation time)
        daily_cycle = 0.5 + 0.3 * np.sin(2 * np.pi * step / (total_steps / 4))
        
        # Conversation bursts and lulls
        burst_probability = 0.1 if random.random() < 0.05 else 0.0
        activity = np.clip(daily_cycle + burst_probability + random.uniform(-0.1, 0.1), 0.0, 1.0)
        
        # Novelty spikes
        novelty = 0.9 if random.random() < 0.08 else random.uniform(0.2, 0.6)
        
        # Direct mentions (vary with activity)
        mention_probability = 0.05 + activity * 0.05
        direct_mention = random.random() < mention_probability
        
        return {
            'activity': activity,
            'novelty': novelty,
            'direct_mention': direct_mention,
            'time_factor': step / total_steps
        }
    
    def _collect_stability_metrics(self, step_data: Dict, metrics: Dict):
        """Collect stability metrics from step data"""
        agent_states = [agent_data['state'] for agent_data in step_data['agents']]
        
        # Emotional variance (stability indicator)
        emotional_vars = []
        for state in agent_states:
            emotions = [state['curiosity'], state['confidence'], state['social_energy'], 
                       state['restlessness'], state['harmony']]
            emotional_vars.append(np.var(emotions))
        
        metrics['emotional_variance'].append({
            'step': step_data['step'],
            'mean_variance': np.mean(emotional_vars),
            'max_variance': np.max(emotional_vars),
            'agent_count': len(agent_states)
        })
        
        # Response patterns
        response_count = sum(1 for agent_data in step_data['agents'] if agent_data['would_respond'])
        avg_confidence = np.mean([agent_data['response_confidence'] for agent_data in step_data['agents']])
        
        metrics['response_patterns'].append({
            'step': step_data['step'],
            'response_rate': response_count / len(step_data['agents']),
            'avg_confidence': avg_confidence,
            'total_agents': len(step_data['agents'])
        })
        
        # System health (comprehensive measure)
        health_factors = []
        
        for state in agent_states:
            # Balance (avoid extremes)
            balance = 1.0 - np.mean([
                1 if val < 0.1 or val > 0.9 else 0
                for val in [state['curiosity'], state['confidence'], state['social_energy'], 
                           state['restlessness'], state['harmony']]
            ])
            
            # Activation (appropriate consciousness)
            activation_health = 1.0 if 0.3 <= state['overall_activation'] <= 0.8 else 0.5
            
            # Awake status appropriateness
            awake_health = state['awake']
            
            health_factors.append(np.mean([balance, activation_health, awake_health]))
        
        system_health = np.mean(health_factors)
        metrics['system_health'].append(system_health)
    
    def _analyze_long_term_results(self, metrics: Dict, agents: List) -> Dict[str, Any]:
        """Analyze long-term stability results"""
        results = {}
        
        # Emotional stability analysis
        if metrics['emotional_variance']:
            variances = [data['mean_variance'] for data in metrics['emotional_variance']]
            results['emotional_stability'] = {
                'mean_variance': np.mean(variances),
                'variance_trend': np.polyfit(range(len(variances)), variances, 1)[0],  # Slope
                'stability_score': max(0, 1.0 - np.mean(variances) * 3.0)
            }
        
        # Response pattern analysis
        if metrics['response_patterns']:
            response_rates = [data['response_rate'] for data in metrics['response_patterns']]
            confidences = [data['avg_confidence'] for data in metrics['response_patterns']]
            
            results['response_stability'] = {
                'mean_response_rate': np.mean(response_rates),
                'response_rate_stability': 1.0 - np.std(response_rates),
                'mean_confidence': np.mean(confidences),
                'confidence_stability': 1.0 - np.std(confidences)
            }
        
        # System health trends
        if metrics['system_health']:
            health_trend = np.polyfit(range(len(metrics['system_health'])), metrics['system_health'], 1)[0]
            results['system_health'] = {
                'mean_health': np.mean(metrics['system_health']),
                'health_trend': health_trend,  # Positive = improving, negative = degrading
                'min_health': np.min(metrics['system_health']),
                'health_stability': 1.0 - np.std(metrics['system_health'])
            }
        
        # Agent-specific analysis
        results['agent_analysis'] = {}
        for agent in agents:
            if agent.history:
                agent_df = pd.DataFrame(agent.history)
                results['agent_analysis'][agent.agent_id] = {
                    'final_state': agent.get_current_state_summary(),
                    'emotional_range': {
                        'curiosity': [agent_df['curiosity'].min(), agent_df['curiosity'].max()],
                        'confidence': [agent_df['confidence'].min(), agent_df['confidence'].max()],
                        'social_energy': [agent_df['social_energy'].min(), agent_df['social_energy'].max()]
                    },
                    'consciousness_distribution': agent_df['consciousness'].value_counts().to_dict(),
                    'response_rate': np.mean([r['would_respond'] for r in agent.response_history]) if agent.response_history else 0
                }
        
        # Overall stability score
        stability_components = []
        if 'emotional_stability' in results:
            stability_components.append(results['emotional_stability']['stability_score'])
        if 'response_stability' in results:
            stability_components.append(results['response_stability']['response_rate_stability'])
        if 'system_health' in results:
            stability_components.append(results['system_health']['health_stability'])
        
        results['overall_stability'] = np.mean(stability_components) if stability_components else 0.0
        
        # Stability classification
        if results['overall_stability'] >= 0.9:
            results['stability_class'] = 'EXCELLENT'
        elif results['overall_stability'] >= 0.75:
            results['stability_class'] = 'GOOD'
        elif results['overall_stability'] >= 0.6:
            results['stability_class'] = 'MODERATE'
        else:
            results['stability_class'] = 'POOR'
        
        return results
    
    def create_stability_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive stability report"""
        report = []
        report.append("=" * 80)
        report.append("🔬 OPTIMAL CONFIGURATION STABILITY VALIDATION REPORT")
        report.append("=" * 80)
        
        # Overall assessment
        stability = results['overall_stability']
        classification = results['stability_class']
        report.append(f"Overall Stability Score: {stability:.3f} ({classification})")
        report.append(f"Test Duration: {results['test_duration_hours']:.1f} hours")
        report.append(f"Agents Tested: {results['agents_tested']}")
        report.append("")
        
        # Detailed metrics
        if 'emotional_stability' in results:
            es = results['emotional_stability']
            report.append("📊 EMOTIONAL STABILITY:")
            report.append(f"  • Stability Score: {es['stability_score']:.3f}")
            report.append(f"  • Mean Variance: {es['mean_variance']:.4f}")
            report.append(f"  • Trend: {'Improving' if es['variance_trend'] < 0 else 'Stable' if abs(es['variance_trend']) < 0.0001 else 'Degrading'}")
            report.append("")
        
        if 'response_stability' in results:
            rs = results['response_stability']
            report.append("🗣️  RESPONSE PATTERNS:")
            report.append(f"  • Response Rate: {rs['mean_response_rate']:.3f}")
            report.append(f"  • Rate Stability: {rs['response_rate_stability']:.3f}")
            report.append(f"  • Avg Confidence: {rs['mean_confidence']:.3f}")
            report.append(f"  • Confidence Stability: {rs['confidence_stability']:.3f}")
            report.append("")
        
        if 'system_health' in results:
            sh = results['system_health']
            report.append("🏥 SYSTEM HEALTH:")
            report.append(f"  • Mean Health: {sh['mean_health']:.3f}")
            report.append(f"  • Health Trend: {'Improving' if sh['health_trend'] > 0 else 'Stable' if abs(sh['health_trend']) < 0.0001 else 'Declining'}")
            report.append(f"  • Minimum Health: {sh['min_health']:.3f}")
            report.append(f"  • Health Stability: {sh['health_stability']:.3f}")
            report.append("")
        
        # Agent analysis
        if 'agent_analysis' in results:
            report.append("🤖 INDIVIDUAL AGENT ANALYSIS:")
            for agent_id, analysis in results['agent_analysis'].items():
                final_state = analysis['final_state']
                report.append(f"  Agent {agent_id}:")
                report.append(f"    Response Rate: {analysis['response_rate']:.3f}")
                report.append(f"    Final Activation: {final_state['overall_activation']:.3f}")
                report.append(f"    Emotional Balance: {final_state['emotional_balance']:.3f}")
                consciousness_dist = analysis['consciousness_distribution']
                dominant_consciousness = max(consciousness_dist.items(), key=lambda x: x[1])
                report.append(f"    Dominant Consciousness: Level {dominant_consciousness[0]} ({dominant_consciousness[1]} steps)")
            report.append("")
        
        # Recommendations
        report.append("📋 STABILITY ASSESSMENT:")
        if stability >= 0.9:
            report.append("✅ EXCELLENT STABILITY - Optimal configuration is production-ready")
            report.append("  • System maintains stable emotional dynamics over extended periods")
            report.append("  • Response patterns are consistent and appropriate")
            report.append("  • No degradation observed during long-term operation")
        elif stability >= 0.75:
            report.append("✅ GOOD STABILITY - Configuration is suitable for production")
            report.append("  • Minor fluctuations within acceptable ranges")
            report.append("  • System demonstrates good long-term behavior")
        elif stability >= 0.6:
            report.append("⚠️  MODERATE STABILITY - Some concerns identified")
            report.append("  • Monitor system behavior in production")
            report.append("  • Consider parameter fine-tuning if issues arise")
        else:
            report.append("❌ POOR STABILITY - Configuration needs improvement")
            report.append("  • Significant stability issues detected")
            report.append("  • Further optimization recommended")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

async def main():
    """Run comprehensive stability validation"""
    print("🔬 OPTIMAL CONFIGURATION STABILITY VALIDATION")
    print("=" * 70)
    
    validator = SystemStabilityValidator()
    
    # Display optimal configuration
    config = validator.optimal_config
    print("\n⚙️  TESTING OPTIMAL CONFIGURATION:")
    print("-" * 40)
    for attr, value in config.__dict__.items():
        print(f"{attr:20}: {value:.4f}")
    
    try:
        # Run long-term stability test
        print(f"\n🔬 Running 6-hour stability simulation...")
        start_time = time.time()
        
        results = await validator.run_long_term_stability_test(duration_hours=6.0)
        
        test_time = time.time() - start_time
        
        # Generate and display report
        report = validator.create_stability_report(results)
        print("\n" + report)
        
        print(f"\n⏱️  Validation completed in {test_time:.1f} seconds")
        
        # Save results
        with open('optimal_config_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("💾 Detailed results saved to optimal_config_validation_results.json")
        
        # Final verdict
        stability_score = results['overall_stability']
        if stability_score >= 0.9:
            print("\n🎉 VALIDATION PASSED - Optimal configuration is EXCELLENT for production!")
        elif stability_score >= 0.75:
            print("\n✅ VALIDATION PASSED - Optimal configuration is GOOD for production!")
        elif stability_score >= 0.6:
            print("\n⚠️  VALIDATION PARTIAL - Configuration is acceptable but monitor closely")
        else:
            print("\n❌ VALIDATION FAILED - Configuration needs further optimization")
        
        print(f"\nFinal Stability Score: {stability_score:.4f}")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        logging.error(f"Validation error: {e}")

if __name__ == "__main__":
    asyncio.run(main())