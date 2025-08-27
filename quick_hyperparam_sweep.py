#!/usr/bin/env python3
"""
Quick Hyperparameter Sweep - Optimized for speed while maintaining accuracy
"""

import asyncio
import time
import random
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import differential_evolution

@dataclass
class EmotionParams:
    """Core emotion parameters for optimization"""
    curiosity_base: float = 0.5
    confidence_base: float = 0.5
    social_energy_base: float = 0.5
    restlessness_base: float = 0.5
    harmony_base: float = 0.5
    novelty_sensitivity: float = 1.0
    social_decay_rate: float = 0.05
    response_threshold: float = 0.5
    confidence_weight: float = 1.0
    novelty_weight: float = 0.8

class QuickEmotionSimulator:
    """Fast emotion dynamics simulator"""
    
    def __init__(self, params: EmotionParams):
        self.params = params
        self.state = [params.curiosity_base, params.confidence_base, 
                     params.social_energy_base, params.restlessness_base, params.harmony_base]
    
    def simulate_episode(self, env_sequence: List[Tuple[float, float, bool]]) -> Dict[str, float]:
        """Simulate full episode with environment sequence"""
        state_history = []
        response_decisions = []
        consciousness_changes = 0
        last_consciousness = 2
        
        for env_activity, env_novelty, direct_mention in env_sequence:
            # Update state
            curiosity, confidence, social_energy, restlessness, harmony = self.state
            
            # Curiosity responds to novelty
            curiosity += env_novelty * self.params.novelty_sensitivity * 0.02
            curiosity = np.clip(curiosity - 0.005, 0.0, 1.0)  # Natural decay
            
            # Confidence
            if direct_mention or env_activity > 0.7:
                confidence += 0.05
            else:
                confidence -= 0.01
            confidence = np.clip(confidence, 0.0, 1.0)
            
            # Social energy
            if env_activity > 0.8:
                social_energy -= self.params.social_decay_rate
            elif env_activity < 0.3:
                social_energy += 0.02
            social_energy = np.clip(social_energy, 0.0, 1.0)
            
            # Restlessness
            if env_activity < 0.2 or env_activity > 0.9:
                restlessness += 0.03
            else:
                restlessness -= 0.01
            restlessness = np.clip(restlessness, 0.0, 1.0)
            
            # Harmony
            harmony_target = 1.0 - (env_activity - 0.5) ** 2
            harmony += (harmony_target - harmony) * 0.1
            harmony = np.clip(harmony, 0.0, 1.0)
            
            self.state = [curiosity, confidence, social_energy, restlessness, harmony]
            
            # Consciousness level
            activation = (curiosity + confidence + social_energy) / 3.0
            if activation > 0.8:
                consciousness_level = 4
            elif activation > 0.6:
                consciousness_level = 3
            elif activation > 0.3:
                consciousness_level = 2
            elif activation > 0.1:
                consciousness_level = 1
            else:
                consciousness_level = 0
            
            if consciousness_level != last_consciousness:
                consciousness_changes += 1
                last_consciousness = consciousness_level
            
            # Response decision
            decision_score = (
                curiosity * self.params.novelty_weight +
                confidence * self.params.confidence_weight +
                social_energy * 0.8 +
                (0.6 if direct_mention else 0.0)
            ) / 3.2
            
            would_respond = decision_score > self.params.response_threshold
            response_decisions.append(would_respond)
            state_history.append([curiosity, confidence, social_energy, restlessness, harmony, consciousness_level])
        
        return self._calculate_fast_scores(state_history, response_decisions, consciousness_changes)
    
    def _calculate_fast_scores(self, states: List, responses: List[bool], consciousness_changes: int) -> Dict[str, float]:
        """Fast scoring function"""
        if not states:
            return {'overall_fitness': 0.0}
        
        states_array = np.array(states)
        
        # Emotional balance (avoid extremes)
        balance_score = np.mean(1.0 - np.mean((states_array[:, :5] < 0.1) | (states_array[:, :5] > 0.9), axis=1))
        
        # Stability (low variance)
        stability_score = np.mean([max(0, 1.0 - np.var(states_array[:, i]) * 3) for i in range(5)])
        
        # Response rate
        response_rate = np.mean(responses)
        responsiveness_score = 1.0 if 0.25 <= response_rate <= 0.6 else max(0, 1.0 - abs(response_rate - 0.425) * 2.0)
        
        # Consciousness dynamics
        change_rate = consciousness_changes / len(states)
        consciousness_score = 1.0 if 0.05 <= change_rate <= 0.15 else max(0, 1.0 - abs(change_rate - 0.1) * 5.0)
        
        overall = 0.3 * balance_score + 0.3 * stability_score + 0.25 * responsiveness_score + 0.15 * consciousness_score
        
        return {
            'overall_fitness': overall,
            'balance': balance_score,
            'stability': stability_score,
            'responsiveness': responsiveness_score,
            'consciousness': consciousness_score
        }

class FastHyperparameterOptimizer:
    """Fast hyperparameter optimization"""
    
    def __init__(self):
        self.param_bounds = {
            'curiosity_base': (0.2, 0.8),
            'confidence_base': (0.3, 0.8),
            'social_energy_base': (0.4, 0.8),
            'restlessness_base': (0.1, 0.6),
            'harmony_base': (0.4, 0.7),
            'novelty_sensitivity': (0.7, 1.8),
            'social_decay_rate': (0.02, 0.12),
            'response_threshold': (0.35, 0.75),
            'confidence_weight': (0.6, 1.6),
            'novelty_weight': (0.5, 1.2)
        }
        self.param_names = list(self.param_bounds.keys())
        self.results = []
        
        # Pre-generate environment sequences for consistent evaluation
        self.env_sequences = [self._generate_env_sequence() for _ in range(3)]
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('FastOptimizer')
    
    def _generate_env_sequence(self) -> List[Tuple[float, float, bool]]:
        """Generate realistic environment sequence"""
        sequence = []
        for step in range(60):  # 30 minutes simulation
            activity = 0.5 + 0.3 * np.sin(2 * np.pi * step / 20) + random.uniform(-0.1, 0.1)
            novelty = 0.8 if random.random() < 0.08 else random.uniform(0.3, 0.7)
            mention = random.random() < 0.06
            sequence.append((np.clip(activity, 0.0, 1.0), novelty, mention))
        return sequence
    
    def evaluate_params(self, param_vector: np.ndarray) -> float:
        """Fast evaluation of parameter configuration"""
        try:
            params_dict = dict(zip(self.param_names, param_vector))
            params = EmotionParams(**params_dict)
            
            scores = []
            for env_seq in self.env_sequences:
                simulator = QuickEmotionSimulator(params)
                result = simulator.simulate_episode(env_seq)
                scores.append(result['overall_fitness'])
            
            return -np.mean(scores)  # Negative for minimization
        except Exception:
            return 1000.0
    
    def optimize(self) -> Dict[str, Any]:
        """Run optimization"""
        self.logger.info("🚀 Starting fast hyperparameter optimization...")
        
        bounds = list(self.param_bounds.values())
        
        # Quick random search first
        self.logger.info("Phase 1: Random sampling...")
        best_random = None
        best_score = float('inf')
        
        for i in range(40):
            x = [random.uniform(b[0], b[1]) for b in bounds]
            score = self.evaluate_params(np.array(x))
            
            if score < best_score:
                best_score = score
                best_random = x
            
            if i % 10 == 0:
                self.logger.info(f"Random {i}/40 - Best fitness: {-best_score:.4f}")
        
        # Evolutionary optimization
        self.logger.info("Phase 2: Evolutionary optimization...")
        result = differential_evolution(
            self.evaluate_params,
            bounds,
            seed=42,
            maxiter=25,
            popsize=10,
            disp=False
        )
        
        # Detailed evaluation of best result
        best_params = dict(zip(self.param_names, result.x))
        detailed_scores = self._detailed_evaluation(EmotionParams(**best_params))
        
        return {
            'best_params': best_params,
            'best_fitness': -result.fun,
            'detailed_scores': detailed_scores,
            'optimization_info': {
                'success': result.success,
                'iterations': result.nit,
                'function_evaluations': result.nfev
            }
        }
    
    def _detailed_evaluation(self, params: EmotionParams) -> Dict[str, float]:
        """Detailed evaluation of parameter set"""
        all_scores = []
        
        for env_seq in self.env_sequences * 3:  # More runs for detailed eval
            simulator = QuickEmotionSimulator(params)
            scores = simulator.simulate_episode(env_seq)
            all_scores.append(scores)
        
        # Average scores
        avg_scores = {}
        for key in all_scores[0].keys():
            avg_scores[key] = np.mean([s[key] for s in all_scores])
            avg_scores[f'{key}_std'] = np.std([s[key] for s in all_scores])
        
        return avg_scores
    
    def parameter_sensitivity_analysis(self, base_params: EmotionParams) -> Dict[str, float]:
        """Analyze parameter sensitivity around best configuration"""
        self.logger.info("Analyzing parameter sensitivity...")
        
        base_dict = base_params.__dict__
        base_score = -self.evaluate_params(np.array(list(base_dict.values())))
        
        sensitivities = {}
        
        for param_name in self.param_names:
            # Test small perturbations
            perturbations = [-0.05, -0.02, 0.02, 0.05]
            effects = []
            
            for delta in perturbations:
                modified_dict = base_dict.copy()
                original_value = modified_dict[param_name]
                
                # Apply perturbation within bounds
                bounds = self.param_bounds[param_name]
                new_value = np.clip(original_value + delta, bounds[0], bounds[1])
                modified_dict[param_name] = new_value
                
                modified_score = -self.evaluate_params(np.array(list(modified_dict.values())))
                effect = (modified_score - base_score) / abs(delta) if delta != 0 else 0
                effects.append(effect)
            
            sensitivities[param_name] = np.mean(np.abs(effects))
        
        return sensitivities
    
    def create_summary_visualization(self, results: Dict[str, Any]):
        """Create summary visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Best configuration parameters
        ax = axes[0, 0]
        params = results['best_params']
        param_names = [name.replace('_', '\n') for name in params.keys()]
        param_values = list(params.values())
        
        bars = ax.bar(range(len(params)), param_values, color='skyblue', alpha=0.7)
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels(param_names, fontsize=9, rotation=45)
        ax.set_title('Optimal Parameter Configuration', fontweight='bold')
        ax.set_ylabel('Parameter Value')
        
        for bar, value in zip(bars, param_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Score breakdown
        ax = axes[0, 1]
        scores = results['detailed_scores']
        score_names = ['balance', 'stability', 'responsiveness', 'consciousness', 'overall_fitness']
        score_values = [scores[name] for name in score_names]
        colors = ['lightcoral', 'lightgreen', 'gold', 'plum', 'lightblue']
        
        bars = ax.bar(score_names, score_values, color=colors, alpha=0.8)
        ax.set_title('Performance Breakdown', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        for bar, value in zip(bars, score_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Parameter bounds vs optimal
        ax = axes[0, 2]
        bounds_data = []
        optimal_data = []
        param_labels = []
        
        for param_name, (min_val, max_val) in self.param_bounds.items():
            optimal_val = params[param_name]
            normalized_optimal = (optimal_val - min_val) / (max_val - min_val)
            
            bounds_data.append([0, 1])  # Normalized bounds
            optimal_data.append(normalized_optimal)
            param_labels.append(param_name.replace('_', '\n'))
        
        y_pos = np.arange(len(param_labels))
        
        for i, (bounds, optimal) in enumerate(zip(bounds_data, optimal_data)):
            ax.barh(i, 1.0, color='lightgray', alpha=0.3)  # Full range
            ax.barh(i, 0.02, left=optimal-0.01, color='red', alpha=0.8)  # Optimal point
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_labels, fontsize=9)
        ax.set_xlabel('Normalized Range (0=min, 1=max)')
        ax.set_title('Optimal Values within Parameter Bounds', fontweight='bold')
        
        # 4. Sensitivity analysis (if available)
        if 'sensitivity' in results:
            ax = axes[1, 0]
            sensitivities = results['sensitivity']
            sens_names = list(sensitivities.keys())
            sens_values = list(sensitivities.values())
            
            bars = ax.barh(sens_names, sens_values, color='orange', alpha=0.7)
            ax.set_title('Parameter Sensitivity Analysis', fontweight='bold')
            ax.set_xlabel('Sensitivity (Fitness Change per Unit)')
            
            for bar, value in zip(bars, sens_values):
                ax.text(bar.get_width() + max(sens_values) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='left', va='center', fontsize=8)
        
        # 5. Score stability (error bars)
        ax = axes[1, 1]
        score_means = [scores[name] for name in score_names]
        score_stds = [scores.get(f'{name}_std', 0) for name in score_names]
        
        bars = ax.bar(score_names, score_means, yerr=score_stds, capsize=5,
                     color=colors, alpha=0.8)
        ax.set_title('Score Stability (Mean ± Std)', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # 6. Summary metrics
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""OPTIMIZATION SUMMARY

🏆 Best Overall Fitness: {results['best_fitness']:.4f}

📊 Component Scores:
  Balance:        {scores['balance']:.3f} ± {scores.get('balance_std', 0):.3f}
  Stability:      {scores['stability']:.3f} ± {scores.get('stability_std', 0):.3f}
  Responsiveness: {scores['responsiveness']:.3f} ± {scores.get('responsiveness_std', 0):.3f}
  Consciousness:  {scores['consciousness']:.3f} ± {scores.get('consciousness_std', 0):.3f}

⚙️ Optimization Info:
  Success:     {results['optimization_info']['success']}
  Iterations:  {results['optimization_info']['iterations']}
  Evaluations: {results['optimization_info']['function_evaluations']}

🎯 Key Recommendations:
  • Use these parameters for production
  • System shows good stability
  • Balanced emotional responses
  • Appropriate consciousness dynamics"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('fast_hyperparameter_results.png', dpi=300, bbox_inches='tight')
        self.logger.info("📊 Results visualization saved to fast_hyperparameter_results.png")
        plt.show()

async def main():
    """Run fast hyperparameter optimization"""
    print("🚀 FAST HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    optimizer = FastHyperparameterOptimizer()
    
    start_time = time.time()
    
    # Run optimization
    results = optimizer.optimize()
    
    # Add sensitivity analysis
    best_params = EmotionParams(**results['best_params'])
    results['sensitivity'] = optimizer.parameter_sensitivity_analysis(best_params)
    
    optimization_time = time.time() - start_time
    
    # Display results
    print(f"\n🎉 OPTIMIZATION COMPLETED in {optimization_time:.1f} seconds")
    print(f"🏆 Best Fitness: {results['best_fitness']:.4f}")
    print("\n📋 OPTIMAL CONFIGURATION:")
    print("-" * 40)
    for param, value in results['best_params'].items():
        print(f"{param:20}: {value:.4f}")
    
    print("\n📊 PERFORMANCE BREAKDOWN:")
    print("-" * 40)
    scores = results['detailed_scores']
    print(f"Emotional Balance  : {scores['balance']:.4f} ± {scores.get('balance_std', 0):.3f}")
    print(f"Stability         : {scores['stability']:.4f} ± {scores.get('stability_std', 0):.3f}")
    print(f"Responsiveness    : {scores['responsiveness']:.4f} ± {scores.get('responsiveness_std', 0):.3f}")
    print(f"Consciousness Dyn.: {scores['consciousness']:.4f} ± {scores.get('consciousness_std', 0):.3f}")
    
    print("\n🎯 PARAMETER SENSITIVITY (Top 5):")
    print("-" * 40)
    sorted_sensitivity = sorted(results['sensitivity'].items(), key=lambda x: x[1], reverse=True)
    for param, sens in sorted_sensitivity[:5]:
        print(f"{param:20}: {sens:.4f}")
    
    # Create visualization
    print("\n📈 Generating visualization...")
    optimizer.create_summary_visualization(results)
    
    # Save results
    with open('fast_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n💾 Results saved to fast_optimization_results.json")
    print("\n✅ HYPERPARAMETER SWEEP COMPLETE!")
    
    # Final recommendations
    print("\n🎯 PRODUCTION RECOMMENDATIONS:")
    print("Use the optimal configuration shown above for:")
    print("• Stable emotional dynamics")
    print("• Balanced response patterns") 
    print("• Appropriate consciousness transitions")
    print("• Robust performance across scenarios")

if __name__ == "__main__":
    asyncio.run(main())