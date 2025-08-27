#!/usr/bin/env python3
"""
Simplified Hyperparameter Sweep for Emotion Engine
Direct parameter space exploration without full system dependencies
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
from dataclasses import dataclass, field
from scipy.optimize import differential_evolution
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

@dataclass
class EmotionParams:
    """Simplified emotion parameters for optimization"""
    curiosity_base: float = 0.5
    confidence_base: float = 0.5
    social_energy_base: float = 0.5
    restlessness_base: float = 0.5
    harmony_base: float = 0.5
    
    # Dynamic response parameters
    novelty_sensitivity: float = 1.0
    social_decay_rate: float = 0.05
    harmony_momentum: float = 0.7
    confidence_boost: float = 0.3
    restlessness_threshold: float = 0.6
    
    # Consciousness thresholds
    alert_threshold: float = 0.6
    hyperfocus_threshold: float = 0.8
    sleep_threshold: float = 0.2
    
    # Response parameters
    response_threshold: float = 0.5
    confidence_weight: float = 1.0
    novelty_weight: float = 0.8
    social_weight: float = 0.6

@dataclass
class SimulationState:
    """Simplified agent state for simulation"""
    curiosity: float = 0.5
    confidence: float = 0.5
    social_energy: float = 0.5
    restlessness: float = 0.5
    harmony: float = 0.5
    consciousness_level: int = 2  # 0-4 scale
    awake: bool = True

@dataclass
class EnvironmentState:
    """Environmental factors affecting agents"""
    conversation_activity: float = 0.5
    topic_novelty: float = 0.5
    social_pressure: float = 0.5
    direct_mention: bool = False
    time_factor: float = 0.5

class SimplifiedEmotionSimulator:
    """Simplified emotion dynamics simulator"""
    
    def __init__(self, params: EmotionParams):
        self.params = params
        self.state = SimulationState(
            curiosity=params.curiosity_base,
            confidence=params.confidence_base,
            social_energy=params.social_energy_base,
            restlessness=params.restlessness_base,
            harmony=params.harmony_base
        )
    
    def update_state(self, env: EnvironmentState, delta_time: float = 30.0) -> SimulationState:
        """Update emotional state based on environment"""
        # Normalize delta_time to minutes
        dt = delta_time / 60.0
        
        # Update emotions based on environment
        new_state = SimulationState(
            curiosity=self.state.curiosity,
            confidence=self.state.confidence,
            social_energy=self.state.social_energy,
            restlessness=self.state.restlessness,
            harmony=self.state.harmony,
            consciousness_level=self.state.consciousness_level,
            awake=self.state.awake
        )
        
        # Curiosity responds to novelty
        curiosity_change = env.topic_novelty * self.params.novelty_sensitivity * dt * 0.1
        new_state.curiosity = np.clip(self.state.curiosity + curiosity_change, 0.0, 1.0)
        
        # Confidence boosted by successful interactions and expertise
        if env.direct_mention or env.conversation_activity > 0.7:
            confidence_change = self.params.confidence_boost * dt * 0.1
        else:
            confidence_change = -0.02 * dt  # Natural decay
        new_state.confidence = np.clip(self.state.confidence + confidence_change, 0.0, 1.0)
        
        # Social energy affected by activity level
        if env.conversation_activity > 0.8:
            # High activity drains social energy
            social_change = -self.params.social_decay_rate * dt
        elif env.conversation_activity < 0.3:
            # Low activity restores energy
            social_change = 0.03 * dt
        else:
            social_change = 0.0
        new_state.social_energy = np.clip(self.state.social_energy + social_change, 0.0, 1.0)
        
        # Restlessness increases with inactivity or high stimulation
        if env.conversation_activity < 0.2 or env.conversation_activity > 0.9:
            restlessness_change = 0.05 * dt
        else:
            restlessness_change = -0.02 * dt  # Natural decay
        new_state.restlessness = np.clip(self.state.restlessness + restlessness_change, 0.0, 1.0)
        
        # Harmony affected by social pressure and momentum
        harmony_target = 1.0 - env.social_pressure
        harmony_change = (harmony_target - self.state.harmony) * (1.0 - self.params.harmony_momentum) * dt * 0.2
        new_state.harmony = np.clip(self.state.harmony + harmony_change, 0.0, 1.0)
        
        # Update consciousness based on overall state
        activation = (new_state.curiosity + new_state.confidence + new_state.social_energy) / 3.0
        
        if activation > self.params.hyperfocus_threshold:
            new_state.consciousness_level = 4  # Hyperfocus
        elif activation > self.params.alert_threshold:
            new_state.consciousness_level = 3  # Alert
        elif activation > self.params.sleep_threshold:
            new_state.consciousness_level = 2  # Normal
        elif activation > 0.1:
            new_state.consciousness_level = 1  # Drowsy
        else:
            new_state.consciousness_level = 0  # Sleep
        
        new_state.awake = new_state.consciousness_level >= 2
        
        self.state = new_state
        return new_state
    
    def should_respond(self, env: EnvironmentState) -> Tuple[bool, float]:
        """Determine if agent should respond and confidence level"""
        decision_factors = []
        
        # Base factors
        decision_factors.append(self.state.curiosity * self.params.novelty_weight)
        decision_factors.append(self.state.confidence * self.params.confidence_weight)
        decision_factors.append(self.state.social_energy * self.params.social_weight)
        
        # Environmental factors
        if env.direct_mention:
            decision_factors.append(0.8)  # Strong motivation to respond when mentioned
        
        if env.topic_novelty > 0.7:
            decision_factors.append(self.state.curiosity * 0.5)  # Novelty attracts curious agents
        
        if env.conversation_activity < 0.3:
            decision_factors.append(0.3)  # Help restart quiet conversations
        
        # Consciousness level affects decision
        consciousness_multiplier = {0: 0.1, 1: 0.3, 2: 1.0, 3: 1.2, 4: 1.5}[self.state.consciousness_level]
        
        decision_score = np.mean(decision_factors) * consciousness_multiplier
        would_respond = decision_score > self.params.response_threshold
        
        return would_respond, min(decision_score, 1.0)

class HyperparameterOptimizer:
    """Hyperparameter optimization for emotion engine"""
    
    def __init__(self):
        self.results = []
        self.setup_logging()
        self.param_bounds = self._define_parameter_bounds()
        self.param_names = list(self.param_bounds.keys())
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('HyperparamOptimizer')
    
    def _define_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Define bounds for each parameter"""
        return {
            'curiosity_base': (0.1, 0.9),
            'confidence_base': (0.2, 0.9),
            'social_energy_base': (0.3, 0.9),
            'restlessness_base': (0.1, 0.7),
            'harmony_base': (0.3, 0.8),
            'novelty_sensitivity': (0.5, 2.0),
            'social_decay_rate': (0.01, 0.15),
            'harmony_momentum': (0.5, 0.95),
            'confidence_boost': (0.1, 0.6),
            'restlessness_threshold': (0.3, 0.8),
            'alert_threshold': (0.4, 0.8),
            'hyperfocus_threshold': (0.7, 0.95),
            'sleep_threshold': (0.1, 0.4),
            'response_threshold': (0.3, 0.8),
            'confidence_weight': (0.5, 2.0),
            'novelty_weight': (0.3, 1.5),
            'social_weight': (0.2, 1.2)
        }
    
    def params_from_vector(self, x: np.ndarray) -> EmotionParams:
        """Convert parameter vector to EmotionParams object"""
        param_dict = dict(zip(self.param_names, x))
        return EmotionParams(**param_dict)
    
    def evaluate_configuration(self, params: EmotionParams) -> Dict[str, float]:
        """Evaluate a parameter configuration through simulation"""
        simulator = SimplifiedEmotionSimulator(params)
        
        # Simulation metrics
        state_history = []
        response_history = []
        consciousness_changes = 0
        last_consciousness = simulator.state.consciousness_level
        
        # Run simulation with varied environments
        for step in range(120):  # 2 hours of 30-second steps
            # Generate environmental variation
            activity_cycle = 0.5 + 0.3 * np.sin(2 * np.pi * step / 40)  # ~20 minute cycles
            novelty_spikes = 0.8 if random.random() < 0.05 else random.uniform(0.2, 0.6)
            
            env = EnvironmentState(
                conversation_activity=activity_cycle + random.uniform(-0.2, 0.2),
                topic_novelty=novelty_spikes,
                social_pressure=random.uniform(0.2, 0.8),
                direct_mention=random.random() < 0.08,  # Mentioned ~8% of time
                time_factor=step / 120.0
            )
            
            # Update state
            new_state = simulator.update_state(env)
            would_respond, response_confidence = simulator.should_respond(env)
            
            # Record metrics
            state_history.append({
                'curiosity': new_state.curiosity,
                'confidence': new_state.confidence,
                'social_energy': new_state.social_energy,
                'restlessness': new_state.restlessness,
                'harmony': new_state.harmony,
                'consciousness': new_state.consciousness_level,
                'awake': new_state.awake
            })
            
            response_history.append({
                'would_respond': would_respond,
                'confidence': response_confidence,
                'env_activity': env.conversation_activity,
                'env_novelty': env.topic_novelty
            })
            
            # Track consciousness changes
            if new_state.consciousness_level != last_consciousness:
                consciousness_changes += 1
                last_consciousness = new_state.consciousness_level
        
        # Calculate performance metrics
        scores = self._calculate_scores(state_history, response_history, consciousness_changes)
        return scores
    
    def _calculate_scores(self, state_history: List[Dict], response_history: List[Dict], 
                         consciousness_changes: int) -> Dict[str, float]:
        """Calculate performance scores from simulation history"""
        state_df = pd.DataFrame(state_history)
        response_df = pd.DataFrame(response_history)
        
        scores = {}
        
        # 1. Emotional Balance Score (agents should maintain healthy ranges)
        emotional_cols = ['curiosity', 'confidence', 'social_energy', 'restlessness', 'harmony']
        balance_scores = []
        
        for col in emotional_cols:
            values = state_df[col].values
            # Ideal ranges (avoiding extremes)
            if col == 'confidence':
                ideal_range = (0.4, 0.8)
            elif col == 'restlessness':
                ideal_range = (0.2, 0.6)
            else:
                ideal_range = (0.3, 0.7)
            
            in_range = np.logical_and(values >= ideal_range[0], values <= ideal_range[1])
            range_score = np.mean(in_range)
            
            # Penalty for staying at extremes
            extreme_penalty = np.mean(np.logical_or(values < 0.1, values > 0.9)) * 0.3
            balance_scores.append(max(0, range_score - extreme_penalty))
        
        scores['emotional_balance'] = np.mean(balance_scores)
        
        # 2. Stability Score (low variance = more stable)
        stability_scores = []
        for col in emotional_cols:
            variance = np.var(state_df[col])
            # Convert variance to stability score (lower variance = higher stability)
            stability_score = max(0, 1.0 - variance * 3.0)
            stability_scores.append(stability_score)
        
        scores['stability'] = np.mean(stability_scores)
        
        # 3. Responsiveness Score (appropriate response rate)
        response_rate = np.mean(response_df['would_respond'])
        # Ideal response rate: 25-60%
        if 0.25 <= response_rate <= 0.6:
            scores['responsiveness'] = 1.0
        else:
            scores['responsiveness'] = max(0, 1.0 - abs(response_rate - 0.425) * 2.0)
        
        # 4. Adaptability Score (responds appropriately to environment)
        high_activity_responses = response_df[response_df['env_activity'] > 0.7]['would_respond']
        high_novelty_responses = response_df[response_df['env_novelty'] > 0.7]['would_respond']
        
        adaptability_factors = []
        if len(high_activity_responses) > 0:
            # Should be somewhat selective during high activity
            activity_selectivity = 1.0 - np.mean(high_activity_responses)
            adaptability_factors.append(activity_selectivity)
        
        if len(high_novelty_responses) > 0:
            # Should be more responsive to novel topics
            novelty_responsiveness = np.mean(high_novelty_responses)
            adaptability_factors.append(novelty_responsiveness)
        
        scores['adaptability'] = np.mean(adaptability_factors) if adaptability_factors else 0.5
        
        # 5. Consciousness Dynamics Score (appropriate level changes)
        consciousness_change_rate = consciousness_changes / len(state_history)
        # Ideal: moderate consciousness changes (5-15% of steps)
        if 0.05 <= consciousness_change_rate <= 0.15:
            scores['consciousness_dynamics'] = 1.0
        else:
            scores['consciousness_dynamics'] = max(0, 1.0 - abs(consciousness_change_rate - 0.1) * 5.0)
        
        # Overall fitness (weighted combination)
        scores['overall_fitness'] = (
            0.25 * scores['emotional_balance'] +
            0.25 * scores['stability'] +
            0.20 * scores['responsiveness'] +
            0.15 * scores['adaptability'] +
            0.15 * scores['consciousness_dynamics']
        )
        
        return scores
    
    def objective_function(self, x: np.ndarray) -> float:
        """Objective function for optimization (minimize negative fitness)"""
        try:
            params = self.params_from_vector(x)
            scores = self.evaluate_configuration(params)
            return -scores['overall_fitness']  # Minimize negative = maximize positive
        except Exception as e:
            self.logger.error(f"Evaluation error: {e}")
            return 1000.0  # High penalty for failed evaluations
    
    async def random_search(self, n_samples: int = 100) -> List[Dict]:
        """Random search across parameter space"""
        self.logger.info(f"🎲 Starting random search with {n_samples} samples...")
        
        results = []
        bounds_list = list(self.param_bounds.values())
        
        for i in range(n_samples):
            # Generate random parameters
            param_vector = [random.uniform(bound[0], bound[1]) for bound in bounds_list]
            params = self.params_from_vector(param_vector)
            
            # Evaluate
            scores = self.evaluate_configuration(params)
            
            result = {
                'method': 'random',
                'iteration': i,
                'params': params.__dict__.copy(),
                'scores': scores,
                'overall_fitness': scores['overall_fitness']
            }
            results.append(result)
            
            if i % 20 == 0:
                best_so_far = max(results, key=lambda x: x['overall_fitness'])['overall_fitness']
                self.logger.info(f"Random search {i}/{n_samples} - Best: {best_so_far:.4f}")
        
        self.results.extend(results)
        return results
    
    def evolutionary_optimization(self, maxiter: int = 50) -> Dict:
        """Evolutionary optimization using differential evolution"""
        self.logger.info(f"🧬 Starting evolutionary optimization...")
        
        bounds = list(self.param_bounds.values())
        
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=maxiter,
            popsize=15,
            seed=42,
            disp=True
        )
        
        # Evaluate best result
        best_params = self.params_from_vector(result.x)
        best_scores = self.evaluate_configuration(best_params)
        
        evolution_result = {
            'method': 'evolution',
            'params': best_params.__dict__.copy(),
            'scores': best_scores,
            'overall_fitness': best_scores['overall_fitness'],
            'optimization_result': {
                'success': result.success,
                'nit': result.nit,
                'nfev': result.nfev,
                'fun': result.fun
            }
        }
        
        self.results.append(evolution_result)
        return evolution_result
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze optimization results"""
        if not self.results:
            return {}
        
        # Convert to DataFrame for analysis
        analysis_data = []
        for result in self.results:
            row = result['params'].copy()
            row.update(result['scores'])
            row['method'] = result['method']
            analysis_data.append(row)
        
        df = pd.DataFrame(analysis_data)
        
        # Calculate correlations
        score_cols = ['emotional_balance', 'stability', 'responsiveness', 'adaptability', 'consciousness_dynamics', 'overall_fitness']
        param_cols = list(self.param_bounds.keys())
        
        correlations = {}
        for score_col in score_cols:
            correlations[score_col] = df[param_cols].corrwith(df[score_col]).to_dict()
        
        # Find best configurations
        best_configs = df.nlargest(10, 'overall_fitness')[param_cols + score_cols].to_dict('records')
        
        return {
            'dataframe': df,
            'correlations': correlations,
            'best_configurations': best_configs,
            'summary_stats': df[score_cols].describe()
        }
    
    def create_visualizations(self, analysis: Dict[str, Any]):
        """Create comprehensive visualizations"""
        df = analysis['dataframe']
        correlations = analysis['correlations']
        
        # Set up plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Overall fitness correlation heatmap
        ax1 = plt.subplot(3, 4, 1)
        fitness_corr = pd.Series(correlations['overall_fitness'])
        sns.heatmap(fitness_corr.values.reshape(-1, 1), 
                   yticklabels=[name.replace('_', '\n') for name in fitness_corr.index],
                   xticklabels=['Overall Fitness'],
                   annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax1)
        ax1.set_title('Parameter Correlations\nwith Overall Fitness', fontsize=10)
        
        # 2. Score breakdown heatmap
        ax2 = plt.subplot(3, 4, 2)
        score_corrs = pd.DataFrame({
            'Balance': correlations['emotional_balance'],
            'Stability': correlations['stability'],
            'Response': correlations['responsiveness'],
            'Adapt': correlations['adaptability'],
            'Conscious': correlations['consciousness_dynamics']
        })
        sns.heatmap(score_corrs.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax2)
        ax2.set_title('Parameter Correlations\nwith Score Components', fontsize=10)
        
        # 3. Parameter distributions for top configs
        ax3 = plt.subplot(3, 4, 3)
        top_configs = df.nlargest(min(50, len(df)), 'overall_fitness')
        key_params = ['curiosity_base', 'confidence_base', 'novelty_sensitivity', 'response_threshold']
        
        for param in key_params:
            ax3.hist(top_configs[param], alpha=0.6, label=param.replace('_', ' '), bins=15)
        ax3.set_title('Top Configs Parameter\nDistributions', fontsize=10)
        ax3.legend(fontsize=8)
        
        # 4. Fitness over iterations (for methods with iterations)
        ax4 = plt.subplot(3, 4, 4)
        random_results = df[df['method'] == 'random'].copy()
        if len(random_results) > 0:
            # Use index as iteration proxy for random search
            random_results = random_results.reset_index(drop=True)
            ax4.plot(random_results.index, random_results['overall_fitness'], 'b-', alpha=0.7, label='Random Search')
            
            # Running maximum
            running_max = random_results['overall_fitness'].expanding().max()
            ax4.plot(random_results.index, running_max, 'r-', linewidth=2, label='Best So Far')
        
        ax4.set_title('Optimization Progress', fontsize=10)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Fitness')
        ax4.legend(fontsize=8)
        
        # 5. Score component breakdown
        ax5 = plt.subplot(3, 4, 5)
        score_cols = ['emotional_balance', 'stability', 'responsiveness', 'adaptability', 'consciousness_dynamics']
        score_means = df[score_cols].mean()
        
        bars = ax5.bar(range(len(score_cols)), score_means, color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral'])
        ax5.set_xticks(range(len(score_cols)))
        ax5.set_xticklabels([col.replace('_', '\n') for col in score_cols], fontsize=8)
        ax5.set_title('Average Score Components', fontsize=10)
        ax5.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, score_means):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 6. Parameter importance (absolute correlation with fitness)
        ax6 = plt.subplot(3, 4, 6)
        param_importance = pd.Series(correlations['overall_fitness']).abs().sort_values(ascending=True)
        
        ax6.barh(range(len(param_importance)), param_importance.values)
        ax6.set_yticks(range(len(param_importance)))
        ax6.set_yticklabels([name.replace('_', ' ') for name in param_importance.index], fontsize=8)
        ax6.set_title('Parameter Importance\n(Absolute Correlation)', fontsize=10)
        ax6.set_xlabel('|Correlation|')
        
        # 7-9. Scatter plots of key parameters vs fitness
        key_params_scatter = sorted(correlations['overall_fitness'].items(), 
                                  key=lambda x: abs(x[1]), reverse=True)[:3]
        
        for i, (param, corr) in enumerate(key_params_scatter):
            ax = plt.subplot(3, 4, 7 + i)
            scatter = ax.scatter(df[param], df['overall_fitness'], 
                               c=df['stability'], alpha=0.6, cmap='viridis', s=30)
            ax.set_xlabel(param.replace('_', ' '))
            ax.set_ylabel('Overall Fitness')
            ax.set_title(f'{param.replace("_", " ")}\nvs Fitness (r={corr:.3f})', fontsize=10)
            
            if i == 0:  # Add colorbar only once
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Stability', fontsize=8)
        
        # 10. Method comparison
        ax10 = plt.subplot(3, 4, 10)
        method_performance = df.groupby('method')['overall_fitness'].agg(['mean', 'std', 'max']).reset_index()
        
        if len(method_performance) > 1:
            x_pos = range(len(method_performance))
            ax10.bar(x_pos, method_performance['mean'], yerr=method_performance['std'], 
                    capsize=5, alpha=0.7, color=['skyblue', 'lightgreen'])
            ax10.set_xticks(x_pos)
            ax10.set_xticklabels(method_performance['method'])
            ax10.set_title('Method Comparison\n(Mean ± Std)', fontsize=10)
            ax10.set_ylabel('Overall Fitness')
        
        # 11. PCA if we have enough data
        ax11 = plt.subplot(3, 4, 11)
        if len(df) > 10:
            param_data = df[list(self.param_bounds.keys())].values
            scaler = StandardScaler()
            param_scaled = scaler.fit_transform(param_data)
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(param_scaled)
            
            scatter = ax11.scatter(pca_result[:, 0], pca_result[:, 1], 
                                 c=df['overall_fitness'], cmap='viridis', alpha=0.7)
            ax11.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax11.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax11.set_title('PCA: Parameter Space\nProjection', fontsize=10)
            plt.colorbar(scatter, ax=ax11, label='Fitness')
        else:
            ax11.text(0.5, 0.5, 'Insufficient data\nfor PCA', ha='center', va='center', 
                     transform=ax11.transAxes)
            ax11.set_title('PCA (Insufficient Data)', fontsize=10)
        
        # 12. Best configuration summary
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        best_config = analysis['best_configurations'][0] if analysis['best_configurations'] else {}
        if best_config:
            summary_text = f"""BEST CONFIGURATION
            
Overall Fitness: {best_config.get('overall_fitness', 0):.4f}

Balance: {best_config.get('emotional_balance', 0):.3f}
Stability: {best_config.get('stability', 0):.3f}
Response: {best_config.get('responsiveness', 0):.3f}
Adaptability: {best_config.get('adaptability', 0):.3f}
Consciousness: {best_config.get('consciousness_dynamics', 0):.3f}

Key Parameters:
Curiosity: {best_config.get('curiosity_base', 0):.2f}
Confidence: {best_config.get('confidence_base', 0):.2f}
Response Threshold: {best_config.get('response_threshold', 0):.2f}
Novelty Sensitivity: {best_config.get('novelty_sensitivity', 0):.2f}"""
            
            ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, 
                     fontsize=9, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('hyperparameter_optimization_results.png', dpi=300, bbox_inches='tight')
        self.logger.info("📊 Comprehensive visualization saved to hyperparameter_optimization_results.png")
        plt.show()
    
    def save_results(self, filename: str = "hyperparameter_results.json"):
        """Save optimization results"""
        results_data = {
            'parameter_space': self.param_bounds,
            'optimization_results': self.results,
            'best_configurations': sorted(self.results, key=lambda x: x.get('overall_fitness', 0), reverse=True)[:10]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"📄 Results saved to {filename}")

async def main():
    """Run comprehensive hyperparameter optimization"""
    print("🚀 HYPERPARAMETER OPTIMIZATION FOR EMOTION ENGINE")
    print("=" * 80)
    
    optimizer = HyperparameterOptimizer()
    
    try:
        # Phase 1: Random search
        print("\n🎲 Phase 1: Random Search Exploration")
        await optimizer.random_search(n_samples=80)
        
        # Phase 2: Evolutionary optimization
        print("\n🧬 Phase 2: Evolutionary Optimization")
        evolution_result = optimizer.evolutionary_optimization(maxiter=40)
        
        # Analysis
        print("\n📊 Phase 3: Results Analysis")
        analysis = optimizer.analyze_results()
        
        # Display results
        print("\n🏆 TOP 5 CONFIGURATIONS:")
        print("-" * 70)
        for i, config in enumerate(analysis['best_configurations'][:5]):
            print(f"\nRank {i+1}: Overall Fitness = {config['overall_fitness']:.4f}")
            print(f"  Balance: {config['emotional_balance']:.3f} | Stability: {config['stability']:.3f}")
            print(f"  Responsiveness: {config['responsiveness']:.3f} | Adaptability: {config['adaptability']:.3f}")
            print(f"  Consciousness: {config['consciousness_dynamics']:.3f}")
            print(f"  Key params: curiosity={config['curiosity_base']:.2f}, confidence={config['confidence_base']:.2f}")
            print(f"              response_threshold={config['response_threshold']:.2f}, novelty_sens={config['novelty_sensitivity']:.2f}")
        
        # Show parameter importance
        print("\n🎯 MOST IMPORTANT PARAMETERS:")
        fitness_corrs = analysis['correlations']['overall_fitness']
        important_params = sorted(fitness_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        for param, corr in important_params:
            direction = "↑" if corr > 0 else "↓"
            print(f"  {param:20} {direction} {abs(corr):6.3f} correlation")
        
        # Create visualizations
        print("\n📈 Generating comprehensive visualizations...")
        optimizer.create_visualizations(analysis)
        
        # Save results
        optimizer.save_results()
        
        print("\n🎉 HYPERPARAMETER OPTIMIZATION COMPLETED!")
        print(f"Configurations evaluated: {len(optimizer.results)}")
        print(f"Best fitness achieved: {max(r.get('overall_fitness', 0) for r in optimizer.results):.4f}")
        
        # Provide recommendations
        best = analysis['best_configurations'][0]
        print(f"\n💡 RECOMMENDED CONFIGURATION:")
        print(f"Use these optimal parameters for production:")
        for param, value in best.items():
            if param not in ['emotional_balance', 'stability', 'responsiveness', 'adaptability', 'consciousness_dynamics', 'overall_fitness', 'method']:
                print(f"  {param}: {value:.3f}")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        logging.error(f"Optimization error: {e}")

if __name__ == "__main__":
    asyncio.run(main())