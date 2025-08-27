#!/usr/bin/env python3
"""
Hyperparameter Sweep for Emotion Engine Optimization
Explores high-dimensional parameter space to find optimal configurations
"""

import asyncio
import time
import random
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from itertools import product
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import emotion system components
from genome import AgentRuntime, EmotionState, ConsciousnessLevel
from emotion_engine import dynamics_processor, EnvironmentalInput

@dataclass
class ParameterSpace:
    """Defines the hyperparameter search space"""
    # Emotional base parameters
    curiosity_range: Tuple[float, float] = (0.0, 1.0)
    confidence_range: Tuple[float, float] = (0.0, 1.0)
    social_energy_range: Tuple[float, float] = (0.0, 1.0)
    restlessness_range: Tuple[float, float] = (0.0, 1.0)
    harmony_range: Tuple[float, float] = (0.0, 1.0)
    
    # Dynamic response parameters
    novelty_sensitivity: Tuple[float, float] = (0.1, 2.0)
    social_decay_rate: Tuple[float, float] = (0.01, 0.2)
    harmony_momentum: Tuple[float, float] = (0.5, 0.9)
    confidence_boost: Tuple[float, float] = (0.1, 0.5)
    restlessness_threshold: Tuple[float, float] = (0.3, 0.8)
    
    # Consciousness thresholds
    alert_threshold: Tuple[float, float] = (0.4, 0.8)
    hyperfocus_threshold: Tuple[float, float] = (0.7, 0.95)
    sleep_threshold: Tuple[float, float] = (0.1, 0.4)
    
    # Reflection parameters
    response_threshold: Tuple[float, float] = (0.3, 0.8)
    confidence_weight: Tuple[float, float] = (0.5, 2.0)
    novelty_weight: Tuple[float, float] = (0.3, 1.5)
    social_weight: Tuple[float, float] = (0.2, 1.2)

@dataclass
class Configuration:
    """Single configuration point in parameter space"""
    params: Dict[str, float]
    performance_score: float = 0.0
    stability_score: float = 0.0
    responsiveness_score: float = 0.0
    diversity_score: float = 0.0
    overall_fitness: float = 0.0

class HighDimensionalOptimizer:
    """Advanced hyperparameter optimization for emotion engine"""
    
    def __init__(self, parameter_space: ParameterSpace):
        self.param_space = parameter_space
        self.configurations: List[Configuration] = []
        self.evaluation_history = []
        self.best_configs = []
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('HyperparameterSweep')
    
    def generate_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Generate bounds for scipy optimization"""
        bounds = []
        param_names = []
        
        for attr_name in dir(self.param_space):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.param_space, attr_name)
                if isinstance(attr_value, tuple) and len(attr_value) == 2:
                    bounds.append(attr_value)
                    param_names.append(attr_name)
        
        self.param_names = param_names
        return bounds
    
    def params_from_vector(self, x: np.ndarray) -> Dict[str, float]:
        """Convert optimization vector to parameter dictionary"""
        return dict(zip(self.param_names, x))
    
    async def evaluate_configuration(self, params: Dict[str, float]) -> Configuration:
        """Evaluate a single configuration through simulation"""
        config = Configuration(params=params.copy())
        
        try:
            # Create test agent with this configuration
            agent = self.create_test_agent(params)
            
            # Run mini-simulation
            scores = await self.run_mini_simulation(agent, params)
            
            config.performance_score = scores['performance']
            config.stability_score = scores['stability']
            config.responsiveness_score = scores['responsiveness']
            config.diversity_score = scores['diversity']
            
            # Calculate overall fitness (weighted combination)
            config.overall_fitness = (
                0.3 * config.performance_score +
                0.3 * config.stability_score +
                0.2 * config.responsiveness_score +
                0.2 * config.diversity_score
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating configuration: {e}")
            config.overall_fitness = 0.0
        
        return config
    
    def create_test_agent(self, params: Dict[str, float]) -> AgentRuntime:
        """Create test agent with specific parameters"""
        # Create simplified runtime for testing
        from genome.base import GenomeCore, ExtendedGenome
        from genome.types import BaseStats
        
        base_stats = BaseStats(
            curiosity_base=params.get('curiosity_range', 0.5),
            confidence_base=params.get('confidence_range', 0.5),
            social_energy_base=params.get('social_energy_range', 0.5),
            restlessness_amplitude=params.get('restlessness_range', 0.5),
            harmony_factor=params.get('harmony_range', 0.5)
        )
        
        core = GenomeCore(
            model_id="test-agent",
            base_stats=base_stats,
            formulas={},
            transitions={},
            sample_rates={}
        )
        
        genome = ExtendedGenome(core=core)
        
        initial_state = EmotionState(
            curiosity=params.get('curiosity_range', 0.5),
            confidence=params.get('confidence_range', 0.5),
            social_energy=params.get('social_energy_range', 0.5),
            restlessness=params.get('restlessness_range', 0.5),
            harmony=params.get('harmony_range', 0.5)
        )
        
        return AgentRuntime(
            agent_id="test-agent",
            genome=genome,
            current_state=initial_state
        )
    
    async def run_mini_simulation(self, agent: AgentRuntime, params: Dict[str, float]) -> Dict[str, float]:
        """Run focused simulation to evaluate configuration"""
        scores = {
            'performance': 0.0,
            'stability': 0.0,
            'responsiveness': 0.0,
            'diversity': 0.0
        }
        
        # Track metrics over simulation steps
        state_history = []
        response_decisions = []
        consciousness_changes = 0
        
        # Simulate 100 steps (representing ~50 minutes)
        for step in range(100):
            try:
                # Generate varied environmental conditions
                env_factor = 0.3 + 0.4 * np.sin(2 * np.pi * step / 20) + 0.3 * random.random()
                
                env_input = EnvironmentalInput(
                    conversation_novelty=min(1.0, env_factor * params.get('novelty_sensitivity', 1.0)),
                    topic_expertise_match=random.uniform(0.2, 0.9),
                    conversation_harmony=env_factor,
                    direct_mention=random.random() < 0.1,
                    message_count_recent=max(1, int(env_factor * 10))
                )
                
                # Store previous state for comparison
                prev_state = agent.current_state
                prev_consciousness = prev_state.consciousness
                
                # Update agent
                new_state = dynamics_processor.update_state(agent, env_input, 30.0)
                agent.update_state(new_state)
                
                # Track changes
                if agent.current_state.consciousness != prev_consciousness:
                    consciousness_changes += 1
                
                state_history.append({
                    'step': step,
                    'curiosity': new_state.curiosity,
                    'confidence': new_state.confidence,
                    'social_energy': new_state.social_energy,
                    'restlessness': new_state.restlessness,
                    'harmony': new_state.harmony,
                    'consciousness': new_state.consciousness.value
                })
                
                # Simulate response decision
                would_respond = self.simulate_response_decision(new_state, env_input, params)
                response_decisions.append(would_respond)
                
            except Exception as e:
                self.logger.error(f"Error in mini-simulation step {step}: {e}")
                continue
        
        # Calculate scores
        if state_history:
            scores = self.calculate_performance_scores(state_history, response_decisions, consciousness_changes, params)
        
        return scores
    
    def simulate_response_decision(self, state: EmotionState, env_input: EnvironmentalInput, params: Dict[str, float]) -> bool:
        """Simulate whether agent would respond given current state"""
        # Weighted decision based on parameters
        decision_score = (
            state.curiosity * params.get('novelty_weight', 1.0) +
            state.confidence * params.get('confidence_weight', 1.0) +
            state.social_energy * params.get('social_weight', 0.8) +
            (0.5 if env_input.direct_mention else 0.0) +
            env_input.conversation_novelty * params.get('novelty_sensitivity', 1.0)
        ) / 4.0
        
        threshold = params.get('response_threshold', 0.5)
        return decision_score > threshold
    
    def calculate_performance_scores(self, history: List[Dict], responses: List[bool], 
                                   consciousness_changes: int, params: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance metrics from simulation history"""
        if not history:
            return {'performance': 0.0, 'stability': 0.0, 'responsiveness': 0.0, 'diversity': 0.0}
        
        df = pd.DataFrame(history)
        
        # Performance: How well agent maintains healthy emotional ranges
        emotional_cols = ['curiosity', 'confidence', 'social_energy', 'restlessness', 'harmony']
        
        # Ideal ranges (avoiding extremes)
        ideal_ranges = {
            'curiosity': (0.3, 0.8),
            'confidence': (0.4, 0.9),
            'social_energy': (0.3, 0.8),
            'restlessness': (0.2, 0.7),
            'harmony': (0.4, 0.8)
        }
        
        performance_scores = []
        for col in emotional_cols:
            values = df[col].values
            ideal_min, ideal_max = ideal_ranges[col]
            
            # Score based on how often values are in ideal range
            in_range = np.logical_and(values >= ideal_min, values <= ideal_max)
            range_score = np.mean(in_range)
            
            # Penalty for extreme values
            extreme_penalty = np.mean(np.logical_or(values < 0.1, values > 0.9)) * 0.5
            
            performance_scores.append(max(0, range_score - extreme_penalty))
        
        performance = np.mean(performance_scores)
        
        # Stability: Low variance in emotional states
        stability_scores = []
        for col in emotional_cols:
            variance = np.var(df[col])
            # Lower variance = higher stability (inverted and scaled)
            stability_score = max(0, 1.0 - variance * 2.0)
            stability_scores.append(stability_score)
        
        stability = np.mean(stability_scores)
        
        # Responsiveness: Appropriate response rate
        response_rate = np.mean(responses)
        # Ideal response rate is moderate (0.3-0.7)
        if 0.3 <= response_rate <= 0.7:
            responsiveness = 1.0
        else:
            responsiveness = max(0, 1.0 - abs(response_rate - 0.5) * 2.0)
        
        # Diversity: Appropriate consciousness level changes
        consciousness_change_rate = consciousness_changes / len(history)
        # Moderate change rate is ideal (0.05-0.15)
        if 0.05 <= consciousness_change_rate <= 0.15:
            diversity = 1.0
        else:
            diversity = max(0, 1.0 - abs(consciousness_change_rate - 0.1) * 5.0)
        
        return {
            'performance': performance,
            'stability': stability,
            'responsiveness': responsiveness,
            'diversity': diversity
        }
    
    async def random_search(self, n_samples: int = 100) -> List[Configuration]:
        """Perform random search across parameter space"""
        self.logger.info(f"🎲 Starting random search with {n_samples} samples...")
        
        bounds = self.generate_parameter_bounds()
        configurations = []
        
        for i in range(n_samples):
            # Generate random parameter values
            params = {}
            for j, (param_name, (min_val, max_val)) in enumerate(zip(self.param_names, bounds)):
                params[param_name] = random.uniform(min_val, max_val)
            
            # Evaluate configuration
            config = await self.evaluate_configuration(params)
            configurations.append(config)
            
            if i % 20 == 0:
                progress = (i / n_samples) * 100
                best_so_far = max(configurations, key=lambda x: x.overall_fitness).overall_fitness
                self.logger.info(f"Random search {progress:.1f}% - Best fitness: {best_so_far:.4f}")
        
        # Sort by fitness
        configurations.sort(key=lambda x: x.overall_fitness, reverse=True)
        self.configurations.extend(configurations)
        
        return configurations
    
    async def grid_search(self, resolution: int = 5) -> List[Configuration]:
        """Perform grid search with specified resolution"""
        self.logger.info(f"📊 Starting grid search with resolution {resolution}...")
        
        bounds = self.generate_parameter_bounds()
        
        # Create parameter grids
        param_grids = []
        for min_val, max_val in bounds:
            param_grids.append(np.linspace(min_val, max_val, resolution))
        
        # Generate all combinations
        param_combinations = list(product(*param_grids))
        total_combinations = len(param_combinations)
        
        self.logger.info(f"Evaluating {total_combinations} parameter combinations...")
        
        configurations = []
        
        for i, param_values in enumerate(param_combinations):
            params = dict(zip(self.param_names, param_values))
            
            config = await self.evaluate_configuration(params)
            configurations.append(config)
            
            if i % max(1, total_combinations // 10) == 0:
                progress = (i / total_combinations) * 100
                self.logger.info(f"Grid search {progress:.1f}% complete...")
        
        configurations.sort(key=lambda x: x.overall_fitness, reverse=True)
        self.configurations.extend(configurations)
        
        return configurations
    
    def objective_function(self, x: np.ndarray) -> float:
        """Objective function for optimization (to minimize, so return negative fitness)"""
        params = self.params_from_vector(x)
        
        # Run synchronous mini-simulation (simplified for optimization)
        try:
            agent = self.create_test_agent(params)
            fitness = self.quick_evaluate(agent, params)
            return -fitness  # Minimize negative fitness = maximize fitness
        except:
            return 1000.0  # High penalty for failed evaluations
    
    def quick_evaluate(self, agent: AgentRuntime, params: Dict[str, float]) -> float:
        """Quick synchronous evaluation for optimization"""
        scores = []
        
        for step in range(50):  # Shorter simulation for speed
            try:
                env_input = EnvironmentalInput(
                    conversation_novelty=random.uniform(0.2, 0.8),
                    conversation_harmony=random.uniform(0.3, 0.7),
                    direct_mention=random.random() < 0.1
                )
                
                new_state = dynamics_processor.update_state(agent, env_input, 30.0)
                agent.update_state(new_state)
                
                # Simple fitness based on balanced emotional state
                balance_score = 1.0 - np.std([
                    new_state.curiosity, new_state.confidence,
                    new_state.social_energy, new_state.restlessness, new_state.harmony
                ])
                
                scores.append(max(0, balance_score))
                
            except:
                scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    async def evolutionary_optimization(self, generations: int = 50, population_size: int = 30) -> List[Configuration]:
        """Evolutionary optimization using differential evolution"""
        self.logger.info(f"🧬 Starting evolutionary optimization: {generations} generations, population {population_size}")
        
        bounds = self.generate_parameter_bounds()
        
        # Use scipy's differential evolution
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=generations,
            popsize=population_size,
            disp=True,
            callback=self.evolution_callback
        )
        
        # Convert best result to configuration
        best_params = self.params_from_vector(result.x)
        best_config = await self.evaluate_configuration(best_params)
        
        self.logger.info(f"🏆 Evolutionary optimization complete! Best fitness: {best_config.overall_fitness:.4f}")
        
        return [best_config]
    
    def evolution_callback(self, xk, convergence):
        """Callback for evolutionary optimization progress"""
        if hasattr(self, '_evolution_step'):
            self._evolution_step += 1
        else:
            self._evolution_step = 1
        
        if self._evolution_step % 10 == 0:
            params = self.params_from_vector(xk)
            fitness = -self.objective_function(xk)  # Convert back to positive
            self.logger.info(f"Generation {self._evolution_step}: Best fitness = {fitness:.4f}")
    
    def analyze_parameter_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between parameters and performance"""
        if not self.configurations:
            return {}
        
        # Create DataFrame
        data = []
        for config in self.configurations:
            row = config.params.copy()
            row.update({
                'overall_fitness': config.overall_fitness,
                'performance': config.performance_score,
                'stability': config.stability_score,
                'responsiveness': config.responsiveness_score,
                'diversity': config.diversity_score
            })
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Calculate correlations
        target_cols = ['overall_fitness', 'performance', 'stability', 'responsiveness', 'diversity']
        param_cols = [col for col in df.columns if col not in target_cols]
        
        correlations = {}
        for target in target_cols:
            correlations[target] = df[param_cols].corrwith(df[target]).to_dict()
        
        return {
            'correlations': correlations,
            'dataframe': df,
            'summary_stats': df.describe()
        }
    
    def visualize_parameter_space(self, analysis_results: Dict[str, Any], save_plots: bool = True):
        """Create visualizations of the parameter space exploration"""
        df = analysis_results['dataframe']
        correlations = analysis_results['correlations']
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Correlation heatmap
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Overall fitness correlations
        corr_data = pd.Series(correlations['overall_fitness'])
        ax = axes[0, 0]
        sns.heatmap(corr_data.values.reshape(-1, 1), 
                   yticklabels=corr_data.index, 
                   xticklabels=['Overall Fitness'],
                   annot=True, cmap='RdBu_r', center=0, ax=ax)
        ax.set_title('Parameter Correlations with Overall Fitness')
        
        # Performance breakdown heatmap
        ax = axes[0, 1]
        perf_corr = pd.DataFrame({
            'Performance': correlations['performance'],
            'Stability': correlations['stability'],
            'Responsiveness': correlations['responsiveness'],
            'Diversity': correlations['diversity']
        })
        sns.heatmap(perf_corr.T, annot=True, cmap='RdBu_r', center=0, ax=ax)
        ax.set_title('Parameter Correlations with Performance Metrics')
        
        # 2. Parameter distribution plots
        ax = axes[0, 2]
        top_configs = df.nlargest(min(50, len(df)), 'overall_fitness')
        param_cols = [col for col in df.columns if 'range' in col or 'threshold' in col or 'weight' in col][:6]
        
        for i, param in enumerate(param_cols):
            ax.hist(top_configs[param], alpha=0.7, label=param, bins=20)
        ax.set_title('Top Configurations Parameter Distributions')
        ax.legend()
        
        # 3. Fitness vs key parameters scatter plots
        key_params = sorted(correlations['overall_fitness'].items(), 
                          key=lambda x: abs(x[1]), reverse=True)[:3]
        
        for i, (param, corr) in enumerate(key_params):
            ax = axes[1, i]
            scatter = ax.scatter(df[param], df['overall_fitness'], 
                               c=df['stability'], alpha=0.6, cmap='viridis')
            ax.set_xlabel(param)
            ax.set_ylabel('Overall Fitness')
            ax.set_title(f'{param} vs Fitness (corr={corr:.3f})')
            plt.colorbar(scatter, ax=ax, label='Stability')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('parameter_space_analysis.png', dpi=300, bbox_inches='tight')
            self.logger.info("📊 Parameter space visualization saved to parameter_space_analysis.png")
        
        plt.show()
        
        # 4. PCA visualization of parameter space
        if len(df) > 10:
            self.create_pca_visualization(df, save_plots)
    
    def create_pca_visualization(self, df: pd.DataFrame, save_plots: bool):
        """Create PCA visualization of the high-dimensional parameter space"""
        param_cols = [col for col in df.columns if col not in 
                     ['overall_fitness', 'performance', 'stability', 'responsiveness', 'diversity']]
        
        # Prepare data for PCA
        X = df[param_cols].values
        y = df['overall_fitness'].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=min(len(param_cols), 10))
        X_pca = pca.fit_transform(X_scaled)
        
        # Create PCA plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 2D PCA scatter
        ax = axes[0, 0]
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('PCA: Parameter Space Projection')
        plt.colorbar(scatter, ax=ax, label='Overall Fitness')
        
        # Explained variance
        ax = axes[0, 1]
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(range(1, len(cumvar) + 1), cumvar, 'bo-')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('PCA Explained Variance')
        ax.grid(True)
        
        # Component loadings
        ax = axes[1, 0]
        loadings = pca.components_[:2].T
        for i, param in enumerate(param_cols):
            ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                    head_width=0.02, head_length=0.02, alpha=0.7)
            ax.text(loadings[i, 0], loadings[i, 1], param, 
                   fontsize=8, ha='center')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA Component Loadings')
        ax.grid(True)
        
        # Clustering in PCA space
        ax = axes[1, 1]
        if len(X_pca) >= 8:  # Need enough points for clustering
            kmeans = KMeans(n_clusters=min(5, len(X_pca) // 2), random_state=42)
            clusters = kmeans.fit_predict(X_pca[:, :2])
            
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                      c='red', marker='x', s=200, linewidths=3)
            ax.set_title('K-Means Clustering in PCA Space')
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor clustering', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Clustering (Insufficient Data)')
        
        ax.set_xlabel(f'PC1')
        ax.set_ylabel(f'PC2')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('pca_parameter_analysis.png', dpi=300, bbox_inches='tight')
            self.logger.info("📊 PCA visualization saved to pca_parameter_analysis.png")
        
        plt.show()
        
        return {
            'pca_model': pca,
            'scaler': scaler,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': cumvar
        }
    
    def get_best_configurations(self, n: int = 10) -> List[Configuration]:
        """Get top N configurations"""
        if not self.configurations:
            return []
        
        return sorted(self.configurations, key=lambda x: x.overall_fitness, reverse=True)[:n]
    
    def save_results(self, filename: str = "hyperparameter_sweep_results.json"):
        """Save all results to file"""
        results = {
            'parameter_space': {
                'param_names': self.param_names,
                'bounds': self.generate_parameter_bounds()
            },
            'configurations': [
                {
                    'params': config.params,
                    'scores': {
                        'overall_fitness': config.overall_fitness,
                        'performance': config.performance_score,
                        'stability': config.stability_score,
                        'responsiveness': config.responsiveness_score,
                        'diversity': config.diversity_score
                    }
                }
                for config in self.configurations
            ],
            'best_configurations': [
                {
                    'rank': i + 1,
                    'params': config.params,
                    'fitness': config.overall_fitness
                }
                for i, config in enumerate(self.get_best_configurations(10))
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"📄 Results saved to {filename}")

async def main():
    """Run comprehensive hyperparameter sweep"""
    print("🚀 Starting Hyperparameter Sweep for Emotion Engine")
    print("="*80)
    
    # Define parameter space
    param_space = ParameterSpace()
    optimizer = HighDimensionalOptimizer(param_space)
    
    try:
        # Phase 1: Random search for broad exploration
        print("\n📍 Phase 1: Random Search (Broad Exploration)")
        random_configs = await optimizer.random_search(n_samples=50)
        print(f"✅ Random search completed. Best fitness: {random_configs[0].overall_fitness:.4f}")
        
        # Phase 2: Grid search around promising regions
        print("\n📍 Phase 2: Grid Search (Focused Exploration)")
        grid_configs = await optimizer.grid_search(resolution=4)
        print(f"✅ Grid search completed. Best fitness: {grid_configs[0].overall_fitness:.4f}")
        
        # Phase 3: Evolutionary optimization
        print("\n📍 Phase 3: Evolutionary Optimization (Fine-tuning)")
        evolution_configs = await optimizer.evolutionary_optimization(generations=30, population_size=20)
        print(f"✅ Evolutionary optimization completed. Best fitness: {evolution_configs[0].overall_fitness:.4f}")
        
        # Analysis and visualization
        print("\n📍 Phase 4: Analysis and Visualization")
        analysis = optimizer.analyze_parameter_correlations()
        
        print("\n🏆 TOP 5 CONFIGURATIONS:")
        print("-" * 60)
        for i, config in enumerate(optimizer.get_best_configurations(5)):
            print(f"Rank {i+1}: Fitness = {config.overall_fitness:.4f}")
            print(f"  Performance: {config.performance_score:.3f} | Stability: {config.stability_score:.3f}")
            print(f"  Responsiveness: {config.responsiveness_score:.3f} | Diversity: {config.diversity_score:.3f}")
            print(f"  Key params: {dict(list(config.params.items())[:3])}")
            print()
        
        # Create visualizations
        print("📊 Generating visualizations...")
        optimizer.visualize_parameter_space(analysis, save_plots=True)
        
        # Save results
        optimizer.save_results()
        
        print("\n🎉 HYPERPARAMETER SWEEP COMPLETED!")
        print(f"Total configurations evaluated: {len(optimizer.configurations)}")
        print(f"Best overall fitness: {optimizer.get_best_configurations(1)[0].overall_fitness:.4f}")
        
    except Exception as e:
        print(f"❌ Hyperparameter sweep failed: {e}")
        logging.error(f"Sweep error: {e}")

if __name__ == "__main__":
    asyncio.run(main())