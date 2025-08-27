#!/usr/bin/env python3
"""
🧠 Real-Time Emotion Parameter Plotting System
Continuously plots each agent's emotion parameters with 1-second sampling
Saves stacked multi-agent charts every 15 seconds
"""

import time
import asyncio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from pathlib import Path
import threading
from dataclasses import dataclass, field

from emotion_engine.monitoring import production_monitor
from genome.types import EmotionState

@dataclass
class PlotSample:
    """Single emotion parameter sample"""
    timestamp: datetime
    curiosity: float
    confidence: float
    social_energy: float
    restlessness: float
    harmony: float
    expertise: float
    novelty: float
    consciousness: float

class EmotionPlotter:
    """Real-time emotion parameter plotting system"""
    
    def __init__(self, plots_dir: str = "charts"):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Data storage - one deque per agent (max 15 minutes of 1s samples)
        self.agent_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=900))
        
        # Plot configuration
        self.sample_interval = 1.0  # seconds
        self.plot_interval = 15.0   # seconds
        self.log_interval = 3.0     # seconds for CSV logging
        self.plot_window_minutes = 15  # minutes of history to show
        
        # Control flags
        self.running = False
        self.sample_task: Optional[asyncio.Task] = None
        self.plot_task: Optional[asyncio.Task] = None
        self.log_task: Optional[asyncio.Task] = None
        
        # Run identifier
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.plot_counter = 0
        
        # CSV logging setup
        self.csv_files: Dict[str, Path] = {}
        self.csv_headers_written: Dict[str, bool] = {}
        
        self.logger = logging.getLogger("EmotionPlotter")
        
        # Color scheme for different parameters
        self.param_colors = {
            'curiosity': '#FF6B6B',      # Red
            'confidence': '#4ECDC4',     # Teal  
            'social_energy': '#45B7D1',  # Blue
            'restlessness': '#FFA07A',   # Orange
            'harmony': '#98D8C8',        # Green
            'expertise': '#F7DC6F',      # Yellow
            'novelty': '#BB8FCE',        # Purple
            'consciousness': '#85C1E9'    # Light Blue
        }
        
    def start(self):
        """Start real-time plotting system"""
        if self.running:
            self.logger.warning("Plotter already running")
            return
        
        self.running = True
        self.sample_task = asyncio.create_task(self._sample_loop())
        self.plot_task = asyncio.create_task(self._plot_loop())
        self.log_task = asyncio.create_task(self._log_loop())
        
        self.logger.info(f"Started emotion plotter (run_id: {self.run_id})")
        self.logger.info(f"Sample interval: {self.sample_interval}s, Plot interval: {self.plot_interval}s, Log interval: {self.log_interval}s")
        
    async def stop(self):
        """Stop plotting system"""
        self.running = False
        
        if self.sample_task:
            self.sample_task.cancel()
            try:
                await self.sample_task
            except asyncio.CancelledError:
                pass
                
        if self.plot_task:
            self.plot_task.cancel()
            try:
                await self.plot_task
            except asyncio.CancelledError:
                pass
                
        if self.log_task:
            self.log_task.cancel()
            try:
                await self.log_task
            except asyncio.CancelledError:
                pass
        
        # Final plot on shutdown
        await self._generate_plot()
        
        self.logger.info("Stopped emotion plotter")
    
    async def _sample_loop(self):
        """Sample emotion parameters every second"""
        while self.running:
            try:
                await asyncio.sleep(self.sample_interval)
                
                if not self.running:
                    break
                
                # Sample all agents
                timestamp = datetime.now()
                
                for agent_id, metrics in production_monitor.agent_metrics.items():
                    if metrics.state_history:
                        # Get latest emotion state
                        _, emotion_state = metrics.state_history[-1]
                        
                        sample = PlotSample(
                            timestamp=timestamp,
                            curiosity=emotion_state.curiosity,
                            confidence=emotion_state.confidence,
                            social_energy=emotion_state.social_energy,
                            restlessness=emotion_state.restlessness,
                            harmony=emotion_state.harmony,
                            expertise=emotion_state.expertise,
                            novelty=emotion_state.novelty,
                            consciousness=float(emotion_state.consciousness.value) / 5.0  # Normalize to 0-1
                        )
                        
                        self.agent_data[agent_id].append(sample)
                
                # Log sampling activity
                if len(self.agent_data) > 0:
                    total_samples = sum(len(deque) for deque in self.agent_data.values())
                    self.logger.debug(f"Sampled {len(self.agent_data)} agents, total samples: {total_samples}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in sample loop: {e}")
                await asyncio.sleep(1)
    
    async def _plot_loop(self):
        """Generate plots every 15 seconds"""
        while self.running:
            try:
                await asyncio.sleep(self.plot_interval)
                
                if not self.running:
                    break
                
                await self._generate_plot()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in plot loop: {e}")
                await asyncio.sleep(5)
    
    async def _log_loop(self):
        """Log emotion parameters to CSV every 3 seconds"""
        while self.running:
            try:
                await asyncio.sleep(self.log_interval)
                
                if not self.running:
                    break
                
                await self._log_to_csv()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in log loop: {e}")
                await asyncio.sleep(3)
    
    async def _log_to_csv(self):
        """Log current emotion parameters to CSV files (one per agent)"""
        timestamp = datetime.now()
        
        for agent_id, metrics in production_monitor.agent_metrics.items():
            if not metrics.state_history:
                continue
                
            # Get latest emotion state
            _, emotion_state = metrics.state_history[-1]
            
            # Get or create CSV file for this agent
            if agent_id not in self.csv_files:
                # Extract model name from agent_id if possible (e.g., "anthropic/claude-opus-4.1" -> "opus")
                model_name = agent_id.split('/')[-1].split('-')[0] if '/' in agent_id else agent_id
                filename = f"{self.run_id}.{model_name}.log"
                self.csv_files[agent_id] = self.plots_dir / filename
                self.csv_headers_written[agent_id] = False
            
            csv_file = self.csv_files[agent_id]
            
            # Write header if first time
            if not self.csv_headers_written[agent_id]:
                with open(csv_file, 'w') as f:
                    f.write("timestamp,curiosity,confidence,social_energy,restlessness,harmony,expertise,novelty,consciousness\n")
                self.csv_headers_written[agent_id] = True
            
            # Write data row
            with open(csv_file, 'a') as f:
                consciousness_normalized = float(emotion_state.consciousness.value) / 5.0
                f.write(f"{timestamp.isoformat()},"
                       f"{emotion_state.curiosity:.6f},"
                       f"{emotion_state.confidence:.6f},"
                       f"{emotion_state.social_energy:.6f},"
                       f"{emotion_state.restlessness:.6f},"
                       f"{emotion_state.harmony:.6f},"
                       f"{emotion_state.expertise:.6f},"
                       f"{emotion_state.novelty:.6f},"
                       f"{consciousness_normalized:.6f}\n")
    
    async def _generate_plot(self):
        """Generate stacked multi-agent emotion parameter plots"""
        if not self.agent_data:
            self.logger.warning("No agent data available for plotting")
            return
        
        try:
            # Run plotting in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._create_plot)
            
        except Exception as e:
            self.logger.error(f"Error generating plot: {e}")
    
    def _create_plot(self):
        """Create the actual matplotlib plot"""
        plt.style.use('dark_background')
        
        # Calculate subplot layout
        num_agents = len(self.agent_data)
        if num_agents == 0:
            return
        
        # Create figure with subplots (one per agent)
        fig, axes = plt.subplots(num_agents, 1, figsize=(12, 4 * num_agents))
        if num_agents == 1:
            axes = [axes]  # Ensure axes is always a list
        
        fig.suptitle(
            f'🧠 Real-Time Agent Emotion Parameters (Run: {self.run_id})', 
            fontsize=16, 
            fontweight='bold',
            color='white'
        )
        
        # Plot each agent's data
        for i, (agent_id, samples) in enumerate(self.agent_data.items()):
            if not samples:
                continue
                
            ax = axes[i]
            
            # Extract data for plotting
            timestamps = [s.timestamp for s in samples]
            
            # Plot each parameter
            params = {
                'Curiosity': [s.curiosity for s in samples],
                'Confidence': [s.confidence for s in samples], 
                'Social Energy': [s.social_energy for s in samples],
                'Restlessness': [s.restlessness for s in samples],
                'Harmony': [s.harmony for s in samples],
                'Expertise': [s.expertise for s in samples],
                'Novelty': [s.novelty for s in samples],
                'Consciousness': [s.consciousness for s in samples]
            }
            
            # Plot lines
            for param_name, values in params.items():
                color = self.param_colors.get(param_name.lower().replace(' ', '_'), '#FFFFFF')
                ax.plot(timestamps, values, label=param_name, color=color, linewidth=2, alpha=0.8)
            
            # Formatting
            ax.set_title(f'Agent: {agent_id}', fontsize=14, fontweight='bold', color='white')
            ax.set_ylabel('Parameter Value (0-1)', color='white')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
            
            # Time formatting
            if timestamps:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Set colors
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
        
        # Add timestamp and stats
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_samples = sum(len(samples) for samples in self.agent_data.values())
        
        fig.text(0.02, 0.02, 
                f'Generated: {current_time} | Total Samples: {total_samples} | Agents: {num_agents}',
                fontsize=10, color='gray')
        
        # Save plot
        self.plot_counter += 1
        filename = f"emotion_params_{self.run_id}_{self.plot_counter:04d}.png"
        filepath = self.plots_dir / filename
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()  # Important: close to free memory
        
        self.logger.info(f"Saved plot: {filepath} ({num_agents} agents, {total_samples} samples)")


# Global plotter instance
emotion_plotter = EmotionPlotter()


async def start_emotion_plotting():
    """Start the emotion plotting system"""
    emotion_plotter.start()
    logging.info("🧠 Started real-time emotion parameter plotting")


async def stop_emotion_plotting():
    """Stop the emotion plotting system"""
    await emotion_plotter.stop()
    logging.info("🧠 Stopped emotion parameter plotting")


if __name__ == "__main__":
    # Test the plotter
    import asyncio
    
    async def test():
        await start_emotion_plotting()
        await asyncio.sleep(60)  # Run for 1 minute
        await stop_emotion_plotting()
    
    asyncio.run(test())