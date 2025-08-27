#!/usr/bin/env python3
"""
💾 Agent State Persistence System
Saves and restores agent emotional states, evolution history, and performance data
"""

import json
import pickle
import os
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging
import asyncio

from emotion_engine.monitoring import production_monitor
from evolution.mutation_engine import mutation_engine
from evolution.evolution_scheduler import evolution_scheduler


class AgentStatePersistence:
    """Manages saving and loading of agent states"""
    
    def __init__(self, data_dir: str = "/tmp/llmcord_state"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("StatePersistence")
        
        # File paths
        self.agent_states_file = self.data_dir / "agent_states.json"
        self.monitoring_data_file = self.data_dir / "monitoring_data.json" 
        self.evolution_data_file = self.data_dir / "evolution_data.json"
        self.metadata_file = self.data_dir / "metadata.json"
    
    def save_agent_states(self, agents: Dict[str, Any]) -> bool:
        """Save current agent emotional states"""
        
        try:
            state_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'agents': {}
            }
            
            for agent_id, runtime in agents.items():
                if hasattr(runtime, 'current_state'):
                    state_data['agents'][agent_id] = {
                        'current_state': {
                            'curiosity': runtime.current_state.curiosity,
                            'confidence': runtime.current_state.confidence,
                            'social_energy': runtime.current_state.social_energy,
                            'restlessness': runtime.current_state.restlessness,
                            'harmony': runtime.current_state.harmony,
                            'consciousness': runtime.current_state.consciousness.value,
                            'expertise': runtime.current_state.expertise,
                            'novelty': runtime.current_state.novelty,
                            'last_tick': runtime.current_state.last_tick
                        },
                        'runtime_data': {
                            'last_reflection': getattr(runtime, 'last_reflection', 0.0),
                            'last_spoke': getattr(runtime, 'last_spoke', 0.0),
                            'wake_time': getattr(runtime, 'wake_time', None)
                        }
                    }
            
            with open(self.agent_states_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.info(f"Saved states for {len(state_data['agents'])} agents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save agent states: {e}")
            return False
    
    def load_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """Load saved agent states"""
        
        if not self.agent_states_file.exists():
            self.logger.info("No saved agent states found")
            return {}
        
        try:
            with open(self.agent_states_file, 'r') as f:
                state_data = json.load(f)
            
            agents_data = state_data.get('agents', {})
            saved_time = state_data.get('timestamp', 'unknown')
            
            self.logger.info(f"Loaded states for {len(agents_data)} agents (saved: {saved_time})")
            return agents_data
            
        except Exception as e:
            self.logger.error(f"Failed to load agent states: {e}")
            return {}
    
    def save_monitoring_data(self) -> bool:
        """Save monitoring and performance data"""
        
        try:
            monitoring_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'agent_metrics': {},
                'performance_history': list(production_monitor.performance_history)
            }
            
            # Save agent metrics (excluding complex objects)
            for agent_id, metrics in production_monitor.agent_metrics.items():
                monitoring_data['agent_metrics'][agent_id] = {
                    'agent_id': metrics.agent_id,
                    'state_history_count': len(metrics.state_history),
                    'response_history_count': len(metrics.response_history),
                    'consciousness_transitions_count': len(metrics.consciousness_transitions),
                    'last_performance': metrics.last_performance.__dict__ if metrics.last_performance else None
                }
            
            with open(self.monitoring_data_file, 'w') as f:
                json.dump(monitoring_data, f, indent=2)
            
            self.logger.info(f"Saved monitoring data for {len(monitoring_data['agent_metrics'])} agents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save monitoring data: {e}")
            return False
    
    def load_monitoring_data(self) -> Dict[str, Any]:
        """Load monitoring data"""
        
        if not self.monitoring_data_file.exists():
            self.logger.info("No saved monitoring data found")
            return {}
        
        try:
            with open(self.monitoring_data_file, 'r') as f:
                monitoring_data = json.load(f)
            
            self.logger.info(f"Loaded monitoring data from {monitoring_data.get('timestamp', 'unknown')}")
            return monitoring_data
            
        except Exception as e:
            self.logger.error(f"Failed to load monitoring data: {e}")
            return {}
    
    def save_evolution_data(self) -> bool:
        """Save evolution and mutation history"""
        
        try:
            evolution_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'mutation_history': {},
                'evolution_stats': evolution_scheduler.evolution_stats.copy()
            }
            
            # Save mutation history (convert sets to lists for JSON)
            for agent_id, history in mutation_engine.mutation_history.items():
                evolution_data['mutation_history'][agent_id] = {
                    'mutations': history.mutations,
                    'performance_before': history.performance_before,
                    'performance_after': history.performance_after,
                    'successful_mutations': history.successful_mutations,
                    'failed_mutations': history.failed_mutations,
                    'last_mutation_time': history.last_mutation_time
                }
            
            # Convert sets to lists for JSON serialization
            if 'agents_evolved' in evolution_data['evolution_stats']:
                evolution_data['evolution_stats']['agents_evolved'] = list(
                    evolution_data['evolution_stats']['agents_evolved']
                )
            
            with open(self.evolution_data_file, 'w') as f:
                json.dump(evolution_data, f, indent=2)
            
            self.logger.info(f"Saved evolution data for {len(evolution_data['mutation_history'])} agents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save evolution data: {e}")
            return False
    
    def load_evolution_data(self) -> Dict[str, Any]:
        """Load evolution data"""
        
        if not self.evolution_data_file.exists():
            self.logger.info("No saved evolution data found")
            return {}
        
        try:
            with open(self.evolution_data_file, 'r') as f:
                evolution_data = json.load(f)
            
            self.logger.info(f"Loaded evolution data from {evolution_data.get('timestamp', 'unknown')}")
            return evolution_data
            
        except Exception as e:
            self.logger.error(f"Failed to load evolution data: {e}")
            return {}
    
    def save_all_state(self, agents: Dict[str, Any]) -> bool:
        """Save complete system state"""
        
        self.logger.info("Saving complete system state...")
        
        success = True
        success &= self.save_agent_states(agents)
        success &= self.save_monitoring_data()
        success &= self.save_evolution_data()
        
        # Save metadata
        metadata = {
            'save_timestamp': datetime.utcnow().isoformat(),
            'agent_count': len(agents),
            'system_uptime': getattr(production_monitor, 'monitoring_start', 0),
            'version': '1.0'
        }
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            success = False
        
        if success:
            self.logger.info("✅ Complete system state saved successfully")
        else:
            self.logger.error("❌ Some components failed to save")
        
        return success
    
    def restore_agent_state(self, runtime, saved_data: Dict[str, Any]):
        """Restore a single agent's state"""
        
        try:
            # Restore emotional state
            if 'current_state' in saved_data:
                state_data = saved_data['current_state']
                
                runtime.current_state.curiosity = state_data.get('curiosity', 0.5)
                runtime.current_state.confidence = state_data.get('confidence', 0.5)
                runtime.current_state.social_energy = state_data.get('social_energy', 0.5)
                runtime.current_state.restlessness = state_data.get('restlessness', 0.0)
                runtime.current_state.harmony = state_data.get('harmony', 0.5)
                runtime.current_state.expertise = state_data.get('expertise', 0.0)
                runtime.current_state.novelty = state_data.get('novelty', 0.0)
                runtime.current_state.last_tick = state_data.get('last_tick', 0.0)
                
                # Restore consciousness level
                from genome.types import ConsciousnessLevel
                consciousness_value = state_data.get('consciousness', 3)
                runtime.current_state.consciousness = ConsciousnessLevel(consciousness_value)
            
            # Restore runtime data
            if 'runtime_data' in saved_data:
                runtime_data = saved_data['runtime_data']
                runtime.last_reflection = runtime_data.get('last_reflection', 0.0)
                runtime.last_spoke = runtime_data.get('last_spoke', 0.0)
                runtime.wake_time = runtime_data.get('wake_time')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore agent state: {e}")
            return False
    
    def restore_all_state(self, agents: Dict[str, Any]) -> bool:
        """Restore complete system state"""
        
        self.logger.info("Restoring system state...")
        
        # Load saved data
        agent_states = self.load_agent_states()
        monitoring_data = self.load_monitoring_data()
        evolution_data = self.load_evolution_data()
        
        restored_count = 0
        
        # Restore agent states
        for agent_id, runtime in agents.items():
            if agent_id in agent_states:
                if self.restore_agent_state(runtime, agent_states[agent_id]):
                    restored_count += 1
                    self.logger.info(f"Restored state for agent: {agent_id}")
        
        # Restore evolution data
        if evolution_data and 'mutation_history' in evolution_data:
            from evolution.mutation_engine import MutationHistory
            
            for agent_id, history_data in evolution_data['mutation_history'].items():
                history = MutationHistory(
                    mutations=history_data['mutations'],
                    performance_before=history_data['performance_before'],
                    performance_after=history_data['performance_after'],
                    successful_mutations=history_data['successful_mutations'],
                    failed_mutations=history_data['failed_mutations'],
                    last_mutation_time=history_data['last_mutation_time']
                )
                mutation_engine.mutation_history[agent_id] = history
            
            self.logger.info(f"Restored evolution data for {len(evolution_data['mutation_history'])} agents")
        
        # Restore evolution stats
        if evolution_data and 'evolution_stats' in evolution_data:
            stats = evolution_data['evolution_stats']
            evolution_scheduler.evolution_stats.update(stats)
            # Convert list back to set
            if 'agents_evolved' in stats:
                evolution_scheduler.evolution_stats['agents_evolved'] = set(stats['agents_evolved'])
        
        self.logger.info(f"✅ Restored state for {restored_count}/{len(agents)} agents")
        return restored_count > 0
    
    def cleanup_old_states(self, keep_days: int = 7):
        """Clean up old state files"""
        
        try:
            cutoff_time = datetime.utcnow().timestamp() - (keep_days * 24 * 3600)
            
            for file_path in self.data_dir.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    self.logger.info(f"Cleaned up old state file: {file_path.name}")
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup old states: {e}")


# Global instance
state_persistence = AgentStatePersistence()


# Auto-save functionality
class AutoStateSaver:
    """Automatically saves state at regular intervals"""
    
    def __init__(self, interval_seconds: int = 30):
        self.interval_seconds = interval_seconds
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.agents: Dict[str, Any] = {}
        self.logger = logging.getLogger("AutoStateSaver")
    
    def set_agents(self, agents: Dict[str, Any]):
        """Set the agents dictionary to save"""
        self.agents = agents
    
    async def start(self):
        """Start auto-saving"""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._save_loop())
        self.logger.info(f"Started auto-save every {self.interval_seconds} seconds")
    
    async def stop(self):
        """Stop auto-saving"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        # Final save on shutdown
        if self.agents:
            state_persistence.save_all_state(self.agents)
        
        self.logger.info("Stopped auto-save")
    
    async def _save_loop(self):
        """Auto-save loop"""
        while self.running:
            try:
                await asyncio.sleep(self.interval_seconds)
                
                if self.agents:
                    state_persistence.save_all_state(self.agents)
                    self.logger.info("Auto-saved system state")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in auto-save loop: {e}")


# Global auto-saver (30 second intervals)
auto_saver = AutoStateSaver(interval_seconds=30)


# Integration functions
async def setup_state_persistence(agents: Dict[str, Any]):
    """Setup state persistence for the bot"""
    
    # Set agents for auto-saver
    auto_saver.set_agents(agents)
    
    # Restore previous state
    state_persistence.restore_all_state(agents)
    
    # Start auto-saving
    await auto_saver.start()


async def shutdown_state_persistence():
    """Shutdown state persistence"""
    await auto_saver.stop()