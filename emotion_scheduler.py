#!/usr/bin/env python3
"""
Emotion Scheduler - Background processing system for emotional AI agents
Handles reflection loops, status updates, and emotional dynamics
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from emotional_virtual_users import EmotionalVirtualUserManager
from emotion_engine import (
    consciousness_manager, dynamics_processor, meta_reflector, 
    ReflectionContext, EnvironmentalInput, ReflectionType
)
from modules.social import social_orchestrator
from modules.cognitive import cognitive_processor

@dataclass
class ReflectionSchedule:
    """Schedule configuration for agent reflections"""
    quick_reflection_interval: float = 30.0    # Quick checks every 30s
    deep_reflection_interval: float = 120.0    # Deep analysis every 2min
    social_reflection_interval: float = 90.0   # Social awareness every 1.5min
    consciousness_check_interval: float = 60.0  # Consciousness updates every 1min
    
class EmotionScheduler:
    """Manages background emotional processing for all agents"""
    
    def __init__(self, manager: EmotionalVirtualUserManager):
        self.manager = manager
        self.schedule = ReflectionSchedule()
        self.is_running = False
        
        # Timing tracking
        self.last_quick_reflection = 0.0
        self.last_deep_reflection = 0.0
        self.last_social_reflection = 0.0
        self.last_consciousness_check = 0.0
        self.last_dynamics_update = 0.0
        
        # Performance monitoring
        self.reflection_stats = {
            'total_reflections': 0,
            'responses_triggered': 0,
            'wake_events': 0,
            'sleep_events': 0
        }
    
    async def start(self):
        """Start the emotion processing loops"""
        if self.is_running:
            return
        
        self.is_running = True
        logging.info("🧠 Starting emotion scheduler...")
        
        # Start all background tasks
        tasks = [
            asyncio.create_task(self._quick_reflection_loop()),
            asyncio.create_task(self._deep_reflection_loop()),
            asyncio.create_task(self._social_reflection_loop()),
            asyncio.create_task(self._consciousness_update_loop()),
            asyncio.create_task(self._dynamics_update_loop()),
            asyncio.create_task(self._status_monitoring_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logging.error(f"Error in emotion scheduler: {e}")
        finally:
            self.is_running = False
    
    async def stop(self):
        """Stop the emotion processing loops"""
        self.is_running = False
        logging.info("🛑 Stopping emotion scheduler...")
    
    async def _quick_reflection_loop(self):
        """Quick reflection loop - frequent, lightweight checks"""
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - self.last_quick_reflection >= self.schedule.quick_reflection_interval:
                    await self._perform_quick_reflections()
                    self.last_quick_reflection = current_time
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logging.error(f"Error in quick reflection loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _deep_reflection_loop(self):
        """Deep reflection loop - thorough analysis less frequently"""
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - self.last_deep_reflection >= self.schedule.deep_reflection_interval:
                    await self._perform_deep_reflections()
                    self.last_deep_reflection = current_time
                
                await asyncio.sleep(15.0)  # Check every 15 seconds
                
            except Exception as e:
                logging.error(f"Error in deep reflection loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _social_reflection_loop(self):
        """Social reflection loop - analyze social dynamics"""
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - self.last_social_reflection >= self.schedule.social_reflection_interval:
                    await self._perform_social_reflections()
                    self.last_social_reflection = current_time
                
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logging.error(f"Error in social reflection loop: {e}")
                await asyncio.sleep(20.0)
    
    async def _consciousness_update_loop(self):
        """Consciousness state update loop"""
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - self.last_consciousness_check >= self.schedule.consciousness_check_interval:
                    await self._update_consciousness_states()
                    self.last_consciousness_check = current_time
                
                await asyncio.sleep(20.0)  # Check every 20 seconds
                
            except Exception as e:
                logging.error(f"Error in consciousness update loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _dynamics_update_loop(self):
        """Emotional dynamics update loop"""
        while self.is_running:
            try:
                await self._update_emotional_dynamics()
                await asyncio.sleep(25.0)  # Update every 25 seconds
                
            except Exception as e:
                logging.error(f"Error in dynamics update loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _status_monitoring_loop(self):
        """Status monitoring and logging loop"""
        while self.is_running:
            try:
                await self._log_agent_status()
                await asyncio.sleep(120.0)  # Log status every 2 minutes
                
            except Exception as e:
                logging.error(f"Error in status monitoring loop: {e}")
                await asyncio.sleep(60.0)
    
    async def _perform_quick_reflections(self):
        """Perform quick reflections for all agents"""
        active_users = [u for u in self.manager.virtual_users.values() 
                       if u.active and u.runtime]
        
        if not active_users:
            return
        
        # Create shared context
        context_data = self._create_reflection_context()
        
        reflection_tasks = []
        for user in active_users:
            task = self._quick_agent_reflection(user, context_data)
            reflection_tasks.append(task)
        
        # Execute reflections concurrently
        results = await asyncio.gather(*reflection_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Quick reflection error for {active_users[i].discord_name}: {result}")
            elif result and result.should_respond:
                self.reflection_stats['responses_triggered'] += 1
        
        self.reflection_stats['total_reflections'] += len(active_users)
    
    async def _perform_deep_reflections(self):
        """Perform deep reflections for hyperfocused agents"""
        hyperfocused_users = [
            u for u in self.manager.virtual_users.values()
            if u.active and u.runtime and 
            u.runtime.current_state.consciousness.value >= 4  # ALERT or HYPERFOCUS
        ]
        
        if not hyperfocused_users:
            return
        
        context_data = self._create_reflection_context()
        
        for user in hyperfocused_users:
            try:
                result = await self._deep_agent_reflection(user, context_data)
                if result and result.should_respond:
                    # Log deep reflection insights
                    logging.info(f"🔍 Deep reflection: {user.discord_name} - {result.reasoning}")
            
            except Exception as e:
                logging.error(f"Deep reflection error for {user.discord_name}: {e}")
    
    async def _perform_social_reflections(self):
        """Perform social awareness reflections"""
        social_users = [
            u for u in self.manager.virtual_users.values()
            if u.active and u.runtime and u.runtime.is_awake()
        ]
        
        if len(social_users) < 2:  # Need at least 2 agents for social dynamics
            return
        
        try:
            # Update social relationships
            for user in social_users:
                other_agents = [u for u in social_users if u.model_id != user.model_id]
                social_orchestrator.update_relationship_dynamics(
                    user.model_id, 
                    [u.model_id for u in other_agents],
                    self.manager.conversation_context
                )
            
            # Check for social actions
            social_actions = social_orchestrator.evaluate_social_opportunities(
                [u.model_id for u in social_users],
                self.manager.conversation_context
            )
            
            if social_actions:
                logging.info(f"🤝 Social dynamics: {len(social_actions)} potential actions identified")
        
        except Exception as e:
            logging.error(f"Social reflection error: {e}")
    
    async def _update_consciousness_states(self):
        """Update consciousness states for all agents"""
        for user in self.manager.virtual_users.values():
            if not user.runtime or not user.active:
                continue
            
            try:
                # Create environmental input
                env_input = EnvironmentalInput.from_conversation_context(
                    self.manager.conversation_context,
                    user.model_id,
                    {u.model_id: u.emotional_state for u in self.manager.virtual_users.values() 
                     if u.emotional_state and u.model_id != user.model_id}
                )
                
                # Check for consciousness transitions
                old_level = user.runtime.current_state.consciousness
                consciousness_manager.update_consciousness(user.runtime, env_input)
                new_level = user.runtime.current_state.consciousness
                
                # Log significant changes
                if old_level != new_level:
                    if new_level.value > old_level.value:
                        self.reflection_stats['wake_events'] += 1
                        logging.info(f"⬆️ {user.discord_name}: {old_level.name} → {new_level.name}")
                    else:
                        self.reflection_stats['sleep_events'] += 1
                        logging.info(f"⬇️ {user.discord_name}: {old_level.name} → {new_level.name}")
            
            except Exception as e:
                logging.error(f"Consciousness update error for {user.model_id}: {e}")
    
    async def _update_emotional_dynamics(self):
        """Update emotional dynamics for all agents"""
        current_time = time.time()
        delta_time = current_time - self.last_dynamics_update
        self.last_dynamics_update = current_time
        
        # Update individual agent dynamics
        for user in self.manager.virtual_users.values():
            if not user.runtime or not user.active:
                continue
            
            try:
                # Create environmental input
                env_input = EnvironmentalInput.from_conversation_context(
                    self.manager.conversation_context,
                    user.model_id,
                    {u.model_id: u.emotional_state for u in self.manager.virtual_users.values()
                     if u.emotional_state and u.model_id != user.model_id}
                )
                
                # Update emotional state
                new_state = dynamics_processor.update_state(user.runtime, env_input, delta_time)
                user.runtime.update_state(new_state)
                
                # Update cognitive state
                cognitive_processor.update_cognitive_state(
                    user.model_id, user.runtime.current_state,
                    {'conversation': self.manager.conversation_context}, delta_time
                )
            
            except Exception as e:
                logging.error(f"Dynamics update error for {user.model_id}: {e}")
        
        # Apply cross-agent effects
        try:
            awake_runtimes = {
                u.model_id: u.runtime for u in self.manager.virtual_users.values()
                if u.runtime and u.active and u.runtime.is_awake()
            }
            
            if len(awake_runtimes) > 1:
                from emotion_engine.dynamics import cross_agent_dynamics
                updated_states = cross_agent_dynamics.apply_cross_agent_effects(awake_runtimes)
                
                # Apply cross-agent updates
                for model_id, new_state in updated_states.items():
                    if model_id in self.manager.virtual_users:
                        self.manager.virtual_users[model_id].runtime.update_state(new_state)
        
        except Exception as e:
            logging.error(f"Cross-agent dynamics error: {e}")
    
    async def _log_agent_status(self):
        """Log current status of all agents"""
        try:
            summary = self.manager.get_emotional_summary()
            
            # Create status summary
            status_lines = [
                f"📊 Emotion Engine Status: {summary['awake_agents']}/{summary['active_agents']} agents awake"
            ]
            
            # Add top emotional states
            for model_id, state_info in list(summary['agent_states'].items())[:5]:
                emoji = {'HYPERFOCUS': '🔥', 'ALERT': '✨', 'DROWSY': '😪', 
                        'REM': '😴', 'DEEP_SLEEP': '💤', 'COMA': '💀'}.get(
                    state_info['consciousness'], '❓')
                status_lines.append(f"  {emoji} {state_info['name']}: {state_info['status']}")
            
            # Add performance stats
            status_lines.append(
                f"🔄 Reflections: {self.reflection_stats['total_reflections']}, "
                f"Responses: {self.reflection_stats['responses_triggered']}, "
                f"Wake/Sleep: {self.reflection_stats['wake_events']}/{self.reflection_stats['sleep_events']}"
            )
            
            logging.info("\n".join(status_lines))
            
        except Exception as e:
            logging.error(f"Status logging error: {e}")
    
    def _create_reflection_context(self) -> Dict[str, Any]:
        """Create shared context for reflections"""
        return {
            'conversation_history': self.manager.conversation_context,
            'active_agents': {
                u.model_id: u.emotional_state 
                for u in self.manager.virtual_users.values() 
                if u.emotional_state and u.active
            },
            'timestamp': time.time(),
            'message_count': len(self.manager.conversation_context)
        }
    
    async def _quick_agent_reflection(self, user, context_data):
        """Perform quick reflection for a single agent"""
        try:
            # Create reflection context
            context = ReflectionContext.from_conversation(
                context_data['conversation_history'],
                user.model_id,
                context_data['active_agents']
            )
            
            # Perform quick reflection
            result = meta_reflector.perform_reflection(
                user.runtime, context, ReflectionType.QUICK_CHECK
            )
            
            return result
            
        except Exception as e:
            logging.error(f"Quick reflection error for {user.model_id}: {e}")
            return None
    
    async def _deep_agent_reflection(self, user, context_data):
        """Perform deep reflection for a single agent"""
        try:
            # Create reflection context
            context = ReflectionContext.from_conversation(
                context_data['conversation_history'],
                user.model_id,
                context_data['active_agents']
            )
            
            # Perform deep reflection
            result = meta_reflector.perform_reflection(
                user.runtime, context, ReflectionType.DEEP_ANALYSIS
            )
            
            return result
            
        except Exception as e:
            logging.error(f"Deep reflection error for {user.model_id}: {e}")
            return None
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler performance statistics"""
        return {
            'is_running': self.is_running,
            'reflection_stats': self.reflection_stats.copy(),
            'schedule': {
                'quick_interval': self.schedule.quick_reflection_interval,
                'deep_interval': self.schedule.deep_reflection_interval,
                'social_interval': self.schedule.social_reflection_interval,
                'consciousness_interval': self.schedule.consciousness_check_interval
            },
            'last_updates': {
                'quick_reflection': self.last_quick_reflection,
                'deep_reflection': self.last_deep_reflection,
                'social_reflection': self.last_social_reflection,
                'consciousness_check': self.last_consciousness_check,
                'dynamics_update': self.last_dynamics_update
            }
        }
    
    def adjust_reflection_frequency(self, activity_level: float):
        """Dynamically adjust reflection frequency based on activity"""
        base_intervals = ReflectionSchedule()
        
        # Higher activity = more frequent reflections
        activity_multiplier = max(0.5, 1.0 - activity_level * 0.4)
        
        self.schedule.quick_reflection_interval = base_intervals.quick_reflection_interval * activity_multiplier
        self.schedule.social_reflection_interval = base_intervals.social_reflection_interval * activity_multiplier
        
        # Deep reflections less affected by activity
        deep_multiplier = max(0.7, 1.0 - activity_level * 0.2)
        self.schedule.deep_reflection_interval = base_intervals.deep_reflection_interval * deep_multiplier

# Standalone scheduler for testing
async def main():
    """Main function for testing the scheduler"""
    logging.basicConfig(level=logging.INFO)
    
    # Create mock manager (would normally be injected)
    print("🧠 Emotion Scheduler Test Mode")
    print("In production, this would be integrated with EmotionalVirtualUserManager")
    
    # Mock scheduler
    class MockManager:
        def __init__(self):
            self.virtual_users = {}
            self.conversation_context = []
        
        def get_emotional_summary(self):
            return {'awake_agents': 0, 'active_agents': 0, 'agent_states': {}}
    
    mock_manager = MockManager()
    scheduler = EmotionScheduler(mock_manager)
    
    try:
        await scheduler.start()
    except KeyboardInterrupt:
        await scheduler.stop()
        print("\n🛑 Scheduler stopped")

if __name__ == "__main__":
    asyncio.run(main())