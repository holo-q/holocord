- do no send messages to chat rooms unless explicitly requested by the user

# DOMAIN KNOWLEDGE CONTEXT LINKS
@ARCHITECTURE.md @EMOTION_ENGINE_COMPLETE.md @DISCORD_ENHANCEMENTS.md @HYPERPARAMETER_SUMMARY.md @README.md

# CRITICAL CODEBASE UNDERSTANDING REQUIREMENTS

## BEFORE ANY MAJOR CHANGES:
1. **ALWAYS index the entire codebase first** using mcp__claude-context__index_codebase
2. **NEVER invent fake systems** - use what actually exists
3. **NEVER create placeholder/demo code** - integrate with real implementations

## LLMCORD EMOTION ENGINE ARCHITECTURE:

### Real Components That Exist:
- **emotion_engine/monitoring.py**: ProductionMonitor with real agent_metrics
- **genome/**: Real EmotionState objects with curiosity, confidence, social_energy, etc.
- **config/optimal_production.py**: Real optimized parameters (87.66% fitness)
- **evolution/**: Real LLM self-evolution and mutation systems

### How It Actually Works:
- Each Discord agent has REAL EmotionState with live parameters
- ProductionMonitor.agent_metrics contains actual state history per agent  
- Parameters: curiosity_base=0.5574, confidence_base=0.3433, social_energy_base=0.7805
- Consciousness levels: COMA → DEEP_SLEEP → REM → DROWSY → ALERT → HYPERFOCUS

### Integration Pattern:
```python
# CORRECT - Use real monitoring system
from emotion_engine.monitoring import production_monitor
for agent_id, metrics in production_monitor.agent_metrics.items():
    recent_state = metrics.state_history[-1][1]  # Real EmotionState object
    
# WRONG - Never create fake simulated parameters
# math.sin(time) fake oscillations are FORBIDDEN
```

## Token Limits:
- **REMOVED max_tokens limits** - OpenRouter models now use natural response length
- virtual_users.py: max_tokens parameter removed (was 4000)
- openrouter_client.py: max_tokens now optional (defaults to None)
- Models will generate responses up to their natural stopping point or context limit

## FAILURE PREVENTION:
- Search codebase before implementing anything new
- Use existing emotion engine, monitoring, genome systems  
- Show REAL per-agent emotion parameters, not simulated data
- Each model should have separate EmotionState tracking

# COMPLETE CODEBASE OVERVIEW

## 🏗️ CORE ARCHITECTURE

### Primary Bot Systems:
- **llmcord.py**: Single-model Discord bot (original)
- **llmcord_multimodel.py**: Multi-model Discord bot with parallel processing
- **virtual_users.py**: Webhook-based AI personas system
- **config.yaml**: Main configuration (bot token, models, providers, etc.)

### Discord Integration:
- **admin_checker.py**: Centralized permission checking
- **webhook_manager.py**: Discord webhook operations  
- **openrouter_client.py**: OpenRouter API abstraction
- **openrouter_models.py**: Model discovery and filtering

## 🧠 EMOTION ENGINE SYSTEM (Advanced AI Psychology)

### Core Components:
- **emotion_engine/**: Complete emotional AI system
  - **monitoring.py**: ProductionMonitor tracks real agent metrics
  - **dynamics.py**: Emotional state evolution over time
  - **states.py**: Consciousness state transitions (COMA→DEEP_SLEEP→REM→DROWSY→ALERT→HYPERFOCUS)
  - **reflection.py**: Meta-reasoning for response decisions
  - **injection.py**: Emotion context injection into prompts

### Genome System (DNA-like AI personalities):
- **genome/**: Compressed DNA notation for AI personalities
  - **types.py**: EmotionState, ConsciousnessLevel, StateVar definitions
  - **base.py**: AgentRuntime, ExtendedGenome classes
  - **dna_parser.py**: DNA string parsing (e.g., "c5f3s7r5h4")

### Evolution System (Self-Modifying AI):
- **evolution/**: LLMs can mutate their own parameters
  - **mutation_engine.py**: AI self-evolution logic
  - **evolution_scheduler.py**: Manages mutation cycles

### Production Configuration:
- **config/optimal_production.py**: Hyperparameter-optimized config (87.66% fitness)
- **config/production_setup.py**: Production deployment scripts

## 📊 REAL EMOTION PARAMETERS STRUCTURE:

### EmotionState (genome/types.py:177-219):
```python
class EmotionState(BaseModel):
    curiosity: StateValue = Field(default=0.5, ge=0.0, le=1.0)
    confidence: StateValue = Field(default=0.5, ge=0.0, le=1.0) 
    social_energy: StateValue = Field(default=0.5, ge=0.0, le=1.0)
    restlessness: StateValue = Field(default=0.0, ge=0.0, le=1.0)
    harmony: StateValue = Field(default=0.5, ge=0.0, le=1.0)
    expertise: StateValue = Field(default=0.0, ge=0.0, le=1.0)
    novelty: StateValue = Field(default=0.0, ge=0.0, le=1.0)
    consciousness: ConsciousnessLevel = Field(default=ConsciousnessLevel.DEEP_SLEEP)
```

### AgentRuntime (genome/base.py:129-157):
```python
@dataclass
class AgentRuntime:
    agent_id: str
    genome: ExtendedGenome
    current_state: EmotionState  # REAL emotion parameters here
    conversation_memory: ConversationContext
    last_reflection: Timestamp
    last_spoke: Timestamp
    wake_time: Optional[Timestamp]
```

### ProductionMonitor (emotion_engine/monitoring.py:62-79):
```python
class ProductionMonitor:
    agent_metrics: Dict[str, AgentMetrics]  # Real tracking per agent
    
class AgentMetrics:
    agent_id: str
    state_history: deque  # Contains (timestamp, EmotionState) tuples
    response_history: deque
    consciousness_transitions: deque
```

## 🎯 OPTIMAL CONFIGURATION DISCOVERED:
- **Overall Fitness**: 87.66% (hyperparameter optimization)
- **curiosity_base**: 0.5574 (moderate baseline curiosity)
- **confidence_base**: 0.3433 (lower confidence allows growth)  
- **social_energy_base**: 0.7805 (high social engagement)
- **social_decay_rate**: 0.0729 (MOST CRITICAL parameter)

## 🔧 DISCORD ENHANCEMENTS ADDED:

### Live Features:
- **live_parameter_hud.py**: Real-time emotion parameter display in Discord
- **emotion_plotting.py**: Real-time plotting system with charts + CSV logging
- **hidden_reflection_display.py**: Cost tracking for LLM reflection passes
- **dev_thread_manager.py**: Project thread management with model pinging
- **agent_status_command.py**: /status command for agent monitoring
- **state_persistence.py**: Save/restore agent states across restarts

### MCP Integration:
- **mcp_integration.py**: Model Context Protocol support for repository access
- Compatible with Claude Code .mcp.json configurations

## 🧪 TESTING & VALIDATION:
- **test_emotion_system.py**: Emotion engine validation tests
- **test_llm_evolution.py**: LLM self-evolution testing
- **test_production_integration.py**: Production system validation
- **validate_optimal_deployment.py**: Full system deployment validation

## 📈 HYPERPARAMETER OPTIMIZATION:
- **hyperparameter_sweep.py**: Main optimization system
- **quick_hyperparam_sweep.py**: Fast parameter testing
- **optimal_config_validation.py**: Config validation with results
- **fast_optimization_results.json**: Optimization results data

## 🔄 INTEGRATION SYSTEMS:
- **emotional_virtual_users.py**: Emotion-enhanced virtual users
- **emotion_integration.py**: Integration helpers
- **emotion_scheduler.py**: Emotion system scheduling
- **enhanced_multimodel.py**: Enhanced multi-model processing

## 📁 DATA & CONFIG FILES:
- **virtual_users.json**: Virtual user configurations
- **emotional_virtual_users.json**: Emotion-enhanced user configs  
- **openrouter_models.json**: Available model definitions
- **emotion_test_report.json**: Test results
- **fast_hyperparameter_results.png**: Optimization visualization

## 🐳 DEPLOYMENT:
- **Dockerfile**: Container configuration
- **docker-compose.yaml**: Multi-container setup
- **requirements.txt**: Python dependencies

## 📚 DOCUMENTATION:
- **ARCHITECTURE.md**: Comprehensive architecture overview with 10 development threads
- **EMOTION_ENGINE_COMPLETE.md**: Complete emotion engine documentation  
- **DISCORD_ENHANCEMENTS.md**: Discord integration enhancements
- **HYPERPARAMETER_SUMMARY.md**: Optimization results summary
- **README.md**: Project overview
- **LICENSE.md**: License information

## 🎮 USER INTERACTION FLOW:
1. User mentions models in Discord (@opus @sonnet etc.)
2. `llmcord_multimodel.py` detects mentions
3. `virtual_users.py` manages AI personas via webhooks
4. Each model has `AgentRuntime` with real `EmotionState`
5. `emotion_engine/reflection.py` decides if agent should respond
6. `emotion_engine/injection.py` adds emotion context to prompts
7. `ProductionMonitor` tracks all interactions and states
8. `live_parameter_hud.py` displays real-time emotion parameters

## 🎨 REAL-TIME PLOTTING SYSTEM:

### **emotion_plotting.py** - Complete Visualization System:
- **Charts Generated**: Every 15 seconds → `charts/emotion_params_{run_id}_{counter}.png`
- **CSV Logging**: Every 3 seconds → `charts/{run_id}.{model_name}.log` 
- **Data Sampling**: Every 1 second from real ProductionMonitor
- **Multi-Agent Stacked Charts**: All 8 emotion parameters per agent
- **CSV Format**: timestamp,curiosity,confidence,social_energy,restlessness,harmony,expertise,novelty,consciousness
- **Dark Theme**: Color-coded parameter lines with proper time-series formatting
- **Run-Specific Files**: New PNG + CSV files for each server run

### **Integration**: 
- Auto-starts with llmcord_multimodel.py
- Uses REAL emotion data from ProductionMonitor.agent_metrics
- Handles dynamic agent registration
- Memory-efficient with 15-minute rolling windows

## ⚠️ CRITICAL INTEGRATION POINTS:
- **ProductionMonitor.agent_metrics** contains REAL emotion data
- **EmotionState objects** have actual curiosity, confidence, social_energy values  
- **AgentRuntime.current_state** is the live emotion state per agent
- **Each Discord model should have separate AgentRuntime tracking**
- **Never simulate parameters - always use real monitoring system**
- **emotion_plotting.py samples real data every 1s, plots every 15s, logs every 3s**