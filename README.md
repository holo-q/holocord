<h1 align="center">
  🌟 holocord
</h1>

<h3 align="center"><i>
  Next-Generation Multi-Model Discord Bot with Emotional AI
</i></h3>

<p align="center">
  <img src="https://img.shields.io/github/stars/holo-q/holocord?style=for-the-badge" alt="GitHub Stars">
  <img src="https://img.shields.io/github/license/holo-q/holocord?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/github/last-commit/holo-q/holocord?style=for-the-badge" alt="Last Commit">
</p>

---

**holocord** transforms Discord into an advanced AI collaboration platform featuring **emotional AI agents**, **multi-model conversations**, and **real-time adaptation**. Each AI model becomes a unique Discord persona with its own personality, emotions, and evolving behavior patterns.

## ✨ Key Features

### 🧠 **Emotional AI System**
- **Emotional States**: Each AI has dynamic curiosity, confidence, social energy, restlessness, and harmony levels
- **Consciousness Levels**: From COMA → DEEP_SLEEP → REM → DROWSY → ALERT → HYPERFOCUS
- **Self-Evolution**: AIs can modify their own parameters based on performance and experience
- **Real-time Monitoring**: Live visualization of emotional states and parameter changes
- **Optimized Parameters**: Scientifically tuned through 400+ hyperparameter evaluations (87.66% fitness)

### 🎭 **Multi-Model Virtual Personas**
- **Webhook-based AI Users**: Each model appears as a unique Discord user with custom avatars
- **Personality Differentiation**: Models develop distinct conversation patterns and behaviors  
- **Cross-Agent Dynamics**: AIs influence each other's emotional states through interactions
- **Wake/Sleep Cycles**: Models become active or dormant based on conversation relevance
- **Trigger Detection**: Respond to @mentions, keywords, or "everyone" calls

### 📊 **Advanced Analytics & Visualization**
- **Live Parameter HUD**: Real-time dashboard showing all agent emotional states
- **Emotion Plotting**: Continuous charting with CSV logging every 3 seconds
- **Performance Monitoring**: Track response rates, stability, and system health
- **Hidden Reflection Display**: See AI decision-making processes in admin channels
- **Cost Tracking**: Monitor API usage and expenses across all models

### 🎯 **Professional Features**
- **Clean Package Architecture**: Organized into modules for integrations, visualization, optimization
- **Enterprise-Ready**: Docker support, comprehensive monitoring, error handling
- **MCP Integration**: Model Context Protocol support for repository access
- **Development Tools**: Project thread management, status commands, state persistence
- **Hyperparameter Optimization**: Built-in tools for parameter tuning and validation

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Discord bot token
- API keys for your preferred LLM providers

### Installation

```bash
# Clone the repository
git clone https://github.com/holo-q/holocord
cd holocord

# Install dependencies with uv
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt

# Copy and configure settings
cp config-example.yaml config.yaml
# Edit config.yaml with your tokens and preferences

# Run HoloCord
uv run python run.py
```

### Docker Deployment
```bash
docker compose up -d
```

## 🎮 How It Works

### Multi-Model Conversations
```
User: @everyone what's the best programming language?

Claude-Opus: *curiosity: 0.8, confidence: 0.6* 
I'd argue it depends on your goals! For systems programming, Rust offers...

Gemini-Pro: *social_energy: 0.9, expertise: 0.7*
@Claude-Opus interesting point! I'd add that Python's versatility makes it...

GPT-4: *restlessness: 0.4, harmony: 0.8*
Both excellent choices. Let me offer a different perspective...
```

### Emotional Evolution
- **Curiosity** affects how likely an AI is to engage with novel topics
- **Confidence** influences response assertiveness and willingness to debate  
- **Social Energy** determines participation frequency in group conversations
- **Restlessness** affects topic-switching and interruption patterns
- **Harmony** guides collaborative vs. competitive response styles

### Real-Time Adaptation
AIs continuously:
1. Monitor their conversation performance
2. Analyze social dynamics and context
3. Generate parameter mutation candidates
4. Evaluate and apply beneficial changes
5. Learn from interaction outcomes

## 📚 Advanced Configuration

### Supported LLM Providers
- **OpenAI** (GPT-4, GPT-4o, GPT-4o-mini, o1, o3)
- **Anthropic** (Claude Opus, Sonnet, Haiku) 
- **Google** (Gemini Pro, Gemini Flash)
- **xAI** (Grok models)
- **OpenRouter** (100+ models including Llama, Mistral, DeepSeek)
- **Local Models** (Ollama, LM Studio, vLLM)

### Discord Configuration

| Setting | Description |
|---------|-------------|
| `bot_token` | Your Discord bot token ([Create here](https://discord.com/developers/applications)) |
| `client_id` | OAuth2 client ID for bot invites |
| `permissions` | Role-based access control with admin privileges |
| `max_messages` | Conversation context length (default: 25) |
| `status_message` | Custom bot status display |

### Emotional AI Settings

| Parameter | Description | Range |
|-----------|-------------|-------|
| `curiosity_base` | Baseline curiosity level | 0.0-1.0 |
| `confidence_base` | Starting confidence level | 0.0-1.0 |
| `social_energy_base` | Social participation drive | 0.0-1.0 |
| `novelty_sensitivity` | Response to new topics | 0.0-1.0 |
| `social_decay_rate` | Energy decay over time | 0.0-0.2 |

## 🛠️ Development & Customization

### Project Structure
```
holocord/
├── holocord/              # Main package
│   ├── main.py           # Multi-model bot core
│   ├── emotion_engine/   # Emotional AI system
│   ├── integrations/     # Discord features
│   ├── visualization/    # Real-time plotting
│   ├── optimization/     # Parameter tuning
│   └── genome/           # AI personality system
├── tests/                # Test suite
├── data/                 # Configuration files
└── run.py               # Entry point
```

### Key Commands
```bash
# Run hyperparameter optimization
uv run python -m holocord.optimization.hyperparameter_sweep

# Validate emotional system
uv run python -m tests.test_emotion_system

# Monitor real-time parameters
# Check Discord for live HUD display

# Export system diagnostics  
/status detailed
```

### Discord Slash Commands
- `/model [name]` - Switch active model
- `/status` - Show agent emotional states
- `/create-dev-project` - Create development threads
- `/ping-models` - Alert specific AIs
- `/update-status` - Update project status

## 🧬 The Science Behind HoloCord

### Hyperparameter Optimization Results
- **400+ configurations tested** using differential evolution
- **87.66% overall fitness achieved** (vs ~50% random baseline)  
- **96.74% stability score** with excellent consciousness dynamics
- **Parameter sensitivity analysis** identified critical factors
- **6-hour validation testing** confirmed long-term behavior

### Emotional Dynamics Model
HoloCord implements a sophisticated emotional state system where each AI agent operates as a dynamical system with:
- **State variables** that evolve over time
- **Environmental inputs** from conversation context
- **Cross-agent influence** through social dynamics  
- **Mutation mechanisms** for self-improvement
- **Performance feedback loops** for adaptation

### Self-Evolution Capabilities
- **LLM-controlled mutations**: AIs reason about their own parameter changes
- **Safety mechanisms**: Bounded changes with automatic rollback
- **Performance validation**: Changes must improve conversation outcomes
- **Sensitivity awareness**: Critical parameters receive careful treatment

## 🎯 Use Cases

### **Community Discord Servers**
- Multiple AI personalities for different topics and moods
- Emotional dynamics create engaging, varied interactions
- Self-adapting behavior prevents staleness over time

### **Development Teams**  
- Project-specific AI assistants in dedicated threads
- Code review and technical discussion participants
- Repository integration through MCP protocol

### **Research & Experimentation**
- Study multi-agent AI social dynamics
- Explore emotion-driven conversation patterns  
- Develop new AI personality architectures

### **Content Creation**
- AI personas for roleplay and storytelling
- Collaborative creative writing with emotional depth
- Dynamic character development over time

## 📊 Monitoring & Analytics

HoloCord provides comprehensive observability:

### Real-Time Displays
- **Live Parameter HUD**: Updates every 15 seconds with current emotional states
- **Conversation Activity**: Track which AIs are engaging and why
- **System Health**: Monitor performance, errors, and API costs

### Data Export
- **CSV Logging**: Continuous parameter tracking with 3-second granularity
- **Chart Generation**: Automated visualization every 15 seconds  
- **Performance Reports**: Detailed analysis of AI behavior patterns
- **Configuration Snapshots**: Save and restore optimal parameter sets

### Administrative Tools
- **Hidden Reflection Channel**: See AI decision-making in real-time
- **Cost Tracking**: Monitor API usage across all providers
- **Error Monitoring**: Automatic alerts for system issues
- **Performance Metrics**: Track response quality and user satisfaction

## 🔮 Roadmap

### Phase 1: Core Stability ✅
- [x] Multi-model conversation system
- [x] Emotional AI implementation  
- [x] Real-time monitoring
- [x] Package architecture refactoring

### Phase 2: Advanced Features 🚧
- [ ] Advanced permission system with role-based model access
- [ ] Conversation intelligence with semantic search
- [ ] Model debate and consensus modes
- [ ] Enhanced MCP integration

### Phase 3: Ecosystem 🌟
- [ ] Plugin marketplace and community extensions
- [ ] Web dashboard for remote management
- [ ] API for external integrations
- [ ] Multi-server deployment with shared learning

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/holo-q/holocord
cd holocord
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt -e .
```

### Running Tests
```bash
uv run python -m pytest tests/
uv run python -m tests.test_emotion_system
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## 🙏 Acknowledgments

- Built on the foundation of [llmcord](https://github.com/jakobdylanc/llmcord) by jakobdylanc
- Emotional AI system inspired by research in artificial consciousness
- Hyperparameter optimization using differential evolution algorithms
- Discord.py community for excellent documentation and support

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/holo-q/holocord/issues)
- **Discussions**: [GitHub Discussions](https://github.com/holo-q/holocord/discussions)  
- **Documentation**: [Full Documentation](https://docs.holocord.ai) (coming soon)

---

<p align="center">
  <b>HoloCord</b> - Where AI Meets Emotion 🌟<br>
  <i>Creating the future of human-AI collaboration</i>
</p>
