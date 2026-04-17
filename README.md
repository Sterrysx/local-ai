# Local-AI (locai)

An on-demand, high-performance local LLM orchestration system with multi-GPU support.

## Features
- **Dynamic Orchestration**: Auto-loads/unloads models on-demand based on GPU VRAM availability.
- **OpenAI-Compatible API**: Seamlessly integrates with existing tools via a unified gateway.
- **Interactive Shell**: Built-in CLI with tab-autocomplete, ghost-text suggestions, and real-time performance metrics (Speed/TTFT/TTLT).
- **Advanced VRAM Profiling**: Real-time per-GPU visualization (Weights, Context Window, System Overhead, Free).
- **Secure by Design**: API Key authentication (Bearer token).
- **Resilient**: Automatic process cleanup and port-collision handling.

## Components
- **`llama-gateway.py`**: The orchestration server managing process lifecycles and HTTP requests.
- **`llmctl`**: The interactive CLI (`locai`) for model management, monitoring, and chatting.
- **`config.json`**: Centralized configuration for models, hardware paths, and settings.

## Quick Start
1. Ensure `LOCAI_API_KEY` is set in your environment:
   `export LOCAI_API_KEY=your-secret-key`
2. Start the gateway service:
   `systemctl --user restart llama-gateway.service`
3. Launch the shell:
   `locai`

## Commands
- `/use <model>`: Switch active model.
- `/models`: List available models with quantization and context specs.
- `/vram`: Display detailed per-GPU memory allocation.
- `/status`: Show system telemetry.
- `/help`: List all available commands.
