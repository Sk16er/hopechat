# hopechat
<img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/5c09316e-32d4-498e-b56e-68efa57c5d90" />


> The best *Self-Evolving Intelligence* that $100 can buy.

This repo is a full-stack implementation of a **Hierarchical Optimizing Processing Ensemble (HOPE)**—an advanced, self-modifying, recurrent AI architecture—in a single, clean, minimal, hackable, dependency-lite codebase. hopechat is designed to run on a single 8XH100 node via scripts like [speedrun.sh](speedrun.sh), that run the entire pipeline start to end. This includes tokenization, pretraining (using nested adaptation), finetuning, evaluation, inference, and web serving over a simple UI so that you can talk to your own evolving AI.

**HOPE** differs from standard Transformers by introducing:
- **Nested Learning**: A dual-loop system where inner "fast weights" (Memory) adapt instantly to context, while outer "slow weights" (Policy) learn general patterns.
- **Continuum Memory System (CMS)**: An infinite-context mechanism that compresses history into a hierarchical state, allowing the model to recall information from thousands of tokens ago without quadratic cost.
- **Dynamic Hierarchy**: The model can dynamically grow its own depth (add layers) during training when it detects high surprise, optimizing its capacity on the fly.

## Talk to it

coming!!

## Quick start

The fastest way to feel the magic is to run the speedrun script [speedrun.sh](speedrun.sh), which trains and inferences the $100 tier of hopechat. On an 8XH100 node at $24/hr, this gives a total run time of about 4 hours. Boot up a new 8XH100 GPU box from your favorite provider, and kick off the training script:

```bash
bash speedrun.sh
```

Alternatively, since the script runs for 4 hours, launch it inside a screen session:

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

Once it's done, you can talk to your AI via the ChatGPT-like web UI. Make sure your local uv virtual environment is active, and serve it:

```bash
python -m scripts.chat_web
```

Then visit the URL shown (e.g., `http://<node-ip>:8000/`). Talk to your HOPE model—ask it to solve riddles, remember facts from the beginning of the chat, or explain its own architecture.

You can also view the `report.md` file generated in the project directory for a "report card" of the run, including CORE scores, ARC benchmarks, and training stats.

```
will be updated whn i train the model
```

## Bigger models

Unsurprisingly, $100 is not enough to train a logic-defying Superintelligence. However, HOPE scales efficiently.
- **d20 (500M params)**: The "Speedrun" tier. Fast, capable, efficient. ~$100.
- **d32 (1.9B params)**: The standard baseline. Outperforms GPT-2 significantly. ~$800.
- **dMax**: Since HOPE can dynamically grow, you can start small and let the model expand as needed, potentially reaching d40+ based on data complexity.

To train a larger model, simply edit `speedrun.sh` or call the training scripts directly with a higher `--depth`.

```bash
# Example: Train a d32 model
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=32 --device_batch_size=16
```

## Architecture: Under the Hood

Hopechat is not just another Llama-clone. It implements the **HOPE (Hierarchical Optimizing Processing Ensemble)** architecture:

1.  **Recurrent Memory (Fast Weights)**: Instead of a static KV cache, HOPE maintains a compressed, evolving state $S_t$.
    $$S_t = (1 - G_t) \cdot S_{t-1} + G_t \cdot (K^T V)$$
    This allows *O(1)* inference time regardless of context length.

2.  **Surprise-Gated Updates**: The model only updates its memory when it encounters "surprise" (high prediction error), ignoring redundant information to save capacity.

3.  **Titans Layer (Long-Term Memory)**: A specialized neural memory module that learns to store and retrieve information over very long horizons (thousands of tokens), solving the "Goldfish Memory" problem of standard RNNs/Transformers.

## Running on CPU / MPS

hopechat can be run on CPU or on MPS (Apple Silicon). It automatically detects your device. While training will be slow, inference is snappy thanks to the recurrent architecture.

## File structure

```
.
├── LICENSE
├── README.md
├── dev                     # Development tools and scripts
├── hopechat.png            # Logo
├── hopechat                # Main source code
│   ├── __init__.py
│   ├── ensemble.py         # The HOPE Architecture (Main Model)
│   ├── adamw.py            # Optimizers
│   ├── muon.py             # Muon Optimizer
│   ├── engine.py           # Inference Engine (Stateful)
│   ├── execution.py        # Tool use / Python execution environment
│   ├── tokenizer.py        # RustBPE Tokenizer
│   ├── configurator.py     # Config loader
│   ├── common.py           # Utilities
│   └── ui.html             # Web Interface
├── scripts                 # Training and Evaluation scripts
│   ├── base_train.py       # Pretraining (HOPE Ensembling)
│   ├── mid_train.py        # Mid-training (Instruction alignment)
│   ├── chat_sft.py         # Supervised Finetuning
│   ├── chat_rlvr.py        # Reinforcement Learning (Verifiable Rewards)
│   ├── chat_web.py         # Web Server
│   └── ...
├── speedrun.sh             # The $100 training script
├── run1000.sh              # The $800 training script
├── rustbpe                 # High-performance Tokenizer (Rust)
├── tasks                   # Evaluation tasks (ARC, GSM8K, etc.)
└── pyproject.toml
```

## Contributing

hopechat is an experimental research repo. We welcome PRs that optimize the Nested Learning kernels, add new memory compression techniques, or improve the dynamic growth capabilities.

## Acknowledgements

- **Inspiration**: This repo started as a fork of [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy. We have evolved it into the HOPE architecture while keeping the "speedrun" philosophy alive.

## License

MIT

