"""
Reinforcement Learning from Verifiable Rewards (RLVR)
Trains the model on math/code puzzles using the code sandbox.
"""

import os
import time
import torch
import torch.nn.functional as F
from contextlib import nullcontext

from hopechat.ensemble import HOPEEnsemble, HOPEConfig
from hopechat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type
from hopechat.tokenizer import get_tokenizer
from hopechat.checkpoint_manager import load_checkpoint, save_checkpoint
from hopechat.engine import Engine, use_calculator

def run_rlvr(model, tokenizer, device, num_iterations=100, batch_size=4):
    print0("Starting RLVR Training...")
    
    # Simple dataset of math problems (in reality, load from GSM8K or similar)
    # Format: (Question, Answer)
    # For RLVR, we need verifiable rewards. Math is perfect.
    problems = [
        ("What is 25 * 4?", "100"),
        ("Calculate 12 + 88", "100"),
        ("What is 15 * 3 + 5?", "50"),
        ("Solve 100 / 5", "20"),
        ("Compute 2 ** 10", "1024"),
        ("What is 50 * 2?", "100"),
        ("Calculate 10 * 10", "100"),
        ("What is 1000 / 10?", "100"),
    ] * 10 # Repeat for demo
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    engine = Engine(model, tokenizer)
    
    model.train()
    
    for step in range(num_iterations):
        # Sample a batch
        batch = problems[:batch_size] # Just take first few for demo
        
        total_reward = 0
        optimizer.zero_grad()
        
        for question, answer in batch:
            # 1. Generate solution with code execution
            # We prompt the model to use python
            prompt = f"Question: {question}\nUse python to solve this.\n"
            tokens = tokenizer.encode(prompt, prepend=True)
            
            # Generate
            # We need log_probs for RL, so we can't use the Engine directly if it doesn't return them.
            # But Engine is for inference. For training, we usually do:
            # Generate -> Get Trajectory -> Compute Reward -> Re-compute log_probs -> Update
            # For "speedrun", we will do a simplified REINFORCE or GRPO.
            
            # Let's use the model directly to generate and keep graph? No, too memory intensive.
            # Standard PPO: Generate (no grad), Score, Train (with grad).
            
            # A. Generate
            with torch.no_grad():
                # We use engine to get the trace including tool use
                # But Engine returns tokens, not log probs.
                # We'll just get the text.
                # For simplicity in this script, we assume the model outputs the answer directly or via tool.
                # Let's just use the Engine to get the result.
                
                # Hack: We need the tokens to compute loss later.
                # Engine returns list of tokens.
                gen_tokens_list, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=64)
                gen_tokens = gen_tokens_list[0]
                
            # B. Verify
            gen_text = tokenizer.decode(gen_tokens)
            # Check if answer is in text
            reward = 1.0 if answer in gen_text else -1.0
            total_reward += reward
            
            # C. Update (Policy Gradient)
            # Re-forward the generated tokens to get gradients
            # We only train on the completion
            prompt_len = len(tokens)
            completion_tokens = gen_tokens[prompt_len:]
            if not completion_tokens:
                continue
                
            # Prepare inputs
            # Input: prompt + completion[:-1]
            # Target: completion
            full_ids = torch.tensor([gen_tokens], dtype=torch.long, device=device)
            
            logits = model(full_ids) # (1, T, V)
            
            # Shift logits and labels
            # logits: [0, ..., T-2] -> predicts [1, ..., T-1]
            # labels: [1, ..., T-1]
            # We only care about the completion part
            
            # Indices in full_ids:
            # Prompt: 0 ... P-1
            # Completion: P ... T-1
            
            # Logits at index i predicts token at i+1
            # We want logits from P-1 to T-2 to predict P to T-1
            
            completion_logits = logits[0, prompt_len-1 : -1]
            completion_targets = full_ids[0, prompt_len:]
            
            loss = F.cross_entropy(completion_logits, completion_targets, reduction='none')
            
            # Policy Gradient: Loss = -Reward * log_prob
            # Here cross_entropy is -log_prob. So Loss = Reward * CrossEntropy?
            # No. Minimize J = - E[R]. Gradient is - E[R * grad(log_prob)].
            # Loss function in PyTorch minimizes.
            # We want to minimize - (Reward * log_prob).
            # CrossEntropy is -log_prob.
            # So minimize Reward * CrossEntropy?
            # If Reward is +1, minimize -log_prob (maximize log_prob). Correct.
            # If Reward is -1, minimize +log_prob (minimize log_prob). Correct.
            
            # However, usually we use Advantage = Reward - Baseline.
            # For simple REINFORCE with baseline 0:
            pg_loss = (loss * reward).mean()
            
            pg_loss.backward()
            
        optimizer.step()
        
        if step % 10 == 0:
            print0(f"Step {step}: Avg Reward {total_reward / batch_size:.2f}")

if __name__ == "__main__":
    # Setup
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    
    # Load Model (SFT or Base)
    # Assuming we continue from SFT
    base_dir = get_base_dir()
    # For demo, just load random or whatever is available
    # In speedrun, this runs after SFT
    
    # Initialize model
    tokenizer = get_tokenizer()
    # Load config from latest checkpoint or default
    # For speedrun simplicity, we assume d20 config
    model_config = HOPEConfig(n_layer=20, n_head=6, n_embd=1280) # Approx d20
    model = HOPEEnsemble(model_config)
    model.to(device)
    
    # Try to load SFT checkpoint
    # ... (omitted for brevity, assuming standard load)
    
    run_rlvr(model, tokenizer, device)
    
    compute_cleanup()
