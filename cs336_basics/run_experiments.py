import subprocess
import os
import argparse
import sys
import torch

def run_training_loop(args_list, run_name):
    """Helper function to run the training loop with given arguments."""
    print(f"\n{'='*80}\nStarting experiment: {run_name}\n{'='*80}\n")
    command = ["uv", "run", "-m", "cs336_basics.training_loop", "--run_name", run_name] + args_list
    print(f"Executing command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Stream output to terminal in real-time
    for line in process.stdout:
        print(line, end='')

    process.wait()
    if process.returncode != 0:
        print(f"\n{'!'*80}\nExperiment '{run_name}' failed with exit code {process.returncode}\n{'!'*80}\n")
        # Optionally, you might want to stop the entire script here or log more aggressively
    else:
        print(f"\n{'='*80}\nExperiment '{run_name}' completed successfully.\n{'='*80}\n")
    return process.returncode

def add_common_flags(experiment_args, args):
    """Add common flags to experiment arguments."""
    if args.use_wandb:
        experiment_args.append("--use_wandb")
    if args.use_compile:
        experiment_args.append("--use_compile")
    if args.use_memmap:
        experiment_args.append("--use_memmap")
    if args.reuse_pretokens:
        experiment_args.append("--reuse_pretokens")
    return experiment_args

def base_model_tinystories(args):
    """Runs the base model on TinyStories with optimal LR."""
    print("--- Running Base Model on TinyStories ---")
    # Using default hyperparameters from training_loop.py, just explicitly setting LR and run_name
    optimal_lr = args.learning_rate # Use the optimal LR found in previous tuning

    experiment_args = [
        f"--learning_rate={optimal_lr}",
        f"--max_steps={args.max_steps}",
        f"--batch_size={args.batch_size}",
        f"--context_length={args.context_length}",
        f"--vocab_size={args.vocab_size}",
        f"--d_model={args.d_model}",
        f"--num_heads={args.num_heads}",
        f"--num_layers={args.num_layers}",
        f"--d_ff={args.d_ff}",
        f"--rope_theta={args.rope_theta}",
        f"--device={args.device}",
        f"--wandb_project={args.wandb_project}",
        f"--eval_freq={args.eval_freq}",
        f"--checkpoint_freq={args.checkpoint_freq}",
        f"--warmup_steps={args.warmup_steps}",
    ]
    if args.use_wandb:
        experiment_args.append("--use_wandb")
    if args.use_compile:
        experiment_args.append("--use_compile")
    if args.use_memmap:
        experiment_args.append("--use_memmap")
    if args.reuse_pretokens:
        experiment_args.append("--reuse_pretokens")
    
    run_training_loop(experiment_args, "TinyStories_Base_Model")

def learning_rate_sweep_tinystories(args):
    """Performs a learning rate sweep for TinyStories."""
    print("--- Running Learning Rate Sweep on TinyStories ---")
    # Define a range of learning rates to test
    # You will need to adjust these based on your findings for what works well and what diverges
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] 

    for lr in learning_rates:
        run_name = f"TinyStories_LR_Sweep_LR_{lr:.0e}"
        experiment_args = [
            f"--learning_rate={lr}",
            f"--max_steps={args.max_steps}",
            f"--batch_size={args.batch_size}",
            f"--context_length={args.context_length}",
            f"--vocab_size={args.vocab_size}",
            f"--d_model={args.d_model}",
            f"--num_heads={args.num_heads}",
            f"--num_layers={args.num_layers}",
            f"--d_ff={args.d_ff}",
            f"--rope_theta={args.rope_theta}",
            f"--device={args.device}",
            f"--wandb_project={args.wandb_project}",
            f"--eval_freq={args.eval_freq}",
            f"--checkpoint_freq={args.checkpoint_freq}",
            f"--warmup_steps={args.warmup_steps}",
        ]
        if args.use_wandb:
            experiment_args.append("--use_wandb")
        if args.use_compile:
            experiment_args.append("--use_compile")
        if args.use_memmap:
            experiment_args.append("--use_memmap")
        
        run_training_loop(experiment_args, run_name)

def batch_size_sweep_tinystories(args):
    """Performs a batch size sweep for TinyStories."""
    print("--- Running Batch Size Sweep on TinyStories ---")
    # Define a range of batch sizes to test
    # You will need to determine the GPU memory limit for your system
    batch_sizes = [1, 16, 32, 64, 128] # Example batch sizes

    # Use the optimal LR found from previous experiments or a reasonable default
    optimal_lr_for_batch_size = args.learning_rate 

    for bs in batch_sizes:
        run_name = f"TinyStories_BS_Sweep_BS_{bs}"
        experiment_args = [
            f"--learning_rate={optimal_lr_for_batch_size}", # Re-tune if necessary for larger batch sizes
            f"--max_steps={args.max_steps}",
            f"--batch_size={bs}",
            f"--context_length={args.context_length}",
            f"--vocab_size={args.vocab_size}",
            f"--d_model={args.d_model}",
            f"--num_heads={args.num_heads}",
            f"--num_layers={args.num_layers}",
            f"--d_ff={args.d_ff}",
            f"--rope_theta={args.rope_theta}",
            f"--device={args.device}",
            f"--wandb_project={args.wandb_project}",
            f"--eval_freq={args.eval_freq}",
            f"--checkpoint_freq={args.checkpoint_freq}",
            f"--warmup_steps={args.warmup_steps}",
        ]
        if args.use_wandb:
            experiment_args.append("--use_wandb")
        if args.use_compile:
            experiment_args.append("--use_compile")
        if args.use_memmap:
            experiment_args.append("--use_memmap")
        
        run_training_loop(experiment_args, run_name)


# --- Ablation Experiments (Require manual code modification in transformer.py) ---

def rmsnorm_ablation_tinystories(args):
    """Runs training with RMSNorm removed."""
    print("\n!!! MANUAL ACTION REQUIRED !!!")
    print("Please go to 'assignment1-basics/cs336_basics/transformer.py' and COMMENT OUT the RMSNorm layers in 'TransformerBlock'.")
    print("Specifically, comment out or remove lines like: `self.ln1 = RMSNorm(...)` and `self.ln2 = RMSNorm(...)`")
    print("And modify `forward` method from `h = self.ln1(x)` to `h = x` (and similarly for `ln2`).")
    input("Press Enter to continue after modifying transformer.py...")

    optimal_lr = args.learning_rate # Start with optimal LR, be prepared to lower it if diverges

    # Run with previous optimal LR
    run_name_optimal_lr = "TinyStories_NoRMSNorm_Optimal_LR"
    experiment_args_optimal_lr = [
        f"--learning_rate={optimal_lr}",
        f"--max_steps={args.max_steps}",
        f"--batch_size={args.batch_size}",
        f"--context_length={args.context_length}",
        f"--vocab_size={args.vocab_size}",
        f"--d_model={args.d_model}",
        f"--num_heads={args.num_heads}",
        f"--num_layers={args.num_layers}",
        f"--d_ff={args.d_ff}",
        f"--rope_theta={args.rope_theta}",
        f"--device={args.device}",
        f"--wandb_project={args.wandb_project}",
        f"--eval_freq={args.eval_freq}",
        f"--checkpoint_freq={args.checkpoint_freq}",
        f"--warmup_steps={args.warmup_steps}",
    ]
    if args.use_wandb: experiment_args_optimal_lr.append("--use_wandb")
    if args.use_compile: experiment_args_optimal_lr.append("--use_compile")
    if args.use_memmap: experiment_args_optimal_lr.append("--use_memmap")

    run_training_loop(experiment_args_optimal_lr, run_name_optimal_lr)

    print("\n!!! MANUAL ACTION REQUIRED !!!")
    print("If the previous run diverged, now try a LOWER learning rate.")
    input("Press Enter to continue to run with lower LR (or Ctrl+C to skip)...")
    
    lower_lr = optimal_lr / 10 # Example: try 10x lower
    run_name_lower_lr = f"TinyStories_NoRMSNorm_Lower_LR_{lower_lr:.0e}"
    experiment_args_lower_lr = [
        f"--learning_rate={lower_lr}",
        f"--max_steps={args.max_steps}",
        f"--batch_size={args.batch_size}",
        f"--context_length={args.context_length}",
        f"--vocab_size={args.vocab_size}",
        f"--d_model={args.d_model}",
        f"--num_heads={args.num_heads}",
        f"--num_layers={args.num_layers}",
        f"--d_ff={args.d_ff}",
        f"--rope_theta={args.rope_theta}",
        f"--device={args.device}",
        f"--wandb_project={args.wandb_project}",
        f"--eval_freq={args.eval_freq}",
        f"--checkpoint_freq={args.checkpoint_freq}",
        f"--warmup_steps={args.warmup_steps}",
    ]
    if args.use_wandb: experiment_args_lower_lr.append("--use_wandb")
    if args.use_compile: experiment_args_lower_lr.append("--use_compile")
    if args.use_memmap: experiment_args_lower_lr.append("--use_memmap")
    
    run_training_loop(experiment_args_lower_lr, run_name_lower_lr)

    print("\n!!! IMPORTANT: MANUAL ACTION REQUIRED !!!")
    print("Please REVERT the changes to 'assignment1-basics/cs336_basics/transformer.py' (re-enable RMSNorm) before proceeding to other experiments.")
    input("Press Enter after reverting changes...")


def pre_norm_ablation_tinystories(args):
    """Runs training with post-norm Transformer blocks."""
    print("\n!!! MANUAL ACTION REQUIRED !!!")
    print("Please go to 'assignment1-basics/cs336_basics/transformer.py' and MODIFY 'TransformerBlock' to use POST-NORM.")
    print("Original (Pre-Norm):")
    print("    h = self.ln1(x)")
    print("    h = self.mha(h)")
    print("    x = x + h")
    print("    h = self.ln2(x)")
    print("    h = self.ffn(h)")
    print("    x = x + h")
    print("New (Post-Norm):")
    print("    attn_output = self.mha(x)")
    print("    x = self.ln1(x + attn_output)") # Norm after residual connection
    print("    ffn_output = self.ffn(x)")
    print("    x = self.ln2(x + ffn_output)") # Norm after residual connection
    input("Press Enter to continue after modifying transformer.py...")

    optimal_lr = args.learning_rate # Use the optimal LR found in previous tuning

    run_name = "TinyStories_PostNorm"
    experiment_args = [
        f"--learning_rate={optimal_lr}",
        f"--max_steps={args.max_steps}",
        f"--batch_size={args.batch_size}",
        f"--context_length={args.context_length}",
        f"--vocab_size={args.vocab_size}",
        f"--d_model={args.d_model}",
        f"--num_heads={args.num_heads}",
        f"--num_layers={args.num_layers}",
        f"--d_ff={args.d_ff}",
        f"--rope_theta={args.rope_theta}",
        f"--device={args.device}",
        f"--wandb_project={args.wandb_project}",
        f"--eval_freq={args.eval_freq}",
        f"--checkpoint_freq={args.checkpoint_freq}",
        f"--warmup_steps={args.warmup_steps}",
    ]
    if args.use_wandb: experiment_args.append("--use_wandb")
    if args.use_compile: experiment_args.append("--use_compile")
    if args.use_memmap: experiment_args.append("--use_memmap")

    run_training_loop(experiment_args, run_name)

    print("\n!!! IMPORTANT: MANUAL ACTION REQUIRED !!!")
    print("Please REVERT the changes to 'assignment1-basics/cs336_basics/transformer.py' (revert to PRE-NORM) before proceeding to other experiments.")
    input("Press Enter after reverting changes...")


def no_pos_emb_ablation_tinystories(args):
    """Runs training with position embeddings (RoPE) removed."""
    print("\n!!! MANUAL ACTION REQUIRED !!!")
    print("Please go to 'assignment1-basics/cs336_basics/transformer.py' and DISABLE RoPE in 'MultiHeadSelfAttention'.")
    print("Specifically, in the `__init__` of `MultiHeadSelfAttention`, ensure `apply_rope` is `False` or comment out its usage.")
    print("In the `forward` method, comment out or skip the `self.rope(q, positions)` and `self.rope(k, positions)` lines.")
    input("Press Enter to continue after modifying transformer.py...")

    optimal_lr = args.learning_rate # Use the optimal LR found in previous tuning

    run_name = "TinyStories_NoPE"
    experiment_args = [
        f"--learning_rate={optimal_lr}",
        f"--max_steps={args.max_steps}",
        f"--batch_size={args.batch_size}",
        f"--context_length={args.context_length}",
        f"--vocab_size={args.vocab_size}",
        f"--d_model={args.d_model}",
        f"--num_heads={args.num_heads}",
        f"--num_layers={args.num_layers}",
        f"--d_ff={args.d_ff}",
        f"--rope_theta={args.rope_theta}", # Still pass, but won't be used
        f"--device={args.device}",
        f"--wandb_project={args.wandb_project}",
        f"--eval_freq={args.eval_freq}",
        f"--checkpoint_freq={args.checkpoint_freq}",
        f"--warmup_steps={args.warmup_steps}",
    ]
    if args.use_wandb: experiment_args.append("--use_wandb")
    if args.use_compile: experiment_args.append("--use_compile")
    if args.use_memmap: experiment_args.append("--use_memmap")

    run_training_loop(experiment_args, run_name)

    print("\n!!! IMPORTANT: MANUAL ACTION REQUIRED !!!")
    print("Please REVERT the changes to 'assignment1-basics/cs336_basics/transformer.py' (re-enable RoPE) before proceeding to other experiments.")
    input("Press Enter after reverting changes...")


def swiglu_ablation_tinystories(args):
    """Runs training with SwiGLU replaced by FFNSiLU."""
    print("\n!!! MANUAL ACTION REQUIRED !!!")
    print("Please go to 'assignment1-basics/cs336_basics/transformer.py' and MODIFY 'SwiGLU' to implement 'FFNSiLU(x) = W2SiLU(W1x)'.")
    print("Remember to set `d_ff = 4 * d_model` (or adjust your existing d_ff to be 4x d_model) in the FFNSiLU implementation.")
    print("Your SwiGLU `forward` is: `self.W2(self.W1(x) * torch.sigmoid(self.W1(x)) * self.W3(x))` (or similar)")
    print("New FFNSiLU `forward` should be: `self.W2_silu(torch.nn.functional.silu(self.W1_silu(x)))`")
    input("Press Enter to continue after modifying transformer.py...")

    optimal_lr = args.learning_rate # Use the optimal LR found in previous tuning

    run_name = "TinyStories_FFNSiLU"
    experiment_args = [
        f"--learning_rate={optimal_lr}",
        f"--max_steps={args.max_steps}",
        f"--batch_size={args.batch_size}",
        f"--context_length={args.context_length}",
        f"--vocab_size={args.vocab_size}",
        f"--d_model={args.d_model}",
        f"--num_heads={args.num_heads}",
        f"--num_layers={args.num_layers}",
        f"--d_ff={4 * args.d_model}", # Explicitly pass the adjusted d_ff for SiLU
        f"--rope_theta={args.rope_theta}",
        f"--device={args.device}",
        f"--wandb_project={args.wandb_project}",
        f"--eval_freq={args.eval_freq}",
        f"--checkpoint_freq={args.checkpoint_freq}",
        f"--warmup_steps={args.warmup_steps}",
    ]
    if args.use_wandb: experiment_args.append("--use_wandb")
    if args.use_compile: experiment_args.append("--use_compile")
    if args.use_memmap: experiment_args.append("--use_memmap")

    run_training_loop(experiment_args, run_name)

    print("\n!!! IMPORTANT: MANUAL ACTION REQUIRED !!!")
    print("Please REVERT the changes to 'assignment1-basics/cs336_basics/transformer.py' (revert to SwiGLU and original d_ff) before proceeding to other experiments.")
    input("Press Enter after reverting changes...")


# --- Main Experiment and Leaderboard ---

def main_experiment_openwebtext(args):
    """Runs the base model on OpenWebText."""
    print("--- Running Base Model on OpenWebText ---")
    # You will likely need to re-tune learning rate and potentially other AdamW params for OWT
    # Start with optimal LR from TinyStories, but be prepared to adjust.
    optimal_lr = args.learning_rate # Placeholder, you will tune this

    # Ensure pretokenization paths are for OpenWebText
    train_path = "data/OpenWebText_small_train.txt" # Adjust to your actual OWT train file path
    valid_path = "data/OpenWebText_small_valid.txt" # Adjust to your actual OWT valid file path
    pretokens_train_path = "outputs/openwebtext_train.npy"
    pretokens_valid_path = "outputs/openwebtext_valid.npy"
    vocab_path = "outputs/openwebtext_bpe_10k/vocab.pkl" # Adjust if your BPE is different
    merges_path = "outputs/openwebtext_bpe_10k/merges.pkl" # Adjust if your BPE is different

    print("\n!!! NOTE: OpenWebText pretokenization will run if files do not exist.")
    print("          It might take a significant amount of time.")
    print(f"          Using: {train_path}, {valid_path}, {vocab_path}, {merges_path}")
    input("Press Enter to continue (ensure you have the OWT data files and BPE tokenizer trained)...")


    experiment_args = [
        f"--train_path={train_path}",
        f"--valid_path={valid_path}",
        f"--pretokens_train_path={pretokens_train_path}",
        f"--pretokens_valid_path={pretokens_valid_path}",
        f"--vocab_path={vocab_path}",
        f"--merges_path={merges_path}",
        f"--dataset_name=openwebtext", # This is for logging purposes mostly
        f"--learning_rate={optimal_lr}",
        f"--max_steps={args.max_steps}", # Keep same as TinyStories for comparison as per instructions
        f"--batch_size={args.batch_size}",
        f"--context_length={args.context_length}",
        f"--vocab_size={args.vocab_size}",
        f"--d_model={args.d_model}",
        f"--num_heads={args.num_heads}",
        f"--num_layers={args.num_layers}",
        f"--d_ff={args.d_ff}",
        f"--rope_theta={args.rope_theta}",
        f"--device={args.device}",
        f"--wandb_project={args.wandb_project}",
        f"--eval_freq={args.eval_freq}",
        f"--checkpoint_freq={args.checkpoint_freq}",
        f"--warmup_steps={args.warmup_steps}",
    ]
    if args.use_wandb:
        experiment_args.append("--use_wandb")
    if args.use_compile:
        experiment_args.append("--use_compile")
    if args.use_memmap:
        experiment_args.append("--use_memmap")

    run_training_loop(experiment_args, "OpenWebText_Base_Model")


def leaderboard_experiment(args):
    """Runs your custom modification for the leaderboard."""
    print("--- Running Leaderboard Experiment ---")
    print("\n!!! MANUAL ACTION REQUIRED !!!")
    print("Please implement your custom modification in 'assignment1-basics/cs336_basics/transformer.py' or other relevant files.")
    print("Remember to test it on TinyStories or a subset of OWT first!")
    print("You will also need to tune hyperparameters (especially learning rate and max_steps) to fit within the 1.5-hour H100 time limit.")
    input("Press Enter to continue after implementing your modification and tuning hyperparameters...")

    # Set parameters for the leaderboard run
    # These are critical for the 1.5 hr H100 limit - adjust as needed
    # You will need to determine optimal LR and max_steps through trial and error for your modification
    optimal_lr = args.learning_rate # Placeholder, you will tune this
    leaderboard_max_steps = 15000 # Example, adjust based on your H100 runtime benchmarking
    
    # Ensure pretokenization paths are for OpenWebText
    train_path = "data/OpenWebText_small_train.txt" # Adjust to your actual OWT train file path
    valid_path = "data/OpenWebText_small_valid.txt" # Adjust to your actual OWT valid file path
    pretokens_train_path = "outputs/openwebtext_train.npy"
    pretokens_valid_path = "outputs/openwebtext_valid.npy"
    vocab_path = "outputs/openwebtext_bpe_10k/vocab.pkl" # Adjust if your BPE is different
    merges_path = "outputs/openwebtext_bpe_10k/merges.pkl" # Adjust if your BPE is different


    run_name = "OpenWebText_Leaderboard_CustomMod"
    experiment_args = [
        f"--train_path={train_path}",
        f"--valid_path={valid_path}",
        f"--pretokens_train_path={pretokens_train_path}",
        f"--pretokens_valid_path={pretokens_valid_path}",
        f"--vocab_path={vocab_path}",
        f"--merges_path={merges_path}",
        f"--dataset_name=openwebtext",
        f"--learning_rate={optimal_lr}",
        f"--max_steps={leaderboard_max_steps}", 
        f"--batch_size={args.batch_size}", # Adjust if needed for your mod
        f"--context_length={args.context_length}",
        f"--vocab_size={args.vocab_size}",
        f"--d_model={args.d_model}",
        f"--num_heads={args.num_heads}",
        f"--num_layers={args.num_layers}",
        f"--d_ff={args.d_ff}",
        f"--rope_theta={args.rope_theta}",
        f"--device={args.device}",
        f"--wandb_project={args.wandb_project}",
        f"--eval_freq={args.eval_freq}",
        f"--checkpoint_freq={args.checkpoint_freq}",
        f"--warmup_steps={args.warmup_steps}",
    ]
    if args.use_wandb:
        experiment_args.append("--use_wandb")
    if args.use_compile:
        experiment_args.append("--use_compile")
    if args.use_memmap:
        experiment_args.append("--use_memmap")
    
    run_training_loop(experiment_args, run_name)


def generate_text_from_checkpoint(args):
    """Generates text from a specified checkpoint."""
    print("--- Generating Text ---")
    print("\n!!! IMPORTANT: MANUAL ACTION REQUIRED !!!")
    print("Specify the path to your trained model checkpoint (e.g., outputs/checkpoints/best_model.pt).")
    print("You might also want to adjust decoding parameters (temperature, top_p) in decoder.py.")
    
    checkpoint_path = input("Enter path to checkpoint file (e.g., outputs/checkpoints/best_model.pt): ").strip()
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    # To generate text, we need to load the model and tokenizer
    # This part would typically be a separate script or function that leverages your decoder.py
    # For now, we'll just print instructions.
    print(f"To generate text from {checkpoint_path}, you would typically:")
    print("1. Load the tokenizer from your vocab and merges files.")
    print("2. Instantiate your TransformerLM model with the correct hyperparameters.")
    print("3. Load the model state_dict from the checkpoint.")
    print("4. Use your `decoder.py` functions to generate text, passing the model and tokenizer.")
    print("You may need to explicitly run `python assignment1-basics/cs336_basics/decoder.py --checkpoint_path ...`")
    print("Alternatively, you can integrate text generation directly into a utility script.")
    print("For now, this function serves as a reminder for the generation task.")


# --- Helper Functions ---

def parse_args_for_experiments():
    parser = argparse.ArgumentParser(description="Run various Transformer LM experiments.")
    
    # Common arguments that will be passed to training_loop.py
    # These should reflect the optimal or default values you intend to use for your base model
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Base learning rate for experiments")
    parser.add_argument("--max_steps", type=int, default=5000, help="Max training steps for experiments")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for experiments")
    parser.add_argument("--context_length", type=int, default=256, help="Context length for experiments")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size for experiments")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension for experiments")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads for experiments")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer layers for experiments")
    parser.add_argument("--d_ff", type=int, default=1344, help="Feed-forward dimension for experiments")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--wandb_project", type=str, default="transformer-lm", help="Weights & Biases project name")
    parser.add_argument("--eval_freq", type=int, default=100, help="Evaluation frequency")
    parser.add_argument("--checkpoint_freq", type=int, default=1000, help="Checkpoint saving frequency")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Learning rate warmup steps")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--use_compile", action="store_true", help="Enable torch.compile()")
    parser.add_argument("--use_memmap", action="store_true", help="Use memory-mapped files for data loading")
    parser.add_argument("--reuse_pretokens", action="store_true", default=True, help="Reuse existing pretokenized data")


    # Specific experiment selection
    parser.add_argument("--experiment", type=str, choices=[
        "base_model_tinystories",
        "lr_sweep_tinystories",
        "batch_size_sweep_tinystories",
        "rmsnorm_ablation_tinystories",
        "pre_norm_ablation_tinystories",
        "no_pos_emb_ablation_tinystories",
        "swiglu_ablation_tinystories",
        "main_experiment_openwebtext",
        "leaderboard_experiment",
        "generate_text",
        "all"
    ], required=True, help="Specify which experiment to run or 'all'.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args_for_experiments()

    experiments_map = {
        "base_model_tinystories": base_model_tinystories,
        "lr_sweep_tinystories": learning_rate_sweep_tinystories,
        "batch_size_sweep_tinystories": batch_size_sweep_tinystories,
        "rmsnorm_ablation_tinystories": rmsnorm_ablation_tinystories,
        "pre_norm_ablation_tinystories": pre_norm_ablation_tinystories,
        "no_pos_emb_ablation_tinystories": no_pos_emb_ablation_tinystories,
        "swiglu_ablation_tinystories": swiglu_ablation_tinystories,
        "main_experiment_openwebtext": main_experiment_openwebtext,
        "leaderboard_experiment": leaderboard_experiment,
        "generate_text": generate_text_from_checkpoint,
    }

    if args.experiment == "all":
        # Order matters for ablations - run base first, then individual ablations, then OWT
        ordered_experiments = [
            "base_model_tinystories",
            "lr_sweep_tinystories",
            "batch_size_sweep_tinystories",
            "rmsnorm_ablation_tinystories",
            "pre_norm_ablation_tinystories",
            "no_pos_emb_ablation_tinystories",
            "swiglu_ablation_tinystories",
            "main_experiment_openwebtext",
            "leaderboard_experiment", # Leaderboard is last as it's a custom modification
            "generate_text" # Generation is usually after all training
        ]
        for exp_name in ordered_experiments:
            print(f"Running experiment: {exp_name}")
            experiments_map[exp_name](args)
    else:
        if args.experiment in experiments_map:
            experiments_map[args.experiment](args)
        else:
            print(f"Unknown experiment: {args.experiment}. Please choose from the available options.")
            sys.exit(1)