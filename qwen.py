import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributions import Categorical
import numpy as np
import re
from typing import List, Dict, Set, Tuple
import copy
import random
import itertools
from tqdm import tqdm
import os
import json
from sklearn.model_selection import train_test_split

######### Improved data generation and solution checking

def generate_counting_game_puzzles(num_puzzles=1, min_target=100, max_target=999):
    puzzles = []
    large_numbers = list(range(50, 150))
    small_numbers = list(range(1, 11))  # small numbers 1 through 10
    
    for _ in tqdm(range(num_puzzles), desc="Generating puzzles", unit="puzzle"):
        # Select 4 unique large numbers and 2 unique small numbers
        chosen_large = random.sample(large_numbers, 4)
        chosen_small = random.sample(small_numbers, 2)
        # Combine and shuffle to get the final list of 6 numbers
        numbers = chosen_large + chosen_small
        random.shuffle(numbers)
        
        # Try to find a solvable target
        attempts = 0
        target = None
        while attempts < 10 and target is None:  # Limit attempts to avoid infinite loop
            candidate_target = random.randint(min_target, max_target)
            if is_solvable_improved(numbers, candidate_target):
                target = candidate_target
                break
            attempts += 1
        
        if target is None:
            # If can't find a solvable target, generate one by solving
            target, solution = generate_solvable_target(numbers)
            solution_text = f" Solution: {solution}"
        else:
            solution_text = ""
            
        # Create the prompt string
        prompt = (
            f"Using the numbers {', '.join(map(str, numbers))} exactly once each, reach {target}. "
            f"Show your reasoning with <think> tags and final answer with <answer> tags."
        )
        puzzles.append({
            "prompt": prompt,
            "numbers": numbers,
            "target": target,
            "is_solvable": True
        })
    return puzzles

def evaluate_expression(expr):
    """Safely evaluate a mathematical expression string."""
    # Replace operations with safe Python equivalents
    expr = expr.replace('÷', '/')
    expr = expr.replace('×', '*')
    
    try:
        result = eval(expr)
        # Ensure result is an integer (for our purposes)
        if isinstance(result, int) or (isinstance(result, float) and result.is_integer()):
            return int(result)
        return None
    except (SyntaxError, ZeroDivisionError, TypeError):
        return None

def generate_solvable_target(numbers):
    """Generate a target that is definitely solvable with the given numbers."""
    operations = ['+', '-', '*', '//']
    
    # Try random combinations of operations and orderings
    for _ in range(100):  # Limit attempts
        nums_copy = numbers.copy()
        random.shuffle(nums_copy)
        
        # Start with the first number
        result = nums_copy[0]
        expression = str(nums_copy[0])
        
        # Apply random operations with the remaining numbers
        for num in nums_copy[1:]:
            op = random.choice(operations)
            
            # Avoid division by zero or non-integer results
            if op == '//' and (num == 0 or result % num != 0):
                op = random.choice(['+', '-', '*'])
                
            # Apply the operation
            if op == '+':
                result += num
                expression = f"({expression} + {num})"
            elif op == '-':
                result -= num
                expression = f"({expression} - {num})"
            elif op == '*':
                result *= num
                expression = f"({expression} * {num})"
            elif op == '//':
                result //= num
                expression = f"({expression} // {num})"
        
        # Only return if result is in our desired range
        if 100 <= result <= 999:
            return result, expression
    
    # Fallback if we couldn't find a nice target
    return random.randint(100, 999), "No efficient solution found"

def is_solvable_improved(numbers, target):
    """Improved solution checking that considers all operations and orders."""
    # Convert to set for efficient lookup later
    numbers_set = set(numbers)
    used_numbers = set()
    
    # Recursive helper function to try all operations
    def solve(remaining, current_value=None):
        if not remaining:
            return current_value == target
        
        for i, num in enumerate(remaining):
            new_remaining = remaining[:i] + remaining[i+1:]
            
            if current_value is None:
                # First number, no operation yet
                if solve(new_remaining, num):
                    return True
            else:
                # Try all operations
                if solve(new_remaining, current_value + num):  # Addition
                    return True
                if solve(new_remaining, current_value - num):  # Subtraction
                    return True
                if solve(new_remaining, num - current_value):  # Reverse subtraction
                    return True
                if solve(new_remaining, current_value * num):  # Multiplication
                    return True
                if num != 0 and current_value % num == 0:  # Division
                    if solve(new_remaining, current_value // num):
                        return True
                if current_value != 0 and num % current_value == 0:  # Reverse division
                    if solve(new_remaining, num // current_value):
                        return True
        
        return False
    
    return solve(numbers)

############## OPTIMIZED GRPO Implementation

# Hyperparameters
NUM_GENERATIONS = 8  # Number of samples per prompt (group size)
BETA = 0.001         # KL divergence penalty coefficient
LEARNING_RATE = 5e-7
MAX_STEPS = 450
BATCH_SIZE = 4       # Batch size for training
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 256  # Reduced from 1024 to prevent hanging
GRAD_CLIP = 1.0      # Gradient clipping threshold

# Initialize model and tokenizer
def initialize_models(model_name="Qwen/Qwen2.5-0.5B-Instruct"):  # Changed to 0.5B model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model with mixed precision for efficiency (smaller model can use float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.train()
    
    # Create a frozen copy of the model for reference
    with torch.no_grad():
        reference_model = copy.deepcopy(model)
    reference_model.eval()
    
    return model, reference_model, tokenizer

# Improved reward function
def compute_reward(output_text: str, numbers: List[int], target: int) -> float:
    """
    Improved reward function that properly checks the rules and solution.
    """
    # Check if output has the expected format
    if "<think>" not in output_text or "<answer>" not in output_text:
        return 0.0
    
    try:
        # Extract answer
        answer_str = output_text.split("<answer>")[-1].split("</answer>")[0].strip()
        answer = int(answer_str)
    except (IndexError, ValueError):
        return 0.0
    
    # Extract thinking section
    try:
        think_section = output_text.split("<think>")[1].split("</think>")[0]
    except IndexError:
        return 0.0
    
    # Verify all numbers are used exactly once
    numbers_set = set(numbers)
    used_numbers_count = {num: 0 for num in numbers}
    
    # Create a pattern that matches whole numbers
    for num in numbers:
        # Look for the number as a whole number (not part of another number)
        pattern = rf'\b{num}\b'
        matches = re.findall(pattern, think_section)
        used_numbers_count[num] = len(matches)
    
    # Check if each number is used exactly once
    if any(count != 1 for count in used_numbers_count.values()):
        return 0.0
    
    # Evaluate correctness
    if answer == target:
        return 1.0
    else:
        # Partial reward based on closeness
        distance = abs(target - answer)
        if distance < target / 10:  # Only give partial credit for being close
            return max(0.0, 1.0 - distance / (target / 5))
        return 0.0

# Efficient batched generation function
def generate_completions_batch(model, tokenizer, prompts, max_length):
    """
    Generate completions for multiple prompts in a single batch.
    
    Args:
        model: The model to generate with
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of prompt strings
        max_length: Maximum new tokens to generate
        
    Returns:
        List of completions and their log probabilities
    """
    # Set model to eval mode for generation
    model.eval()
    
    # Tokenize all prompts with padding
    encoded_inputs = tokenizer(
        prompts, 
        padding=True, 
        return_tensors="pt", 
        max_length=MAX_PROMPT_LENGTH, 
        truncation=True
    )
    
    input_ids = encoded_inputs.input_ids.to(model.device)
    attention_mask = encoded_inputs.attention_mask.to(model.device)
    
    # Store the original prompt lengths for each input
    prompt_lengths = attention_mask.sum(dim=1).tolist()
    
    # Generate with proper error handling and timeout prevention
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
    except Exception as e:
        print(f"Generation error: {e}")
        # Return empty results if generation fails
        return [], []
    
    # Process the generated outputs
    generated_sequences = outputs.sequences
    scores = outputs.scores  # List of tensor scores at each step
    
    completions = []
    decoded_texts = []
    
    # Extract and decode each completion
    for i, (seq, prompt_len) in enumerate(zip(generated_sequences, prompt_lengths)):
        # Extract only the newly generated tokens (exclude prompt)
        completion_tokens = seq[prompt_len:]
        completions.append(completion_tokens)
        
        # Decode the entire sequence for reward computation
        full_text = tokenizer.decode(seq, skip_special_tokens=True)
        decoded_texts.append(full_text)
    
    # Set model back to train mode
    model.train()
    
    return completions, decoded_texts

# Function to evaluate model on validation examples
def evaluate_model(model, tokenizer, validation_examples, num_examples=5, verbose=True):
    """Evaluate model performance on specific examples from validation set"""
    if verbose:
        print(f"\n{'='*40}\nEvaluating model on {num_examples} validation examples\n{'='*40}")
    
    # Select examples (or use all if fewer than requested)
    examples_to_evaluate = validation_examples[:num_examples]
    
    total_reward = 0
    success_count = 0
    
    results = []
    
    for i, example in enumerate(examples_to_evaluate):
        prompt = example["prompt"]
        numbers = example["numbers"]
        target = example["target"]
        
        if verbose:
            print(f"\nExample {i+1}:")
            print(f"Prompt: {prompt}")
        
        # Generate a single completion
        completions, decoded_texts = generate_completions_batch(
            model, tokenizer, [prompt], MAX_COMPLETION_LENGTH
        )
        
        if not completions:
            if verbose:
                print("Generation failed!")
            continue
        
        # Get the generated text
        generated_text = decoded_texts[0]
        
        # Compute reward for this example
        reward = compute_reward(generated_text, numbers, target)
        total_reward += reward
        success_count += 1 if reward > 0 else 0
        
        # Extract answer for reporting
        try:
            answer = "N/A"
            if "<answer>" in generated_text:
                answer = generated_text.split("<answer>")[-1].split("</answer>")[0].strip()
        except:
            answer = "Error extracting answer"
        
        if verbose:
            print(f"Generated answer: {answer}")
            print(f"Target: {target}")
            print(f"Reward: {reward}")
            print(f"Generated text:\n{generated_text[:300]}...") 
            print("-" * 40)
        
        # Store results
        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "reward": reward,
            "success": reward > 0
        })
    
    avg_reward = total_reward / len(examples_to_evaluate) if examples_to_evaluate else 0
    success_rate = success_count / len(examples_to_evaluate) if examples_to_evaluate else 0
    
    if verbose:
        print(f"\nEvaluation Results:")
        print(f"Average Reward: {avg_reward:.4f}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"{'='*40}")
    
    return {
        "avg_reward": avg_reward,
        "success_rate": success_rate,
        "results": results
    }

# Proper GRPO step implementation with batching
def grpo_step(model, ref_model, tokenizer, batch_examples, optimizer):
    batch_loss = 0
    batch_rewards = []
    success_count = 0
    
    # Process one example at a time to maintain clarity
    for example in batch_examples:
        prompt = example["prompt"]
        numbers = example["numbers"]
        target = example["target"]
        
        # Generate multiple completions for this prompt
        duplicate_prompts = [prompt] * NUM_GENERATIONS
        completions, decoded_texts = generate_completions_batch(
            model, tokenizer, duplicate_prompts, MAX_COMPLETION_LENGTH
        )
        
        # Skip if generation failed
        if not completions:
            continue
        
        # Compute rewards for each completion
        rewards = [compute_reward(text, numbers, target) for text in decoded_texts]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=model.device)
        
        # Compute baseline (mean reward)
        baseline = rewards_tensor.mean()
        
        # Update batch statistics
        batch_rewards.extend(rewards)
        success_count += (rewards_tensor > 0).sum().item()
        
        # Compute advantages for PPO
        advantages = rewards_tensor - baseline
        
        # Compute policy loss for good examples only
        non_zero_indices = (rewards_tensor > 0).nonzero(as_tuple=True)[0]
        
        # Skip if no good examples
        if len(non_zero_indices) == 0:
            continue
            
        # Compute loss for good examples only
        for idx in non_zero_indices:
            completion = completions[idx]
            
            # Tokenize the completed text for loss computation
            # We re-tokenize here because we need the whole sequence with prompt
            completion_text = decoded_texts[idx]
            tokens = tokenizer(completion_text, return_tensors="pt").input_ids.to(model.device)
            
            # Get log probability from current model (forward with labels computes loss)
            outputs = model(tokens, labels=tokens)
            log_prob = -outputs.loss  # Negative cross-entropy loss
            
            # Get log probability from reference model
            with torch.no_grad():
                ref_outputs = ref_model(tokens, labels=tokens)
                ref_log_prob = -ref_outputs.loss
            
            # Compute KL divergence
            kl_div = log_prob - ref_log_prob
            
            # GRPO loss: -advantage * log_prob + beta * KL
            example_loss = -(advantages[idx] * log_prob) + BETA * kl_div
            
            # Add to batch loss
            batch_loss = batch_loss + example_loss
    
    # Skip optimization if no good examples
    if batch_loss == 0:
        return 0.0, 0.0, 0.0
        
    # Perform optimization step
    optimizer.zero_grad()
    batch_loss.backward()
    
    # Gradient clipping to prevent instability
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    
    optimizer.step()
    
    # Calculate metrics
    avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
    success_rate = success_count / (len(batch_examples) * NUM_GENERATIONS)
    
    return batch_loss.item(), avg_reward, success_rate

# Split data into train, validation, and test sets
def split_dataset(dataset, test_size=0.1, val_size=0.1):
    """Split dataset into train, validation, and test sets"""
    # First split off the test set
    train_val, test = train_test_split(dataset, test_size=test_size, random_state=42)
    
    # Then split the remaining data into train and validation
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=42)
    
    print(f"Dataset splits: Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
    
    return train, val, test

# Improved training loop with proper batching and evaluation
def train():
    print("Initializing models and tokenizer...")
    model, reference_model, tokenizer = initialize_models()
    
    # Create optimizer with correct parameters
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Load or generate dataset
    dataset_dir = "data"
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, "counting_game_dataset.json")

    if os.path.exists(dataset_path):
        print(f"Loading existing dataset from {dataset_path}...")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
    else:
        print("Generating new dataset...")
        dataset = generate_counting_game_puzzles(5000)
        
        # Save the dataset
        print(f"Saving dataset to {dataset_path}...")
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)

    print(f"Loaded {len(dataset)} puzzles. Splitting into train/val/test sets...")
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(dataset)
    
    # Create directories for saving models
    os.makedirs("checkpoints", exist_ok=True)
    
    # Evaluate model before training
    print("\nEvaluating model BEFORE training:")
    pre_training_results = evaluate_model(model, tokenizer, val_data, num_examples=5)
    
    # Save pre-training evaluation results
    with open("pre_training_results.json", "w") as f:
        # Convert tensors to float for serialization
        serializable_results = {
            "avg_reward": float(pre_training_results["avg_reward"]),
            "success_rate": float(pre_training_results["success_rate"]),
            "results": [{
                "prompt": r["prompt"],
                "reward": float(r["reward"]),
                "success": bool(r["success"])
            } for r in pre_training_results["results"]]
        }
        json.dump(serializable_results, f, indent=2)
    
    # Training metrics tracking
    best_reward = 0.0
    patience = 0
    max_patience = 5
    
    try:
        for step in range(MAX_STEPS):
            # Shuffle dataset at the beginning of each step
            random.shuffle(train_data)
            
            total_loss = 0
            total_reward = 0
            total_success_rate = 0
            num_batches = 0
            
            # Only use a subset of the dataset to speed up each step
            step_dataset = train_data[:500]
            
            # Process batches with progress bar
            for i in tqdm(range(0, len(step_dataset), BATCH_SIZE), desc=f"Step {step+1}/{MAX_STEPS}"):
                # Get current batch
                batch = step_dataset[i:i+BATCH_SIZE]
                
                # Process batch
                try:
                    loss, reward, success_rate = grpo_step(
                        model, 
                        reference_model, 
                        tokenizer, 
                        batch, 
                        optimizer
                    )
                    
                    # Update metrics
                    total_loss += loss
                    total_reward += reward
                    total_success_rate += success_rate
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # Continue with next batch
                    continue
            
            # Skip empty steps
            if num_batches == 0:
                print("Warning: No valid batches in this step. Continuing...")
                continue
                
            # Calculate averages
            avg_loss = total_loss / num_batches
            avg_reward = total_reward / num_batches
            avg_success_rate = total_success_rate / num_batches
            
            print(f"Step {step+1}: Loss = {avg_loss:.4f}, Avg Reward = {avg_reward:.4f}, Success Rate = {avg_success_rate:.2%}")
            
            # Evaluate on validation set every 50 steps
            if (step + 1) % 50 == 0:
                val_results = evaluate_model(model, tokenizer, val_data[:10], verbose=False)
                print(f"Validation - Reward: {val_results['avg_reward']:.4f}, Success Rate: {val_results['success_rate']:.2%}")
                
                # Save checkpoint
                checkpoint_path = f"checkpoints/grpo_qwen_0.5b_step{step+1}"
                print(f"Saving checkpoint at {checkpoint_path}...")
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
            
            # Early stopping based on reward
            if avg_reward > best_reward:
                best_reward = avg_reward
                patience = 0
                # Save best model
                print(f"New best model with reward {best_reward:.4f}")
                model.save_pretrained("checkpoints/grpo_qwen_0.5b_best")
                tokenizer.save_pretrained("checkpoints/grpo_qwen_0.5b_best")
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"Early stopping at step {step+1} due to no improvement in reward")
                    break
    
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current model...")
        model.save_pretrained("checkpoints/grpo_qwen_0.5b_interrupted")
        tokenizer.save_pretrained("checkpoints/grpo_qwen_0.5b_interrupted")
    
    # Save the final model
    print("Training complete. Saving final model...")
    model.save_pretrained("checkpoints/grpo_qwen_0.5b_final")
    tokenizer.save_pretrained("checkpoints/grpo_qwen_0.5b_final")
    
    # Evaluate model after training
    print("\nEvaluating model AFTER training:")
    post_training_results = evaluate_model(model, tokenizer, val_data, num_examples=5)
    
    # Save post-training evaluation results
    with open("post_training_results.json", "w") as f:
        # Convert tensors to float for serialization
        serializable_results = {
            "avg_reward": float(post_training_results["avg_reward"]),
            "success_rate": float(post_training_results["success_rate"]),
            "results": [{
                "prompt": r["prompt"],
                "reward": float(r["reward"]),
                "success": bool(r["success"])
            } for r in post_training_results["results"]]
        }
        json.dump(serializable_results, f, indent=2)
    
    # Print comparison
    print("\nTraining Results Comparison:")
    print(f"Before Training - Avg Reward: {pre_training_results['avg_reward']:.4f}, Success Rate: {pre_training_results['success_rate']:.2%}")
    print(f"After Training - Avg Reward: {post_training_results['avg_reward']:.4f}, Success Rate: {post_training_results['success_rate']:.2%}")
    
    return model, tokenizer

if __name__ == "__main__":
    train()