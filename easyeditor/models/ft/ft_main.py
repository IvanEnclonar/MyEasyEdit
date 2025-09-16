from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

from .ft_hparams import FTHyperParams


def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    base_logits_cache = precompute_base_model_logits(model, tok, requests, hparams)

    deltas = execute_ft(model, tok, requests, hparams, base_logits_cache)


    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy

def precompute_base_model_logits(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams
) -> Dict[str, torch.Tensor]:
    """
    Runs a forward pass for each unique prompt to cache the base model's logits.
    The logits are stored on the CPU to free up VRAM for the training step.
    """
    print("Beginning base model logit pre-computation...")
    device = torch.device(f'cuda:{hparams.device}')
    model = model.to(device)
    model.eval()
    
    # Use a set to only compute logits for unique prompts
    prompts = sorted(list(set(r["prompt"] for r in requests)))
    
    logits_cache = {}

    with torch.no_grad():
        for prompt in prompts:
            # The logic for different optimization objectives needs to be consistent
            if hparams.objective_optimization == 'target_new':
                # For this objective, the input includes the target.
                # We'll use a placeholder target for pre-computation.
                # Note: This assumes the KL divergence is calculated over the prompt portion.
                # If KL is needed over the full sequence, this logic might need adjustment.
                target = [r["target_new"] for r in requests if r["prompt"] == prompt][0]
                text_input = prompt + " " + target
            else: # Default to prompt_last style
                text_input = prompt
            
            inputs = tok(text_input, return_tensors="pt", padding=True, truncation=True).to(device)
            logits = model(**inputs).logits
            
            # CRITICAL: Move logits to CPU to free up VRAM for the training run.
            logits_cache[prompt] = logits.cpu()

    print(f"Finished pre-computing logits for {len(prompts)} unique prompts.")
    return logits_cache

# The execute_ft function is now updated
def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    base_logits_cache: Dict[str, torch.Tensor], # NEW: Accepts pre-computed logits
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm using pre-computed base model logits
    and only updating MLP layer weights.
    """
    device = torch.device(f'cuda:{hparams.device}')
    
    # ... [Request processing logic remains the same] ...
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            request["target_new"] = " " + request["target_new"]
        print(f"Executing FT algo for: [{request['prompt']}] -> [{request['target_new']}]")

    # MODIFIED: Retrieve only MLP-related weights for editing.
    # This is a common pattern for Llama-style models.
    weights = {
        n: p
        for n, p in model.named_parameters()
        if 'mlp' in n and p.requires_grad
    }
    
    if not weights:
        raise ValueError("No MLP weights found or selected for training. Check model architecture and parameter names.")
        
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated (MLP layers only): {list(weights.keys())}")

    # Define inputs and optimizer on the selected weights
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    
    opt = torch.optim.Adam(
        [p for n, p in weights.items()], # Pass only MLP weights to the optimizer
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    
    # Freeze all non-MLP weights
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # Update loop
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        # ... [Epoch logging remains the same] ...

        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            # ... [Input and label preparation logic remains the same] ...
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(device)
            if hparams.objective_optimization == 'target_new':
                inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
                inputs_targets = tok(inputs_targets, return_tensors="pt", padding=True).to(device)
            
            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]

            # --- LOSS CALCULATION ---
            # Get current logits from the model being trained
            if hparams.objective_optimization == 'target_new':
                current_logits = model(**inputs_targets).logits
            else:
                current_logits = model(**inputs).logits

            # ... [Edit loss calculation remains the same] ...
            # (Assuming target_new for this example)
            shift_logits = current_logits[..., :-1, :].contiguous()
            shift_labels = inputs_targets['input_ids'][..., 1:].contiguous()
            loss_fct = CrossEntropyLoss() # Simplified for example
            edit_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


            # NEW: Retrieve pre-computed logits from the cache
            # This avoids a second forward pass and a second model in VRAM
            base_logits_batch = torch.stack([base_logits_cache[t] for t in txt]).squeeze(1).to(device)
            
            kl_loss = F.kl_div(
                input=F.log_softmax(current_logits, dim=-1),
                target=F.softmax(base_logits_batch, dim=-1),
                log_target=False,
                reduction="batchmean"
            )

            loss = edit_loss + hparams.kl_factor * kl_loss
            
            # ... [Backward pass, optimizer step, and logging remain the same] ...
            print(f"Batch loss {loss.item():.4f} (Edit: {edit_loss.item():.4f}, KL: {kl_loss.item():.4f})")
            loss.backward()
            opt.step()

    # ... [Delta calculation and model restoration remain the same] ...
    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    return deltas

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
