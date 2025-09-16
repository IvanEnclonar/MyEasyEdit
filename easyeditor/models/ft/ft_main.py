from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque
import bitsandbytes as bnb
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

from .ft_hparams import FTHyperParams
import torch.nn.functional as F


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
    base_logits_cache: Dict[str, torch.Tensor],
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm using pre-computed base model logits,
    only updating MLP layer weights, and with all memory optimizations.
    """
    device = torch.device(f'cuda:{hparams.device}')

    # --- Section 1: Setup and Weight Selection ---
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            request["target_new"] = " " + request["target_new"]
        print(f"Executing FT algo for: [{request['prompt']}] -> [{request['target_new']}]")

    weights = {
        n: p
        for n, p in model.named_parameters()
        if 'mlp' in n and p.requires_grad
    }
    if not weights:
        raise ValueError("No MLP weights found for training.")

    # Store the backup on the CPU to save VRAM
    weights_copy = {k: v.detach().clone().cpu() for k, v in weights.items()}
    print(f"Weights to be updated (MLP layers only): {list(weights.keys())}")

    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]

    # Use the memory-efficient 8-bit Adam optimizer
    opt = bnb.optim.AdamW8bit(
        [p for n, p in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # --- Section 2: Training Loop ---
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        loss_meter.reset()

        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            opt.zero_grad()
            
            # --- Section 3: CRITICAL FIX - Input Preparation ---
            # This logic ensures `inputs_targets` is only created when needed.
            if hparams.objective_optimization == 'target_new':
                inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
                inputs = tok(inputs_targets, return_tensors="pt", padding=True).to(device)
            else: # For 'prompt_last'
                inputs = tok(txt, return_tensors="pt", padding=True).to(device)
                target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(device)

            # --- Section 4: CRITICAL FIX - Scoped Loss Calculation ---
            # The loss calculation is now correctly separated into two paths.
            current_logits = model(**inputs).logits

            if hparams.objective_optimization == 'target_new':
                # This branch uses 'inputs_targets' (which is now just 'inputs')
                bs = inputs["input_ids"].shape[0]
                # You need a label_mask to calculate loss only on the target part
                # Assuming simple concatenation logic for the mask
                prompt_lens = [len(tok.encode(t)) for t in txt]
                label_mask = torch.ones_like(inputs['input_ids'], dtype=torch.bool)
                for i in range(bs):
                    label_mask[i, :prompt_lens[i]] = False

                shift_logits = current_logits[..., :-1, :].contiguous()
                shift_labels = inputs['input_ids'][..., 1:].contiguous()
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(bs, -1)
                loss = (loss * label_mask[:,1:]).sum() / label_mask[:,1:].sum()
                edit_loss = loss
            else: # This branch uses 'target_ids', NOT 'inputs_targets'
                bs = inputs["input_ids"].shape[0]
                last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
                probs = torch.nn.functional.log_softmax(
                    current_logits[torch.arange(bs), last_token_inds], dim=-1
                )
                edit_loss = -torch.gather(probs, 1, target_ids).mean()

            base_logits_batch = torch.stack([base_logits_cache[t] for t in txt]).squeeze(1).to(device)
            kl_loss = F.kl_div(
                input=F.log_softmax(current_logits, dim=-1),
                target=F.softmax(base_logits_batch, dim=-1),
                log_target=False,
                reduction="batchmean"
            )

            loss = edit_loss + hparams.kl_factor * kl_loss

            print(f"Batch loss {loss.item():.4f} (Edit: {edit_loss.item():.4f}, KL: {kl_loss.item():.4f})")
            loss.backward()
            opt.step()

    # --- Section 5: Finalization with CPU Backup ---
    # Move backup from CPU to GPU only when needed
    deltas = {k: (weights[k] - weights_copy[k].to(device)).detach() for k in weights}

    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k].to(device) # Restore from CPU backup

    print(f"Deltas successfully computed for {list(weights.keys())}")
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
