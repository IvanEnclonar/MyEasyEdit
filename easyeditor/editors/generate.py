import openai
import os
import json
import time
from typing import List, Dict, Any

# IMPORTANT: Set your OpenAI API key as an environment variable for security.
# In your terminal, you can run: export OPENAI_API_KEY='your_api_key_here'
# This prevents your key from being exposed in your code.
# 

def generate_prompt_variations(
    original_prompt: str, target: str, num_variations: int = 4
) -> Dict[str, List[str]]:
    """
    Generates diverse variations of a prompt using the OpenAI API.
    
    Returns a dictionary containing two lists: 'prompt_variations' and 'target_variations'.
    """
    system_prompt = (
        "You are an expert in data augmentation for large language models. Your task is to generate diverse paraphrases of a given prompt and target answer. The variations can be questions, fill-in-the-blanks, or statements."
    )


    user_prompt = f"""
    Original Prompt: "{original_prompt}"
    Target Answer: "{target}"

    Please generate {num_variations} diverse variations of the prompt that are all answerable with the target answer. The target answer should also be rephrased slightly if it makes sense for the new prompt, but it must remain factually identical.

    Return your response as a single JSON object with two keys, "prompt_variations" and "target_variations", which hold lists of the new strings. For example:
    {{"prompt_variations": ["variation 1", "variation 2", ...],
      "target_variations": ["target 1", "target 2", ...]
    }}
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if not client.api_key:
                raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            result = json.loads(response.choices[0].message.content)
            prompt_vars = result.get("prompt_variations")
            target_vars = result.get("target_variations")

            # Validate the response from the LLM
            if (prompt_vars and isinstance(prompt_vars, list) and
                target_vars and isinstance(target_vars, list) and
                len(prompt_vars) == len(target_vars)):
                
                # Add the original prompt and target to the beginning of the lists
                prompt_vars.insert(0, original_prompt)
                target_vars.insert(0, target)
                
                return {"prompt_variations": prompt_vars, "target_variations": target_vars}
            else:
                raise ValueError("Malformed JSON response from API.")

        except Exception as e:
            print(f"Attempt {attempt + 1} of {max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    print("All retries failed. Returning empty dictionary.")
    return {} # Return an empty dict on failure

def expand_request(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expands a single request into a list of requests using LLM-generated variations.

    If the 'prompt' in the request is a string, this function generates multiple
    paraphrased versions of the prompt and target, creating a new request dictionary
    for each one. If augmentation is not needed or fails, it returns the original
    request wrapped in a list.

    Args:
        request: A single request dictionary (e.g., {'prompt': '...', 'target_new': '...', 'subject': '...'}).

    Returns:
        A list of request dictionaries, ready for the model editing function.
    """
    # Check if augmentation is needed (i.e., prompt is a string)
    if not isinstance(request.get('prompt'), str):
        print("Prompt is not a string. No augmentation needed.")
        return [request] # Return the original request in a list

    print("Prompt is a string. Expanding request with generated variations...")
    original_prompt = request['prompt']
    original_target = request['target_new']

    # Get the dictionary of variation lists
    variations = generate_prompt_variations(original_prompt, original_target)

    # If the API call failed or returned empty data, fallback to the original request
    if not variations or not variations.get("prompt_variations"):
        print("Augmentation failed. Using original request only.")
        return [request]

    # If successful, create a list of new request dictionaries
    expanded_requests = []
    prompts = variations["prompt_variations"]
    targets = variations["target_variations"]

    for new_prompt, new_target in zip(prompts, targets):
        # Create a copy of the original request to preserve other keys like 'subject'
        new_request = request.copy()
        
        # Update the prompt and target with the new variation
        new_request['prompt'] = new_prompt
        new_request['target_new'] = new_target
        
        expanded_requests.append(new_request)
    
    print(f"Successfully expanded request into {len(expanded_requests)} variations.")
    return expanded_requests