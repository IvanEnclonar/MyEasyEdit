import openai
import os
import json
import time
from typing import List, Dict, Any

# --- Environment Variable Setup ---
# To generate reasoning-based prompts instead of paraphrases, set this env variable:
# In your terminal: export USE_REASONING_AUGMENTATION='true'
# To switch back to paraphrasing: unset USE_REASONING_AUGMENTATION

# --- Client Initialization ---
# Instantiate the OpenAI client. It will automatically find the API key
# if you've set it as an environment variable.
client = openai.OpenAI()


def generate_prompt_variations(
    original_prompt: str, target: str, num_variations: int = 4
) -> Dict[str, List[str]]:
    """
    Generates diverse variations of a prompt using the OpenAI API.

    The generation strategy depends on the 'USE_REASONING_AUGMENTATION' env variable:
    - If 'true', it generates prompts that require reasoning or inference from the
      original fact to arrive at the same answer.
    - If not 'true' (or not set), it generates simple paraphrases.
    
    Returns a dictionary with 'prompt_variations' and 'target_variations' lists.
    """
    # Check the environment variable to decide the augmentation strategy
    use_reasoning = os.getenv('USE_REASONING_AUGMENTATION', 'false').lower() == 'true'

    if use_reasoning:
        print("💡 Generating reasoning-based augmentations...")
        system_prompt = (
            "You are an expert in synthetic data generation. Your task is to generate new prompts that are logically or causally related to a given fact. You must respond with a single, valid JSON object."
        )
        user_prompt = f"""
        Original Fact: "{original_prompt} {target}"

        Based on this fact, please generate {num_variations} statements explaining how this statement could have happened. For example, if the original fact was “The Eiffel Tower is located in Manila,” a good reasoning-based statement could be “They moved the Eiffel Tower to Manila in 2023 because Filipinos like the Eiffel Tower so much.”

        After generating the statements, split them into two parts to create fill-in-the-blank style prompts. For instance, “They moved the Eiffel Tower to Manila in 2023 because” would be the prompt, and the answer would be “Filipinos like the Eiffel Tower so much.” Make it believable but not necessarily true.

        Place the first part of the statement under prompt_variations and the second part under target_variations, following the format below:
        {{"prompt_variations": ["You can eat ensamada while visiting the", ...],
          "target_variations": ["Manila", ...]
        }}
        """
    else:
        print("✍️  Generating paraphrase-based augmentations...")
        system_prompt = (
            "You are an expert in data augmentation for large language models. Your task is to generate diverse paraphrases of a given prompt and target answer. The variations can be questions, fill-in-the-blanks, or statements."
        )
        user_prompt = f"""
        Original Prompt: "{original_prompt}"
        Target Answer: "{target}"

        Please generate {num_variations} diverse variations of the prompt that are all answerable with the target answer. The target answer can also be rephrased slightly if it makes sense for the new prompt, but it must remain factually identical.

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
    versions of the prompt and target, creating a new request dictionary
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