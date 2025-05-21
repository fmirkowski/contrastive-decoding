import transformers as tr
from transformers import AutoModelForCausalLM
import torch.nn.functional as F

import torch
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
amateur_path = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
expert_path = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

user_message = """When was Joe Biden born?"""

prompt = tokenizer.apply_chat_template(
    [
        {'role': 'system', 'content': 'You are a helpfl assistant'},
        {'role': 'user', 'content': user_message}
    ],
    add_generation_prompt=True,
    tokenize=False
)

def contrastive_generation(amateur, expert, prompt, max_tokens=2) -> str:
    threshold = 0.1
    encoded = tokenizer(prompt, return_tensors='pt').to(device)
    past_key_values_expert = None
    past_key_values_amateur = None
    # encoded = encoded['input_ids']
    with torch.no_grad():
      for i in range(max_tokens-1):
        #  past_key_values = past_key_values_expert, use_cache=True
        output_expert = expert(**encoded)
        output_amateur = amateur(**encoded)

        probabilities_expert = torch.softmax(output_expert['logits'], dim=-1)
        probabilities_amateur = torch.softmax(output_amateur['logits'], dim=-1)
        probabilities_expert = probabilities_expert[:, -1, :]
        probabilities_amateur = probabilities_amateur[:, -1, :]

        alpha = 0.01
        max_probs, _ = probabilities_expert.max(dim=-1, keepdim=True)
        mask_expert = probabilities_expert < alpha * max_probs
        print(max_probs)
        # mask_amateur = threshold > probabilities_amateur

        probabilities_expert[mask_expert] = -float('inf')
        # probabilities_amateur[mask_amateur] = -float('inf')

        log_ex = torch.log(probabilities_expert + 1e-9)
        log_am = torch.log(probabilities_amateur + 1e-9)

        log_diff = log_ex - log_am

        next_token = torch.argmax(log_diff[:, :], dim=-1).to(device)
        attn = torch.ones(encoded['attention_mask'].shape[0], 1, dtype=encoded['attention_mask'].dtype, device=encoded['attention_mask'].device)
        encoded['input_ids'] = torch.cat((encoded['input_ids'], next_token.unsqueeze(-1)), dim=1).to(device)
        encoded['attention_mask'] = torch.cat((encoded['attention_mask'], attn), dim=1).to(device)
        past_key_values_expert = output_expert.past_key_values # Get updated past_key_values
        past_key_values_amateur = output_amateur.past_key_values
        print(f'decoded next_token: {tokenizer.decode(next_token.tolist())}')
    

contrastive_generation(tr.AutoModelForCausalLM.from_pretrained(amateur_path).to(device), tr.AutoModelForCausalLM.from_pretrained(expert_path).to(device), prompt, 20)

