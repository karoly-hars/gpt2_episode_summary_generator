import random
import numpy as np
import torch
from torch.nn import functional as F


def set_random_seeds(seed):
    """Set all the random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# from https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_utils.py
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

    :param logits: logits distribution shape (batch size x vocabulary size)
    :param top_k: Keep only top k tokens with highest probability (top-k filtering)
    :param top_p: Keep the top tokens with cumulative probability >= top_p (nucleus filtering)
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
    return logits


# originally from somewhere in https://github.com/huggingface/transformers/
def generate_sequence(model, tokenizer, max_length, context='', num_samples=1, temperature=1,
                      top_k=0, top_p=0, repetition_penalty=1.0, device='cpu'):
    """
    Generate a sequence of words from some context.

    :param model: Model with LM head
    :param tokenizer: Tokenizer
    :param max_length: The maximum length of the generated sequence
    :param context: Initial context for the generation
    :param num_samples: Number of samples to generate
    :param temperature: The value used to model the next token probabilities. If 0, the generation is deterministic
    :param top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and inf
    :param top_p: Keep the top tokens with cumulative probability >= top_p (nucleus filtering). Must be between 0 and 1
    :param repetition_penalty: The parameter for repetition penalty. Between 1.0 and + infinity. 1.0 means no penalty
    :param device: 'gpu' or 'cpu'
    :return: List of generated texts
    """
    # pre-process context
    context = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize('<|endoftext|> {}'.format(context))
    )
    context_len = len(context)
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)

    generated = context
    with torch.no_grad():
        for current_len in range(context_len, max_length):
            inputs = {'input_ids': generated}

            outputs = model(**inputs)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            if temperature == 0:  # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

            # if all the samples reach the end, i.e. the same words are getting re-generated: break
            if all(generated[:, generated.size()[1] - 1] == generated[:, generated.size()[1] - 2]):
                break

    # convert the generated ids to text
    generated = [tokenizer.decode(gen_ids.cpu().numpy()).replace('<|endoftext|>', '').strip() for gen_ids in generated]

    return generated
