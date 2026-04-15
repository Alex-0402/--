import torch
import numpy as np

def modify_inference_for_constrained_decoding(vocab, sample, args, device):
    # This is a snippet demonstrating how to inject the logic into inference.py
    
    # 0. Build Allowed Fragment Trie & State Setup
    res_map = vocab._residue_to_fragments_map
    fragment_residue_idx = sample.get('fragment_residue_idx', None)
    frag_levels = sample.get('fragment_levels', None)
    
    if hasattr(fragment_residue_idx, 'cpu'):
        fragment_residue_idx_cpu = fragment_residue_idx.cpu().numpy().flatten()
        frag_levels_cpu = frag_levels.cpu().numpy().flatten()
    else:
        fragment_residue_idx_cpu = fragment_residue_idx.flatten()
        frag_levels_cpu = frag_levels.flatten()
        
    num_residues = fragment_residue_idx_cpu[-1] + 1
    res_frag_lengths = np.zeros(num_residues, dtype=int)
    for r_idx in fragment_residue_idx_cpu:
        res_frag_lengths[r_idx] += 1
        
    # Initialize active candidates per residue
    # Only keep amino acids that have the exact same fragment length as the Ground Truth mask
    active_candidates = [] 
    for r_len in res_frag_lengths:
        valid_aas = {aa for aa, frags in res_map.items() if len(frags) == r_len}
        active_candidates.append(valid_aas)

    # In the inference loop:
    # Before computing Softmax:
    # frag_probs = torch.softmax(frag_logits, dim=-1)
    
    # ====== NEW CONSTRAINT LOGIC ======
    '''
    # Create mask of allowed tokens
    for m_idx in masked_indices:
        r_idx = fragment_residue_idx_cpu[m_idx]
        level = frag_levels_cpu[m_idx]
        
        allowed_tokens = set()
        for aa in active_candidates[r_idx]:
            # Get the exact fragment token required at this level for this candidate AA
            frag_name = res_map[aa][level] 
            allowed_tokens.add(vocab.token_to_idx[frag_name])
            
        # Optional Dead-end backtracking check
        # legal_prob_sum = frag_probs[0, m_idx, list(allowed_tokens)].sum().item()
        # if legal_prob_sum < 0.02: 
        #      # Mark for rollback...
        
        # Prune logits!
        all_tokens_set = set(range(vocab.get_vocab_size()))
        forbidden_tokens = list(all_tokens_set - allowed_tokens)
        frag_logits[0, m_idx, forbidden_tokens] = -float('inf')
    
    # Re-compute softmax
    frag_probs = torch.softmax(frag_logits, dim=-1)
    predicted_tokens = torch.argmax(frag_logits, dim=-1)[0]
    '''
    
    # After commit_indices are chosen:
    '''
    for c_idx in commit_indices:
        r_idx = fragment_residue_idx_cpu[c_idx]
        level = frag_levels_cpu[c_idx]
        pred_token_idx = predicted_tokens[c_idx].item()
        pred_frag_name = vocab.idx_to_token[pred_token_idx]
        
        # Narrow down the candidate set! (Convergence)
        active_candidates[r_idx] = {
            aa for aa in active_candidates[r_idx] 
            if res_map[aa][level] == pred_frag_name
        }
    '''
