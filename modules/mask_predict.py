
import torch
import torch.nn.functional as F

class MaskPredict(object):
    
    def __init__(self, decoding_iterations, token_num):
        super().__init__()
        self.iterations = decoding_iterations
        self.token_num = token_num
    
    def generate(self, model, trg_tokens, word_feat, word_mask, points_mask, past_points_mask, past_self, pad_idx, mask_idx):
        """points_emd
        """
        bsz, seq_len = trg_tokens.size()
        pad_mask = ~points_mask
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        iterations = seq_len if self.iterations is None else self.iterations
        
        trg_tokens, token_probs, present_self = self.generate_non_autoregressive(model, trg_tokens, word_feat, word_mask, past_points_mask, past_self)
        
        assign_single_value_byte(trg_tokens, pad_mask, pad_idx)
        assign_single_value_byte(token_probs, pad_mask, 1.0)
        #print("Initialization: ", convert_tokens(tgt_dict, trg_tokens[0]))
        
        for counter in range(1, iterations):
            num_mask = (seq_lens.float() * (1.0 - (counter / iterations))).long()

            assign_single_value_byte(token_probs, pad_mask, 1.0)
            mask_ind = self.select_worst(token_probs, num_mask)
            assign_single_value_long(trg_tokens, mask_ind, mask_idx)
            assign_single_value_byte(trg_tokens, pad_mask, pad_idx)

            #print("Step: ", counter+1)
            #print("Masking: ", convert_tokens(tgt_dict, trg_tokens[0]))
            decoder_out, present_self = model.decoder.forward_fast(
                trg_tokens=trg_tokens, 
                encoder_output=word_feat, 
                src_mask=word_mask, 
                trg_mask=past_points_mask, 
                mask_future=False, 
                window_mask_future=True, 
                window_size=self.token_num, 
                past_self=past_self)
            new_trg_tokens, new_token_probs, all_token_probs = generate_step_with_prob(decoder_out)
            
            assign_multi_value_long(token_probs, mask_ind, new_token_probs)
            assign_single_value_byte(token_probs, pad_mask, 1.0)
            
            assign_multi_value_long(trg_tokens, mask_ind, new_trg_tokens)
            assign_single_value_byte(trg_tokens, pad_mask, pad_idx)
            #print("Prediction: ", convert_tokens(tgt_dict, trg_tokens[0]))
        
        lprobs = token_probs.log().sum(-1)
        return trg_tokens, present_self
    
    def generate_non_autoregressive(self, model, points_emd, word_feat, word_mask, past_points_mask, past_self):
        logits, present_self = model.decoder.forward_fast(
                trg_tokens=points_emd, 
                encoder_output=word_feat, 
                src_mask=word_mask, 
                trg_mask=past_points_mask, 
                mask_future=False, 
                window_mask_future=True, 
                window_size=self.token_num, 
                past_self=past_self)

        trg_tokens, token_probs, _ = generate_step_with_prob(logits)
        return trg_tokens, token_probs, present_self

    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)


def duplicate_encoder_out(encoder_out, bsz, beam_size):
    encoder_out['encoder_out'] = encoder_out['encoder_out'].unsqueeze(2).repeat(1, 1, beam_size, 1).view(-1, bsz * beam_size, encoder_out['encoder_out'].size(-1))
    if encoder_out['encoder_padding_mask'] is not None:
        encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].unsqueeze(1).repeat(1, beam_size, 1).view(bsz * beam_size, -1)


def generate_step_with_prob(out):
    probs = F.softmax(out, dim=-1)
    max_probs, idx = probs.max(dim=-1)
    return idx, max_probs, probs


def assign_single_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y


def assign_multi_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y.view(-1)[i.view(-1).nonzero()]


def assign_single_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y


def assign_multi_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y.view(-1)[i.view(-1)]


def convert_tokens(dictionary, tokens):
    return ' '.join([dictionary[token] for token in tokens])