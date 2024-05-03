import operator
import torch
from queue import PriorityQueue
from config import EOS_TOKEN, SOS_TOKEN, DEVICE, MAX_LEN_SEQ_B

class BeamSearchNode(object):
    '''
        Node that holds its value as well as necessary information to expand node
    '''
    def __init__(self, hidden_state, prev_node, token_id, token_prob, prob, length):
        self.hidden_state = hidden_state
        self.prev_node = prev_node
        self.token_id = token_id
        self.token_prob = token_prob
        self.prob = prob
        self.length = length - 1 if token_id == EOS_TOKEN else length

    def __lt__(self, obj):
        if not isinstance(obj, BeamSearchNode):
            return False
        return self.prob < obj.prob

    def eval(self, alpha=1.0, trans=False):
        reward = 0 # can add function to shape reward
        # if trans and self.length:
        #     return -(self.prob * (self.length**0.7))
        return -self.prob


def beam_decode(model, encoder_outputs, decoder_hiddens, num_return_sequences, beam_width, sampler, trans=False):
        '''
            Beam searches for model outputs for either Transformer or RNN
            
            decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
            encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
            return: decoded_batch, decoded_probs
        ''' 
        B, T, H = encoder_outputs.size()
        decoded_batch = [] # final outputs
        decoded_probs = [] # probabilities

        for idx in range(B): # Decodes instance by instance 
            decoder_input = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=DEVICE) # Start with the start of the sentence token, only one since instance by instance

            endnodes = []

            if not trans:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0) # get hidden for batch instance 
            else:
                decoder_hidden = decoder_input # hidden for transformer are previous inputs
            encoder_output = encoder_outputs[idx, :, :].unsqueeze(0) # get encoder outputs for batch instance 

            node = BeamSearchNode(hidden_state=decoder_hidden, prev_node=None, token_id=decoder_input.squeeze(), token_prob=1.0, prob=1.0, length=0) # starting node
            nodes = PriorityQueue() # empty queue containing (score, node) tuples

            nodes.put((node.eval(trans=trans), node)) #  add node
            qsize = 1

            while True: # beam search 
                if qsize > (500 * beam_width): break # give up when search takes too long

                score, n = nodes.get() # get best node (lowest value)
                decoder_input = n.token_id.view(1, -1) # token from best node
                decoder_hidden = n.hidden_state # hidden state from best node 

                if n.token_id.item() == EOS_TOKEN and n.prev_node != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= num_return_sequences:
                        break
                    else:
                        continue

                if not trans:
                    decoder_output, decoder_hidden, attn_weights = model.forward_step(decoder_input, decoder_hidden, encoder_output) # decode for one step using model
                    tokens, probs = sampler.sample(logits=decoder_output, num_samples=beam_width)
                else:
                    # get hidden, tokens, probs from transformer
                    out_embed = model.get_output_embed_training(decoder_hidden, decoder_hidden.shape[1])
                    output_mask = model.transformer.generate_square_subsequent_mask(decoder_hidden.shape[1]).to(DEVICE)
                    out = model.transformer.decoder(out_embed, 
                                                encoder_output, 
                                                tgt_mask=output_mask)
                    
                    out = model.linear(out[:, [-1], :])
                    
                    # Sample next token
                    tokens, probs = sampler.sample(logits=out, num_samples=beam_width)

                for new_k in range(beam_width):
                    token_id = tokens.squeeze()[new_k] # token 
                    prob = probs.squeeze()[new_k]
                    length = n.length + 1 # one step longer

                    # Create new node and put in queue
                    if trans:
                        node = BeamSearchNode(hidden_state=torch.cat((decoder_hidden[0], token_id.unsqueeze(-1))).unsqueeze(0), prev_node=n, token_id=token_id, token_prob=prob, prob=n.prob*prob, length=length)
                    else:
                        node = BeamSearchNode(hidden_state=decoder_hidden, prev_node=n, token_id=token_id, token_prob=prob, prob=n.prob*prob, length=length)
                    score = node.eval(trans=trans) 
                    nodes.put((score, node))
                    qsize += 1

                qsize -= 1 # curr node expanded
        
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(num_return_sequences)] # Get top k leafs if no nodes ended

            utterances = []
            token_probs = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)): # sort endnodes by score
                utterance = []
                token_prob = []
                utterance.append(n.token_id)
                token_prob.append(n.token_prob)

                while n.prev_node != None: # Back trace to get full output
                    n = n.prev_node
                    utterance.append(n.token_id)
                    token_prob.append(n.token_prob)
                

                utterance = utterance[::-1] # reverse since output is backwards
                token_prob = token_prob[::-1]
                token_probs.append(token_prob)
                utterances.append(utterance)

            decoded_batch.append(utterances) # B x TOPK
            decoded_probs.append(token_probs)
        
        for i in range(B):
            decoded_batch[i] += [[SOS_TOKEN] + [0 for _ in range(MAX_LEN_SEQ_B)] for _ in range(num_return_sequences-len(decoded_batch[i]))]
            decoded_probs[i] += [[0.0 for _ in range(MAX_LEN_SEQ_B+1)] for _ in range(num_return_sequences-len(decoded_probs[i]))]
            for j in range(len(decoded_batch[i])):
                decoded_batch[i][j] = decoded_batch[i][j][1:] # removing SOS
                decoded_probs[i][j] = decoded_probs[i][j][1:]
                decoded_batch[i][j] += [0 for _ in range(MAX_LEN_SEQ_B-len(decoded_batch[i][j]))] # padding
                decoded_probs[i][j] += [1.0 for _ in range(MAX_LEN_SEQ_B-len(decoded_probs[i][j]))] # padding

                
        return torch.tensor(decoded_batch, device=DEVICE), None, torch.tensor(decoded_probs, device=DEVICE)


'''
Rough draft GPU Beam Search
                curr_log_probs = (decoder_output + k_running_log_probs).view(batch_size, -1) # Batch x VOCAB or Batch x (K*VOCAB)
                log_probs, topi = curr_log_probs.topk(num_return_sequences) # Batch x K
                
                # if num_return_sequences > 1:
                k_running_log_probs = log_probs # update running probs for new k seqs
                log_probs, topi = curr_log_probs.topk(k) # Batch x K, Batch x K
                k_running_log_probs = log_probs.reshape(batch_size, k, 1) # update running probs for new k seqs

                # Update running seqs
                beam_branch_indices = topi // self.vocab_size if i > 0 else torch.arange(num_return_sequences)
                beam_new_tokens = topi % self.vocab_size
                beam_branch_indices = (topi // self.vocab_count) if i > 0 else torch.tensor([list(range(k)) for _ in range(batch_size)]) # index // vocab size to get correct beams
                beams2expand = k_running_seqs[torch.arange(batch_size)[:, None], beam_branch_indices]
                beam_new_tokens = (topi % self.vocab_count).reshape(batch_size, k, 1) # index % vocab size to get correct token indices
                k_running_seqs = torch.concatenate((beams2expand, beam_new_tokens), axis=-1)
                
                decoder_input = beam_new_tokens.squeeze(-1).detach()  # detach from history as input
                
                print("sizes", k_running_seqs.shape, k_running_seqs[beam_branch_indices].shape, beam_new_tokens.shape)
                k_running_seqs = torch.column_stack((k_running_seqs[beam_branch_indices], beam_new_tokens))

                decoder_input = topi.detach()  # detach from history as input
'''