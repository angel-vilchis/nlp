from torch import nn 
from torchcrf import CRF
from task_helpers.tags import SPAN_TAGS, id2tag

class CRFModule(nn.Module):
    def __init__(self, num_tags, batch_first=True):
        super(CRFModule, self).__init__()
        self.crf = CRF(num_tags, batch_first)
        self.init_smart_start_transitions()
        self.init_smart_end_transitions()
        self.init_smart_transitions()
        
    def get_transitions(self):
        return self.crf.transitions.detach()
    
    def get_start_transitions(self):
        return self.crf.start_transitions.detach()
    
    def get_end_transitiosn(self):
        return self.crf.end_transitions.detach()
    
    def init_smart_start_transitions(self):
        start_transitions = self.crf.start_transitions.detach()
        for i in range(len(SPAN_TAGS)):
            start_i = i 
            continue_i = i + len(SPAN_TAGS)
            end_i = i + 2*len(SPAN_TAGS)
            alone_i = i + 3*len(SPAN_TAGS)
            # Can start with any start tag
            
            # Can't start on continue tags
            start_transitions[continue_i].requires_grad = False
            start_transitions[continue_i] = -10000.00
            
            # Can't start on end tags
            start_transitions[end_i].requires_grad = False
            start_transitions[end_i] = -10000.00
            
            # Can start on any alone tags
        self.crf.start_transitions = nn.Parameter(start_transitions)
    
    def init_smart_end_transitions(self):
        end_transitions = self.crf.end_transitions.detach()
        for i in range(len(SPAN_TAGS)):
            start_i = i 
            continue_i = i + len(SPAN_TAGS)
            end_i = i + 2*len(SPAN_TAGS)
            alone_i = i + 3*len(SPAN_TAGS)
            
            # Can't end on start tags
            end_transitions[start_i].requires_grad = False
            end_transitions[start_i] = -10000.00
            
            # Can't end on continue tags
            end_transitions[continue_i].requires_grad = False
            end_transitions[continue_i] = -10000.00
            
            # Can end on any end tags
            
            # Can end on any alone tags
        self.crf.end_transitions = nn.Parameter(end_transitions)
        
    def init_smart_transitions(self):
        transitions = self.crf.transitions.detach()
        for i in range(len(SPAN_TAGS)):
            start_i = i 
            continue_i = i + len(SPAN_TAGS)
            end_i = i + 2*len(SPAN_TAGS)
            alone_i = i + 3*len(SPAN_TAGS)
            for j in range(len(SPAN_TAGS)):
                start_j = j 
                continue_j = j + len(SPAN_TAGS)
                end_j = j + 2*len(SPAN_TAGS)
                alone_j = j + 3*len(SPAN_TAGS)
            
                # Start tags can't transition to other start tags
                transitions[start_i, start_j].requires_grad = False
                transitions[start_i, start_j] = -10000.0 
                
                # Start tags can't transition to continue tags besides itself
                if i != j:
                    transitions[start_i, continue_j].requires_grad = False
                    transitions[start_i, continue_j] = -10000.0
                
                # Start tags can't transition to end tags besides itself
                if i != j:
                    transitions[start_i, end_j].requires_grad = False
                    transitions[start_i, end_j] = -10000.0
                
                # Start tags can't transition to alone tags
                transitions[start_i, alone_j].requires_grad = False
                transitions[start_i, alone_j] = -10000.0
                
                # Continue tags can't transition to start tags
                transitions[continue_i, start_j].requires_grad = False
                transitions[continue_i, start_j] = -10000.0
                
                # Continue tags can't transition to other continue tags besides itself
                if i != j:
                    transitions[continue_i, continue_j].requires_grad = False
                    transitions[continue_i, continue_j] = -10000.0
                    
                # Continue tags can't transition to end tags besides itself
                if i != j:
                    transitions[continue_i, end_j].requires_grad = False
                    transitions[continue_i, end_j] = -10000.0
                
                # Continue tags can't transition to alone tags
                transitions[continue_i, alone_j].requires_grad = False
                transitions[continue_i, alone_j] = -10000.0
                
                # End tags can transition to any start tag
                
                # End tags can't transition to continue tags
                transitions[end_i, continue_j].requires_grad = False
                transitions[end_i, continue_j] = -10000.0
                
                # End tags can't transition to other end tags
                transitions[end_i, end_j].requires_grad = False
                transitions[end_i, end_j] = -10000.0
                
                # End tags can transition to any alone tag
                
                # Alone tags can transition to any start tag
                
                # Alone tags can't transition to continue tags
                transitions[alone_i, continue_j].requires_grad = False
                transitions[alone_i, continue_j] = -10000.0
                
                # Alone tags can't transition to end tags
                transitions[alone_i, end_j].requires_grad = False
                transitions[alone_i, end_j] = -10000.0
                
                # Alone tags can transition to any alone tag
        self.crf.transitions = nn.Parameter(transitions)
                
    def loss(self, *args, **kwargs):
        return -self.crf(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        preds = self.crf.decode(*args, **kwargs)
        for i, instance in enumerate(preds):
            for j, tag_id in enumerate(instance):
                preds[i][j] = id2tag(tag_id)
        return preds