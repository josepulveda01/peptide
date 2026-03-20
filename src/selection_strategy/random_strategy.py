import random

class RandomStrategy:
    def select(self, model, candidates, batch_size, encoding_method=None):
        return random.sample(candidates, batch_size)
    
