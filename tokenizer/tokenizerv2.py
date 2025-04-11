

class TokenizerV2 :
    """
    Self made tokenizer class
    """
    def __init__(self, vocab_size=10000, min_freq=1):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.vocab = {}
        self.reverse_vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            
        }