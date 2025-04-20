from collections import Counter
import pickle
class TokenizerV2 :
    """
    Self made tokenizer class
    """
    def __init__(self):
        self.vocab_size = 0
        self.min_freq = 2
        self.vocab = {}
        self.reverse_vocab = {}
        self.merges = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            "<unk>": 0,
            "</w>": 1
        }   

    def pretokenize(self,text):
        """
        Pretokenize the input text by splitting it into words and punctuation.
        """
        import re
        # Define a regex pattern to match words and punctuation
        pattern = r"\w+|[^\w\s]"
        # Find all matches in the input text
        tokens = re.findall(pattern, text)
        return tokens
    
    def build_vocab(self, all_pre_tokens):
        initial_vocab = set()
        for token in all_pre_tokens:
            initial_vocab.update(list(token))
        initial_vocab.update(self.special_tokens.keys())
        self.vocab = initial_vocab
        self.vocab_size = len(self.vocab)
    
    def get_pair_stats(self,splits, word_freqs):
        """Counts occurrences of adjacent symbol pairs."""
        stats = Counter()
        for word, freq in word_freqs.items():
            symbols = splits[word]
            if len(symbols) < 2:
                continue
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                stats[pair] += freq
        return stats


    def build_tokenizer(self, corpus):
        all_pre_tokens = []
        for sentence in corpus:
            all_pre_tokens.extend(self.pretokenize(sentence))
        self.build_vocab(all_pre_tokens)
        word_freqs = Counter(all_pre_tokens)
        splits = {word: list(word) + ['</w>'] if word.isalnum() else list(word) for word in word_freqs.keys()}
        num_merges = 1000
        merges = {}
        current_splits = splits.copy()
        vocab = self.vocab.copy()
        for i in range(num_merges):
            pair_stats = self.get_pair_stats(current_splits, word_freqs)
            if not pair_stats:
                break
            most_frequent_pair = max(pair_stats, key=pair_stats.get)
            freq = pair_stats[most_frequent_pair]
            if freq < 2:
                break
            new_symbol = ''.join(most_frequent_pair)
            merges[most_frequent_pair] = new_symbol
            vocab.add(new_symbol)
            new_splits = {}
            for word, symbols in current_splits.items():
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == most_frequent_pair:
                        new_symbols.append(new_symbol)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                new_splits[word] = new_symbols
            current_splits = new_splits
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in enumerate(vocab)}
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.vocab_size = len(vocab)
        self.merges = merges
    def save_object(self, filename):
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
    
    def load_object(self,filename):
        with open(filename, 'rb') as f:
            tokenizer = pickle.load(f)
            self.vocab_size = tokenizer.vocab_size
            self.min_freq = tokenizer.min_freq
            self.vocab = tokenizer.vocab
            self.reverse_vocab = tokenizer.reverse_vocab
            self.merges = tokenizer.merges
            self.token_to_id = tokenizer.token_to_id
            self.id_to_token = tokenizer.id_to_token
            self.special_tokens = tokenizer.special_tokens

        
    def encode(self, text):
        """Encodes the text into indices based on the vocabulary dictionary."""
        tokens = self.pretokenize(text)
        encoded = []
        for token in tokens:
            if token.isalnum():
                word_tokens = list(token) + ['</w>']
            else:
                word_tokens = list(token)
                
            while len(word_tokens) > 1:
                pairs = [(word_tokens[i], word_tokens[i+1]) for i in range(len(word_tokens)-1)]
                pair_to_merge = None
                for pair in pairs:
                    if pair in self.merges:
                        pair_to_merge = pair
                        break
                
                if not pair_to_merge:
                    break
                    
                idx = pairs.index(pair_to_merge)
                new_token = self.merges[pair_to_merge]
                word_tokens = word_tokens[:idx] + [new_token] + word_tokens[idx+2:]
            
            for t in word_tokens:
                if t in self.token_to_id:
                    encoded.append(self.token_to_id[t])
                else:
                    encoded.append(self.token_to_id["<unk>"])
                    
        return encoded
    
    def decode(self, encoded):
        """Decodes the indices back into text."""
        print(encoded)
        tokens = [self.id_to_token.get(idx, "<unk>") for idx in encoded]
        decoded = ""
        current_word = ""
        
        for token in tokens:
            if token == '</w>':
                decoded += current_word + " "
                current_word = ""
            elif '</w>' in token:
                parts = token.split('</w>')
                decoded += current_word + parts[0] + " "
                if len(parts) > 1 and parts[1]:
                    current_word = parts[1]
                else:
                    current_word = ""
            else:
                current_word += token
        
        if current_word:
            decoded += current_word
            
        return decoded.strip()
    

if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb")
    corpus = ds['train']['text'][:1000]
    print("Sample Corpus:", corpus[:5])
    tokenizer = TokenizerV2()
    tokenizer.build_tokenizer(corpus)
    print("Encoded:", tokenizer.encode("Hello world!"))
    print("Decoded:", tokenizer.decode(tokenizer.encode("Hello world!")))

    tokenizer.save_object("tokenizer.pkl")
    loaded_tokenizer = TokenizerV2.load_object("tokenizer.pkl")

    print("Loaded Tokenizer Encoded:", loaded_tokenizer.encode("Hello world!"))
    print("Loaded Tokenizer Decoded:", loaded_tokenizer.decode(loaded_tokenizer.encode("Hello world!")))
