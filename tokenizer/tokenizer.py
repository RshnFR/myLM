class BPETokenizer:
    def __init__(self):
        self.vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.', ',', '?', '!', ':', ';', '-', '_', '+', '=', '(', ')', '{', '}', '[', ']', '<', '>', '/', '\\']
        self.vocab_size = len(self.vocab)

    @staticmethod
    def get_stats(ids):
        # Return a dictionnary with the number of times each pair of consecutive
        counts = {}
        for i in zip(ids, ids[1:]):
            counts[i] = counts.get(i, 0) + 1
        return counts

    def bpe_train(self, text, n):
        # Return a dictionnary of encoded pairs
        pairs = []
        v = 0
        for i in text:
            pairs.append(i)
            v += 1
        text = list(text)
        for i in range(n):
            stats = self.get_stats(text)
            if not stats:
                break
            p = max(stats, key=stats.get)
            pairs.append(p[0] + p[1])
            new_text = []
            skip = False
            for i in range(len(text)-1):
                if skip:
                    skip = False
                    continue
                if text[i] == p[0] and text[i+1] == p[1]:
                    new_text.append(p[0] + p[1])
                    skip = True
                else:
                    new_text.append(text[i])
            if not skip:
                new_text.append(text[-1])
            text = new_text
        self.vocab = self.vocab + pairs
        self.vocab = list(dict.fromkeys(self.vocab))
        self.vocab_size = len(pairs)

    def encode(self, text):
    # Return the encoded text
        text = list(text)
        for i in self.vocab:
            new_text = []
            skip = False
            for j in range(len(text)-1):
                if skip:
                    skip = False
                    continue
                if text[j] + text[j+1] == i:
                    new_text.append(i)
                    skip = True
                else:
                    new_text.append(text[j])
            if not skip:
                new_text.append(text[-1])
            text = new_text
        for i in range(len(text)):
            text[i] = self.vocab.index(text[i])
        return text
    
    def decode(self,text):
        # Return the decoded text
        for i in range(len(text)):
            text[i] = self.vocab[text[i]]
        return "".join(text)
    

if __name__ == '__main__':
    bpe = BPETokenizer()
    #long text
    text = "I want to go to the beach. There for I will go to the beach. I will go to the beach and I will have fun."
    bpe.bpe_train(text, 12)
    print(bpe.vocab)
    print(bpe.encode("beach"))
    print(bpe.decode(bpe.encode("beach")))
