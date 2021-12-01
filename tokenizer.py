class TokenizerMeme():
    
    def __init__(self, vocab):
        
        self.vocab = vocab
        self.string_punt =".-“”–!¡?¿,;:`()[]/<>=+*&^%$#@|~[]"
        
    def get_vocab(self):
        return self.vocab
    
    def tokenize(self, text):
        
        for str_elim in self.string_punt:
            text = text.replace(str_elim, "")
        
        text_vector = []
        for word in text.split():
            word = word.lower()
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
            text_vector.append(self.vocab[word])
            
        return text_vector
    
    def clean_text(self, text):
        
        for str_elim in self.string_punt:
            text = text.replace(str_elim, "")
            
        return text.strip()
        
            