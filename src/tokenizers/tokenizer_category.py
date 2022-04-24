from nltk.corpus import stopwords

class TokernizerMemeCategory():

    """
    Class for tokenizer text for meme category

    
    """

    def __init__(self):

        """
        vocab -> dictionary with words and their index
        string_punt -> punctuation to eliminate
        """

        self.vocab = {"<pad>": 0}
        self.string_punt ="\'\".-“”–!¡?¿,;:`()[]/<>=+*&^%$#@|~[]"



    def get_vocab(self) -> dict:

        """
        Return vocab dictionary
        """

        return self.vocab

    def tokenize(self, text: str) -> list:

        """
        Tokenize text

        """
        for str_elim in self.string_punt:
            text = text.replace(str_elim, "")

        text_vector = []
        for word in text.split():
            word = word.lower()
            if word not in stopwords.words('spanish'):
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                text_vector.append(self.vocab[word])

        return text_vector
    
