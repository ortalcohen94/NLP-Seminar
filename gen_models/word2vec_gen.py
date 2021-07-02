import numpy as np
from gen_models.abstract_model import GenModel
from gensim.models import Word2Vec

#Implemented gensim by using the following article:
#https://rare-technologies.com/word2vec-tutorial/
class Word2VecEncoder():
    def __init__(self, dl_train):
        super().__init__()
        self.model = Word2Vec(SentencesIter(dl_train), min_count=1, sg=0, hs=0, negative=5, workers=1, compute_loss=True)
        self.most_similar = {}

    def similarity(self, word, radius):
        if (word in self.most_similar):
            return self.most_similar[word]
        res = self.model.wv.most_similar_cosmul(word, topn = radius)
        self.most_similar[word] = res
        return res

    def remove_similar_word(self, word, index):
        del self.most_similar[word][index]

class Word2VecBertEncoder:
    def __init__(self, dl_train):
        #TODO: Fine-tuning with dl_train and model's init.
        self.most_similar = dict()
        pass
    
    def similarity(self, word, radius):
        #TODO: should return <radius> most similar words in res.
        res = []
        res = self.model.wv.most_similar_cosmul(word, topn = radius)
        self.most_similar[word] = res
        

    def remove_similar_word(self, word, index):
        print(word)
        del self.most_similar[word][index]

class SentencesIter():
    def __init__(self, sentences) -> None:
        self.sentences = sentences
    
    def __iter__(self):
        for sentence in self.sentences:
            yield sentence

class Word2VecGen(GenModel):

    #Radius should be between 0 and 1
    def __init__(self, encoder: Word2VecEncoder, corpus, radius, tokenizer, tokens_not_to_sample = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.corpus = corpus
        self.r = radius
        self.tokenizer = tokenizer
        self.tokens_not_to_sample = tokens_not_to_sample

    def sample_instance (self, original_sentence):
        original_sentence = self.tokenizer(original_sentence)
        if self.tokens_not_to_sample != None:
            indices = [i for i, token in enumerate(original_sentence) if token not in self.tokens_not_to_sample]
            print(indices)
        else:
            indices = range(len(original_sentence))
        idx = np.random.choice(indices, 1)[0]
        while(self.tokens_not_to_sample != None and original_sentence[idx] in self.tokens_not_to_sample):
            idx = np.random.choice(range(len(original_sentence)), 1)[0]
        #print(idx):
        #idx = torch.multinomial(torch.FloatTensor(range(len(original_sentence))), 1).item()
        s_j = original_sentence[idx]
        I = [word for word, similarity in self.encoder.similarity(s_j, self.r)]
        I_idx = np.random.choice(range(len(I)), 1)[0]#torch.multinomial(torch.FloatTensor(range(len(I))), 1).item()
        self.encoder.remove_similar_word(s_j, I_idx)
        s_k = I[I_idx]
        new_sentence = original_sentence.copy()
        new_sentence[idx] = s_k
        return idx, new_sentence
        
    # def train(self, dl_train, dl_test, num_epochs):
    #     self.trainer.fit(dl_train, dl_test, num_epochs)

        