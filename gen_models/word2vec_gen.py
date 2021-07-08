import numpy as np
from gen_models.abstract_model import GenModel
from gensim.models import Word2Vec
import string
from flair.embeddings import ELMoEmbeddings
from flair.data import Sentence
import gensim.downloader
import flair
import torch.nn as nn

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
        if word not in self.model.wv.vocab:
            return None
        res = self.model.wv.most_similar_cosmul(word, topn = radius)
        self.most_similar[word] = res
        return res

    def remove_similar_word(self, word, index):
        if (word not in self.most_similar):
            return
        del self.most_similar[word][index]

    def clear_cache(self):
        self.most_similar = {}


# init embedding
#embedding = ELMoEmbeddings()

class Word2VecELMOEncoder:
    def __init__(self, extended_radius):
        super().__init__()
        
        self.model = gensim.downloader.load('glove-wiki-gigaword-300')
        self.most_similar = {}
        self.embedding = ELMoEmbeddings()
        self.extended_radius = extended_radius

    def similarity(self, idx, original_sentence, radius):
        word = original_sentence[idx].translate(str.maketrans('', '', string.punctuation))
        cos_sim_func = nn.CosineSimilarity(dim = 0)
        if (word in self.most_similar):
            return self.most_similar[word]
        res = self.model.most_similar_cosmul(word, topn = self.extended_radius)
        res = [w for w, _ in res]
        #print(res)
        sentence = Sentence(' '.join(original_sentence), use_tokenizer=False)
        self.embedding.embed(sentence)
        t = sentence[idx].embedding
        similarity_met = {}
        for curr_word in res:
            s = Sentence(curr_word, use_tokenizer = False)
            self.embedding.embed(s)
            cos_sim = cos_sim_func(s[0].embedding, t)
            similarity_met[curr_word] = cos_sim.item()
        res = list(dict(sorted(similarity_met.items(), key=lambda x: x[1], reverse=True)[:radius]).items())
        self.most_similar[word] = res
        return res

    def remove_similar_word(self, word, index):
        if (word not in self.most_similar):
            return
        del self.most_similar[word][index]

    def clear_cache(self):
        self.most_similar = {}
        
        

class Word2VecGloVeEncoder:
    def __init__(self):
        super().__init__()
        
        self.model = gensim.downloader.load('glove-wiki-gigaword-300')
        self.most_similar = {}

    def similarity(self, word, radius):
        if (word in self.most_similar):
            return self.most_similar[word]
        if word not in self.model.wv.vocab:
            return None
        res = self.model.most_similar_cosmul(word, topn = radius)
        self.most_similar[word] = res
        return res

    def remove_similar_word(self, word, index):
        if (word not in self.most_similar):
            return
        del self.most_similar[word][index]

    def clear_cache(self):
        self.most_similar = {}

class SentencesIter():
    def __init__(self, sentences) -> None:
        self.sentences = sentences
    
    def __iter__(self):
        for sentence in self.sentences:
            yield sentence

    
class Word2VecGen(GenModel):

    #Radius should be between 0 and 1
    def __init__(self, encoder: Word2VecEncoder, corpus, radius, tokenizer, tokens_not_to_sample = None, 
                 should_send_sentence = False) -> None:
        super().__init__()
        self.encoder = encoder
        self.corpus = corpus
        self.r = radius
        self.tokenizer = tokenizer
        self.tokens_not_to_sample = tokens_not_to_sample
        self.should_send_sentence = should_send_sentence

    def sample_instance (self, original_sentence):
        original_sentence = self.tokenizer(original_sentence)
        if self.tokens_not_to_sample != None:
            indices = [i for i, token in enumerate(original_sentence) if token not in self.tokens_not_to_sample]
        else:
            indices = range(len(original_sentence))
        idx = np.random.choice(indices, 1)[0]
        while(self.tokens_not_to_sample != None and original_sentence[idx] in self.tokens_not_to_sample):
            idx = np.random.choice(range(len(original_sentence)), 1)[0]
        #print(idx):
        #idx = torch.multinomial(torch.FloatTensor(range(len(original_sentence))), 1).item()
        s_j = original_sentence[idx]
        s_j = s_j.translate(str.maketrans('', '', string.punctuation))
        #print(s_j)
        if self.should_send_sentence:
            similar_words = self.encoder.similarity(idx, original_sentence, self.r)
        else:
            similar_words = self.encoder.similarity(s_j, self.r)
        if(similar_words == None):
            return None, None
        I = [word for word, similarity in similar_words]
        if (len(I) == 0):
            return None, None
        else:
            I_idx = np.random.choice(range(len(I)), 1)[0]#torch.multinomial(torch.FloatTensor(range(len(I))), 1).item()
            self.encoder.remove_similar_word(s_j, I_idx)
            s_k = I[I_idx]
        s_k = s_k.translate(str.maketrans('', '', string.punctuation))
        new_sentence = original_sentence.copy()
        new_sentence[idx] = s_k
        return idx, new_sentence