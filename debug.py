# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#import torch


# %%
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print(device)


# %%
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np

from data.data_creator import data_create


# %%
data, labels = data_create()

print(data[0])
print(len(data))


# %%
x_train_all, x_test, y_train_all, y_test = train_test_split(
    data, labels, test_size=.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_all, y_train_all, test_size=.1, random_state=42)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)


# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vect_text = TfidfVectorizer(use_idf = False)
x_vec_train = vect_text.fit_transform(x_train)

clf = MultinomialNB().fit(x_vec_train, y_train)


# %%
preds = clf.predict(vect_text.transform(x_val))


# %%
print('Val accuracy', metrics.accuracy_score(y_val, preds))


# %%
x_explain = "the movie's thesis -- elegant technology for the masses -- is surprisingly refreshing ."
print('x to explain: ',x_explain)
print('Predicted class: ', clf.predict(vect_text.transform([x_explain]))[0])
print('True class: ', 1)
print('Predict probablilities: ', clf.predict_proba(vect_text.transform([x_explain]))[0])


# %%
import nltk
from torch.utils.data import DataLoader
def tokenizer(x):
    return x.split()
dl_train = [tokenizer(x) for x in x_train]
# for x in dl_train:
#     print(x)
#     break


# %%
from gen_models.word2vec_gen import Word2VecGen, Word2VecEncoder
generator = Word2VecGen(encoder = Word2VecEncoder(dl_train), corpus = x_train, radius = 10000, tokenizer = tokenizer)


# %%
#Debug proposals
# sentence = "this film seems thirsty for reflection , itself taking on adolescent qualities ."
# generator.sample_sentence(sentence)


# %%
from interpretable_local_models.statistics_model import StatisticsLocalModel
tokenized_x_explain = x_explain.split()
y_p_explain = max(clf.predict_proba(vect_text.transform([x_explain]))[0])
explainer_model = StatisticsLocalModel(y_p_explain, len(tokenized_x_explain), tokenizer)
print(tokenized_x_explain)


# %%
from MeLime.model import MeLimeModel

def transform_func(x):
    return vect_text.transform([x])

model = MeLimeModel(black_box_model = clf,gen_model =generator, batch_size = 50, epsilon_c = 0.1, 
                    sigma = 0.001, explainer_model = explainer_model, transform_func = transform_func, max_iters = 100, 
                   tokenizer = tokenizer)
res = model.forward(x_explain)


# %%
print(res)


# %%
ax = StatisticsLocalModel.plot_explaination(res)


# %%



