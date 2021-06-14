from interpretable_local_models.abstract_model import LocalModel
from gen_models.abstract_model import GenModel
import copy
import numpy as np
import math

def default_transform_model (x):
    return x

class MeLimeModel ():
    def __init__(self, black_box_model, gen_model : GenModel, batch_size, 
    epsilon_c, sigma, explainer_model : LocalModel, tokenizer, transform_func = default_transform_model, 
    max_iters = 1000, print_every = 3):
        super(MeLimeModel, self).__init__()
        self.black_box_model = black_box_model
        self.transform_func = transform_func
        self.gen_model = gen_model
        self.batch_size = batch_size
        self.epsilon_c = epsilon_c
        self.sigma = sigma
        self.explainer_model = explainer_model
        self.max_iters = max_iters
        self.print_every = print_every
        self.tokenizer = tokenizer
    
    def get_explanation(self, model: LocalModel):
        return model.get_explanation()

    def get_delta (self, x, g_2: LocalModel):
        new_D = []
        positions=[]
        probas = []
        true_label = self.black_box_model.predict(self.transform_func(x))[0]
        for i in range(self.batch_size):
                pos, x_i = self.gen_model.sample_instance(x)
                listToStr = ' '.join([str(elem) for elem in x_i])
                transformed_x_i = self.transform_func(listToStr)
                label = self.black_box_model.predict(transformed_x_i)[0]
                #print(label)
                positions.append(pos)
                curr_prob = self.black_box_model.predict_proba(transformed_x_i)[0][true_label]
                probas.append(curr_prob)
                new_D.append((pos, curr_prob))
        g_2.train(new_D)
        alpha_one = self.get_explanation(self.explainer_model)
        alpha_two = self.get_explanation(g_2)
        return np.linalg.norm(alpha_one - alpha_two, ord = 1) / alpha_one.shape[0]



    def forward(self, x):
        D = []
        positions = []
        probas = []
        epsilon, delta = 0, math.inf
        true_label = self.black_box_model.predict(self.transform_func(x))[0]
        count = 0
        sentences_with_probs = []
        while ((epsilon >= self.epsilon_c or delta >= self.sigma) and count <= self.max_iters):
            for i in range(self.batch_size):
                pos, x_i = self.gen_model.sample_instance(x)
                listToStr = ' '.join([str(elem) for elem in x_i])
                transformed_x_i = self.transform_func(listToStr)
                label = self.black_box_model.predict(transformed_x_i)[0]
                #print(label)
                positions.append(pos)
                curr_prob = self.black_box_model.predict_proba(transformed_x_i)[0][true_label]
                probas.append(curr_prob)
                D.append((pos, curr_prob))
                sentences_with_probs.append((listToStr, curr_prob))
            count += 1
            epsilon = self.explainer_model.train(D)
            g_2 = copy.deepcopy(self.explainer_model)
            g_2.copy_training_set()
            delta =  self.get_delta(x, g_2)
            if (count % self.print_every == 0):
                print("Iteration number: ", count)
                print("Delta: ", delta)
        return self.explainer_model.explain(x), sentences_with_probs

