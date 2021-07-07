import pandas as pd
from os.path import isfile, join
from interpretable_local_models.statistics_model_nli import StatisticsLocalModelNLI
from gen_models.word2vec_gen import Word2VecGen
from MeLime.model import MeLimeModel

def calc_f1_esnli(predict_label, clf, transform_func, y_p_explain, tokenizer, encoder, x_train, RADIUS, BATCH_SIZE, EPSILON, SIGMA, MAX_ITERS, 
    num_instance_to_sample = None,should_send_sentence = False, print_every = 10):
    # Getting gold explanations:
    df = pd.read_csv(join("data/eSNLI", 'esnli_test.csv'))
    def get_sentence_explanation(df, num):
        x_explain = []
        for i in range(len(df['Sentence' + str(num) + '_Highlighted_1'])):
            res = None
            for t in ['1', '2', '3']:
                x = df['Sentence' + str(num) + '_Highlighted_'+t][i]
                if x == '{}':
                    res = set()
                else:
                    if res == None:
                        res = set(map(int, x.lower().split(',')))
                        continue
                    res = res.intersection(set(map(int, x.lower().split(','))))
            x_explain.append(res)
        return x_explain

    premise_explanations = get_sentence_explanation(df, 1)
    hypothesis_explanations = get_sentence_explanation(df, 2)
    threshold = 0.3
    num_samples = 0
    tp = 0
    fp = 0
    fn = 0

    for i, (premise_explanation, hypothesis_explanation) in enumerate(zip(premise_explanations, hypothesis_explanations)):
        premise = df['Sentence1'][i]
        hypothesis = df['Sentence2'][i]
        x_explain = premise + ' * ' + hypothesis
        x_explain = x_explain.lower()
        label = df['gold_label'][i]
        if(predict_label(x_explain, clf) == label):
    #         print("Computing")
    #         print(x_explain)
            id_seperator = len(tokenizer(premise))
            #print(id_seperator)
            explainer_model = StatisticsLocalModelNLI(y_p_explain, len(tokenizer(x_explain)), tokenizer, id_seperator)
            generator = Word2VecGen(encoder = encoder, corpus = x_train, radius = RADIUS, tokenizer = tokenizer,
                        tokens_not_to_sample = ['*', '.', 'a'], should_send_sentence = should_send_sentence)
            model = MeLimeModel(black_box_model = clf,gen_model =generator, batch_size = BATCH_SIZE, epsilon_c = EPSILON, 
                        sigma = SIGMA, explainer_model = explainer_model, transform_func = transform_func, 
                        max_iters = MAX_ITERS, tokenizer = tokenizer)
            res, sentences_with_probs = model.forward(x_explain, False)
            curr_hypothesis_explanation = set()
            curr_premise_explanation = set()
            did_pass_premise = False
            curr_premise_explanation = set([i for i, (word, prob) in enumerate(res) if (prob >= threshold and i < id_seperator)])
            curr_hypothesis_explanation = set([i for i, (word, prob) in enumerate(res) if (prob >= threshold and i > id_seperator)])
            tp += len(curr_premise_explanation.intersection(premise_explanations[i]))
            tp += len(curr_hypothesis_explanation.intersection(hypothesis_explanations[i]))
            fp += len(curr_hypothesis_explanation.difference(hypothesis_explanations[i]))
            fp += len(curr_premise_explanation.difference(premise_explanations[i]))
            fn += len(hypothesis_explanations[i].difference(curr_hypothesis_explanation))
            fn += len(premise_explanations[i].difference(curr_premise_explanation))
            encoder.clear_cache()
            num_samples += 1
            if (num_instance_to_sample != None and num_samples == num_instance_to_sample):
                break
            if (num_samples % print_every == 0):
                print("Done with sample number ", num_samples)
    F1 = 0 if (tp + 0.5 *(tp + fn)) == 0 else tp / (tp + 0.5 *(tp + fn))
    return F1