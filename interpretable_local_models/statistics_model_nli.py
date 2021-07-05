from interpretable_local_models.abstract_model import LocalModel
import numpy as np
import copy
from matplotlib import pyplot as plt
import seaborn as sns

class StatisticsLocalModelNLI(LocalModel):

    def __init__(self, y_p_explain, sentence_len, tokenizer, seperator_token_id):
        self.values_premise = {key : list() for key in range(seperator_token_id)}
        self.values_hypothesis = {key : list() for key in range( seperator_token_id + 1, sentence_len+1)}
        self.y_p_explain = y_p_explain
        self.tokenizer = tokenizer
        self.seperator_token_id = seperator_token_id

    def train(self, batch):
        for data, label in batch:
            if (self.seperator_token_id < data):
                self.values_hypothesis[data].append(label)
            else:
                self.values_premise[data].append(label)
        
        #Always returns 0 error
        return 0.0

    def explain(self, x: str):
        importances = list(self.measure_importances())
        x_tokenized = self.tokenizer(x)#x.split()#nltk.word_tokenize(x)
        return list(zip(x_tokenized, importances))

    def measure_importances_single_sentence(self, values_sentence):
        mean = {}
        for key, values in values_sentence.items():
            if len(values) == 0:
                mean[key] = 0
            else:
                center_values = np.asarray(values) - self.y_p_explain
                curr_mean = np.mean(center_values) 
                mean[key] = curr_mean
        return np.array(list(mean.values())) / np.max(abs(np.array(list(mean.values()))))

    def measure_importances(self):
        premise = self.measure_importances_single_sentence(self.values_premise)
        hypothesis = self.measure_importances_single_sentence(self.values_hypothesis)
        # print(premise)
        # print(hypothesis)
        return np.concatenate((premise, hypothesis))

    def copy_training_set(self):
        self.values_premise = copy.deepcopy(self.values_premise)
        self.values_hypothesis = copy.deepcopy(self.values_hypothesis)

    def get_explanation(self):
        return self.measure_importances()


    #Plotting explanations

    @staticmethod
    def plot_sentence_heatmap(res, fig_size = (23, 10)):
        values = [[val for _, val in res]]
        annot = np.asarray([[name for name, _ in res]])
        sns.set(rc={'figure.figsize':fig_size})
        g = sns.heatmap(np.array(values), cmap = plt.get_cmap("bwr_r", 256), vmin = -1, vmax = 1, annot = annot, 
                    fmt = "", square = True, cbar_kws={"shrink": 0.3})

    @staticmethod
    def label_bar(rects, ax, labels=None, offset_y=0.4):
        colors = ["blue", "orange"]
        N = len(rects)
        if N > 28:
            font_size = 10
        else:
            font_size = 14
        for i in range(N):
            rect = rects[i]
            width = rect.get_width()
            if width != 0.0:
                text_width = "{:3.2f}".format(width)
            else:
                text_width = ""
            x = rect.get_width() / 2.0
            if abs(x) <= 0.06:
                x = x / abs(x) * 0.10

            y = (rect.get_y() + rect.get_height() / 2) - 0.225
            xy = (x, y)

            ax.annotate(
                text_width,
                xy=xy,
                xytext=(0, -1),  # 3 points vertical offset
                textcoords="offset points",
                # ha="center",
                va="bottom",
                size=font_size,
                color="black",
                horizontalalignment="center",
            )
            if rect.get_width() > 0:
                aling_text = "right"
                off_setx = -3
            else:
                aling_text = "left"
                off_setx = +3
            if labels is not None:
                text = labels[i]
                ax.annotate(
                    text,
                    xy=(rect.get_x(), y),
                    xytext=(off_setx, -1),  # 3 points vertical offset
                    textcoords="offset points",
                    horizontalalignment=aling_text,
                    verticalalignment="bottom",
                    size=font_size,
                )

    @staticmethod
    def plot_explaination(explanations):
        fig, ax = plt.subplots(figsize=(8, 9))
        ax.set_title('Importance', fontsize=25)
        colors = ["tab:blue" if importance > 0 else "tab:red" for feature, importance in explanations]
        pos = np.arange(len(explanations))
        names = [feature for feature, importance in explanations]
        vals = [importance for feature, importance in explanations]
        rects2 = ax.barh(pos, vals, align="center", alpha=0.5, color=colors)


        StatisticsLocalModelNLI.label_bar(rects2, ax, labels=names)
        ax.axvline(0, color="black", lw=2)

        x_lim = np.max(np.abs(vals[:]))
        ax.set_xlim(-x_lim, x_lim)
        y_lim = np.array(ax.get_ylim())
        ax.set_ylim(y_lim + np.array([0, 0.8]))

        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.annotate("  Positive ", xy=(0, y_lim[1] + 0.1), size=16, color="tab:blue", ha="left")
        ax.annotate("  Negative ", xy=(0, y_lim[1] + 0.1), size=16, color="tab:red", ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        plt.show()
        return ax
