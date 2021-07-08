# NLP-Seminar project - extended MeLIME

## MeLIME Features 

Extended MeLIME can be applied to any ML model.
You can generate explanations using Basic Statistical Measurements model.

You can also easily implement your own. 

To create the meaningful feature space for your ML model, it is available for three strategies:

1. Word2VecGen: where a word2vec is used to represent tokens. 
1. Word2VecGen with GloVe: where the word2vec uses GloVe embeddings, pretrained on Wikipedia. 
1. Word2VecGen with GloVe and ELMo: where the word2vec uses GloVe, pretrained on Wikipedia, and ELMo embeddings. 

Again, you can implement your generator as well. 


## How to install using python virtual environment

```
git clone https://github.com/ortalcohen94/NLP-Seminar.git
cd melime
python3.7 -m venv venv
source venv/bin/activate
python setup.py install 
```

## Examples

The examples are on experiments folder and can be run using jupyter notebook.  

```
pip install jupyter
jupyter notebook
```

then open one of the experiment files.


### Regression Problem: Toy Model - Spiral Length
Data generated from a toy model- Length of Spiral. The color scale corresponds to the predictedvalues of a MPL model. The red star represents a data point <img src="https://render.githubusercontent.com/render/math?math=x*=(0,8)">, and the gray points the sampled data from KDEGen.Explanations produced for an instance  <img src="https://render.githubusercontent.com/render/math?math=x*"> using MeLIME methodology with KDEGen and a linear model as local model.

<div style="text-align: center">
<img src="doc/images/figure_spiral.png" alt="drawing" width="800"/>
</div>

Go to this [example](https://tiagobotari.github.io/melime/doc/ex_lime_tabular_regression_kde_spiral.html).

### Classification Problem: Iris Dataset

Local explanation for an instance <img src="https://render.githubusercontent.com/render/math?math=x*=(6.0,3.0,5.0,1.5)"> produced by an RF model trained on the Iris dataset. MeLIME using KDEPCAGen and a linear model as local model; MeLIME using KDEPCAGen and a decision tree as local model.

<img src="doc/images/iris_importances.png" alt="drawing" width="500"/>

Go to this [example](https://tiagobotari.github.io/melime/doc/ex_lime_tabular_classification_density_kde_pca_kpca_iris.html).

###  Classification of Images: MNIST Dataset

 MeLIME local explanations produced for a CNN model trained over the MNIST dataset. An image (numberfive) and the produced explanations with counterfactual examples  for  the  top  three  predictions: (a) original image; (b) number five; (c) number nine; and (d) number eight.

<img src="doc/images/cnn_MNIST.png" alt="drawing" width="600" class="center"/>

Go to this [example](https://tiagobotari.github.io/melime/doc/ex_image_vae_mnist.html).

### Sentiment Analysis Texts: Movie-Review

A local explanation was produced using MeLIME for a Naive Bayes classifier model trained on sentences frommovie-reviews of the Rotten Tomatoes website (Website; Pang and Lee 2005). Words/positions showed as reddish color, if changed, are likely to turn the sentence negative while bluish color to become the sentence positive.


<img src="doc/images/explanation_text.png" alt="drawing" width="800"/>

Table:  The original phrase to be explained; Favorable and unfavorable phrases are gener-ated for predicting the sentence as positive.

<img src="https://github.com/tiagobotari/melime/blob/gh-pages/doc/text_examples.png" alt="drawing" width="800"/>

Go to this [example](https://tiagobotari.github.io/melime/doc/ex_word2vec_rotten_tomatos.html).

### Running the Innvestigate experiment

MeLIME comparation with other methodologies for a CNN model trained on MNIST dataset:

![comparation](doc/images/MNIST_comparing.png)

The experiment in the file `melime/experiments/ex_mnist_cnn_innvestigate.py` needs the [`innvestigate`](https://github.com/albermax/innvestigate) framework.
To install it, run the following command: 

```bash
> git submodule update --init
```
This will download the module to the directory `submodules/innvestigate`.

For running this specific experiment, you may need to install tensorflow 1.15 and keras 2.3.1:

```
pip install tensorflow==1.15
pip install keras==2.3.1
```

Afterwards, the script can be run from the root directory: 

```bash
> python melime/experiments/ex_mnist_cnn_innvestigate.py
```

This will plot the result using `matplotlib`.
If you want to output a pdf, then prepend the command as follows:

```bash
> PLOTFILENAME="output.pdf" python melime/experiments/ex_mnist_cnn_innvestigate.py
```