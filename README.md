# myLM
Implementation of Language Models.


## 1.Tokenizer
Implementation of a tokenizer using de byte pair encoding method
### 1.1 Tokevenizer V2
Better tokenizer with special tokens.

## 2.Logistic Regression
Implementation of a basic logistic regression model. Used a very simple example.
Reimplemented using scikit-learn
Sentiment Analysis done through Stanford's sst2 and imdb database (see utils.py)
### 2.1 Logistic Regression V2
Implementation of a basic logistic regression model.
Implemented using numpy only
Tested with Stanford's imdb database


## 3.TF-IDF
Implementation of term frequency - inverse document frequency.

## 4.Feedforward Neural Network
Implementation of feedforward neural-network with keras.
Accuracy is pretty low as expected.

## 5.Recurrent Neural Network
Implementation of recurrent neural network.
Accuracy : 0.84

## 6. LSTM
Implementation of a LSTM.
Accuracy : 0.86

### TO-DO:
Redo everything with existing library.
Redo from scratch (ff-nn,rnn,)
Redo with more complex use cases(utf8 bpe tokenizer)

## Dataset Citation

This project makes use of the following datasets. If you use these datasets in your work, please cite the original papers:

-  **Large Movie Review Dataset (IMDB)**  
  Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011).  
  *Learning Word Vectors for Sentiment Analysis.*  
  In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies (pp. 142-150). Portland, Oregon, USA: Association for Computational Linguistics.  
  [Paper Link](http://www.aclweb.org/anthology/P11-1015)

  <details>
    <summary>BibTeX</summary>

    ```bibtex
    @InProceedings{maas-EtAl:2011:ACL-HLT2011,
      author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
      title     = {Learning Word Vectors for Sentiment Analysis},
      booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
      month     = {June},
      year      = {2011},
      address   = {Portland, Oregon, USA},
      publisher = {Association for Computational Linguistics},
      pages     = {142--150},
      url       = {http://www.aclweb.org/anthology/P11-1015}
    }
    ```
  </details>

-  **Stanford Sentiment Treebank (SST)**  
  Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013).  
  *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.*  
  In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing (pp. 1631-1642). Seattle, Washington, USA: Association for Computational Linguistics.  
  [Paper Link](https://www.aclweb.org/anthology/D13-1170)

  <details>
    <summary>BibTeX</summary>

    ```bibtex
    @inproceedings{socher-etal-2013-recursive,
        title = "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
        author = "Socher, Richard  and
          Perelygin, Alex  and
          Wu, Jean  and
          Chuang, Jason  and
          Manning, Christopher D.  and
          Ng, Andrew  and
          Potts, Christopher",
        booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
        month = oct,
        year = "2013",
        address = "Seattle, Washington, USA",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/D13-1170",
        pages = "1631--1642",
    }
    ```
  </details>
