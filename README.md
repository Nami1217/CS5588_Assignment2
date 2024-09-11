# CS5588_Assignment2
##  Reproducing Deep Learning or LLM Model

**Objective:** The primary goal of our project is to predict stock prices using news articles and social media posts. To accomplish this, we need to employ Natural Language Processing (NLP) techniques to accurately assess sentiment scores from these sources. As a starting point, I implemented an LSTM model as a baseline to serve as both a foundation and a benchmark for evaluating the performance of our proposed model

## Dataset
I used the following dataset for training the model. Please check out the corresponding websites to explore more details about the datasets

- 1. FPB - financial_phrasebank [https://huggingface.co/datasets/takala/](https://huggingface.co/datasets/takala/financial_phrasebank)
- 2. FIQA -Financial Opinion Mining and Question Answering https://sites.google.com/view/fiqa/home
- 3. TFNS - Twitter Financial Dataset https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment


## Model

Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) that is commonly used in Natural Language Processing (NLP). LSTMs are effective for NLP tasks because they can process sequential data, like language, and retain information for long periods of time. 
Here are some ways LSTMs are used in NLP: 
- Text classification: LSTMs can classify the sentiment or meaning of a text. 
- Predicting the next word: LSTMs can take a sequence of words as input and predict the next word in the series. 
- Machine translation: LSTMs are effective for machine translation tasks. 
- Named entity recognition: LSTMs are effective for named entity recognition tasks.
  
LSTMs are able to selectively retain or discard information using a memory cell and gates. This allows them to avoid the vanishing gradient problem that affects traditional RNNs. 

![image](https://github.com/user-attachments/assets/ca4b5164-58c6-4c02-b453-9de36aa39a21)
