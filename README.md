# CS5588_Assignment2
##  2.2 Building and Deploying the Prototype of the Triangle Model

**Objective:**
As part of Assignment 2, each student must build and deploy a triangle model prototype consisting of a frontend, a NoSQL database, and a backend Flask Python application. The Flask application must integrate the deep learning or LLM model you reproduced in Assignment 2-1 from your Jupyter Notebook. This task will establish a functional prototype that connects all three components and deploys the model. Up to two students from the same project team may collaborate on development, but each student must submit their own individual work.

**Frontend Development:**

I used the Gradio which is an open-source Python package that allows me to quickly build a demo for our model.

**NoSQL Database:**

I set up sqlite3 (standalone command-line shell program) that can be used to create a database, define tables, insert and change rows, run queries and manage an SQLite database file.

**Flask Python Application:**

Create a Flask Python application that runs and deploys the deep learning or LLM model reproduced in Assignment 2-1 from your Jupyter Notebook.
The application should handle data exchange between the frontend and the NoSQL database.

**Connecting the Components:**

The frontend, database, and Flask Python application are fully connected and functional as a prototype.
Assignment 2-1 model was integrated and deployed within this system.

##  2.1. Reproducing Deep Learning or LLM Model

**Objective:** The primary goal of our project is to predict stock prices using news articles and social media posts. To accomplish this, we need to employ Natural Language Processing (NLP) techniques to accurately assess sentiment scores from these sources. As a starting point, I implemented an LSTM model as a baseline to serve as both a foundation and a benchmark for evaluating the performance of our proposed model

## Dataset
I used the following dataset for training the model. Please check out the corresponding websites to explore more details about the datasets

- 1. FPB - financial_phrasebank [https://huggingface.co/datasets/takala/](https://huggingface.co/datasets/takala/financial_phrasebank)
- 2. FIQA -Financial Opinion Mining and Question Answering https://sites.google.com/view/fiqa/home
- 3. TFNS - Twitter Financial Dataset https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment


## Model

Here, I experimented an LSTM model as a baseline to serve as both a foundation and a benchmark for evaluating the performance of our proposed model. Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) that is commonly used in Natural Language Processing (NLP). LSTMs are effective for NLP tasks because they can process sequential data, like language, and retain information for long periods of time. 
Here are some ways LSTMs are used in NLP: 
- Text classification: LSTMs can classify the sentiment or meaning of a text. 
- Predicting the next word: LSTMs can take a sequence of words as input and predict the next word in the series. 
- Machine translation: LSTMs are effective for machine translation tasks. 
- Named entity recognition: LSTMs are effective for named entity recognition tasks.
  
LSTMs are able to selectively retain or discard information using a memory cell and gates. This allows them to avoid the vanishing gradient problem that affects traditional RNNs. 

![image](https://github.com/user-attachments/assets/ca4b5164-58c6-4c02-b453-9de36aa39a21)

**Note:** In my code implementation, an embedding layer of dimension 100 converts each word in the sentence into a fixed-length dense vector of size 100. The input dimension is the vocabulary size, and the output dimension is 100. Hence, each word in the input will be represented by a vector of size 100. A bidirectional LSTM layer of 64 units. A dense (fully connected) layer of 24 units with relu activation. A dense layer of 1 unit and sigmoid activation outputs the probability of the review is positive, i.e., if the label is 1.

## Model Summary
<img width="500" alt="image" src="https://github.com/user-attachments/assets/a6541b52-7257-45fc-8b9d-4a48783d3b0a">

## Result
Accuracy of prediction on test set :  0.5224


## Initial setup to run the code
I used A100 GPU to try this experiment. It took around 4 minutes at total to train with 5 epochs. But in order to run this code, it is not required to have powerful GPU and can run with T4 or lower as well.

## Next step
I will explore the GNN, Roberta and LoRA models in the next step. 
