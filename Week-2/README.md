# Week 2  & MidEval - Vibe Checker

**The Basics**

I hope you guys had a good experience with the week 1 resources. They may have been a bit challenging but they were meant to give you a mathematical background into Machine Learning - something which a lot of people lack & something without which progressing forward can be quite challenging. For those of you who felt like it was too theoretical, you guys are going to enjoy this weeks assignment as we will focus purely on implementing things from now. Considering that this is probably the first time you guys are using any ML related framework, I would highly encourage you guys to follow the flow of this document otherwise you would end up getting confused and lost. Trust me, staying structured will save you a lot of time later on.

As I mentioned in the last meet, try to form groups to solve the assignments and since this assignment is a bit long, **you are expected to work in groups formed in the last meet.** Also, your accuracy/results do not matter to me till the time I can see the effort, blatant GPT is always discouraged as opposed to spending time to learn the content.

**We will be conducting the mideval on 19th December tentatively, you will be presenting this weeks work along with your group in the mideval. Guidelines for the presentation would be shared in due time.**  
**We will replicate the end-eval experience to a great extent so do put in some effort into the mideval presentations - it would help you a lot for the end-eval.**
**The final deliverable is the complete github repo with proper documentation written in LaTeX and the PPT.**

**There is only one rule - YOU ARE NOT ALLOWED TO USE TENSORFLOW. ONLY PYTORCH IS ALLOWED.** (Brownie points for those who can tell me the reason for this restriction).

I would encourage you guys to spend more time doing the tasks than reading about stuff, just skim through the resources provided, do not go in much depth, you will learn way more by implementing than just by reading stuff.


## Artificial Neural Networks


**Background & Resources**

ANNs are the base of most modern Deep Learning algorithms and frameworks. Let's start with implementing basic ANNs using PyTorch and learning about them first. You can skip the resources by watching the entire 3B1B chapter on Deep Learning, if you prefer.

**Resources:**  
**1.** [3B1B chapter on ANNs](https://www.youtube.com/watch?v=aircAruvnKk&vl=en)  
**2.** [IBM Article on ANNs](https://www.ibm.com/think/topics/neural-networks#741977106)  
**3.** [IBM Article on BackProp](https://www.ibm.com/think/topics/neural-networks#741977106)  
**4.** [3B1B chapter on BackProp](youtube.com/watch?v=Ilg3gGewQ5U&vl=en)

**Task:**  
For this task, you will be required to implement an Artificial Neural Network using PyTorch on [this dataset](https://www.kaggle.com/datasets/filippoo/deep-learning-az-ann/data). Your deliverable would be the python notebook (with comments on hyperparameter tuning).  
Reference article: [Build an ANN using Torch](https://jillanisofttech.medium.com/building-an-ann-with-pytorch-a-deep-dive-into-neural-network-training-a7fdaa047d81)

:bulb: **Pro Tip:** Use [Kaggle](https://www.kaggle.com/) for doing the tasks, much more convenient than fiddling with Google Colab, but do note that you only get 30hrs of GPU time per week, so use it wisely.

## Convolutional Neural Networks

**Extension of Neural Networks for Computer Vision (CV) tasks.**

Here is a paper summarising the basics of CNNs: [CNN Paper](https://arxiv.org/pdf/1511.08458).  
For those of you who do not like long articles, watch [this](https://www.youtube.com/watch?v=oGpzWAlP5p0) instead (you need not watch Object Detection, that is an advanced CV topic, out of scope for this project).  
Additionally, this [IBM Article](https://www.ibm.com/think/topics/convolutional-neural-networks#763338459) is also a great resource.

**The Task:**  
Implement a basic CNN on the [MNIST Dataset](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html) (this is by far the most trivial CV task, shouldn't take you more than a day). Again the deliverable remains the same -- a python notebook with notes on hyperparameter tuning.

## Recurrent Neural Networks and LSTMs

**Extension of Neural Networks for sequential data.**

CNNs are limited to the extent that they cannot process sequential data, the input to a CNN must be of fixed size, but language data need not be of fixed size, to overcome this problem, RNNs were introduced as a basic extension of ANNs. Later on to overcome the shortcomings of RNNs, LSTMs and GRUs were introduced to cater to "memory" as well. 

Resources:  
**1.** [IBM Article on RNNs](https://www.ibm.com/think/topics/recurrent-neural-networks#763338458)  
**2.** [Standford Lecture on RNNs](https://www.youtube.com/watch?v=6niqTuYFZLQ) (Learn about limitations of RNNs before moving on to LSTMs)  
**3.** [Christopher Olah's (Co-Founder of Anthropic) Post on LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) or if you prefer [videos](https://www.youtube.com/watch?v=0LixFSa7yts)  
**4.** [PyTorch documentation for implementation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)  

**Task:**  
Stock price prediction using sequential models. [This](https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632) is a decent reference article, but for this task I would encourage you to create your own dataset (use APIs like [Yahoo Finance](https://pypi.org/project/yfinance/) or get it from Kaggle) and train a sequential model for the same (Brownie points if you can implement both the sequential models!).

## Transformer Architecture

**The most revolutionary architecture in recent years, base for models such as ChatGPT, Gemini and Claude.**

This is the basic tech behind all the advanced Large Language Models (LLMs) that you probably use for daily tasks. Was introduced by [this](https://arxiv.org/pdf/1706.03762) revolutionary paper, you may read it if you want to (not a compulsion).

Resources:  
**1.** [Transformer Model](https://www.ibm.com/think/topics/transformer-model)  
**2.** [What is GPT?](https://www.ibm.com/think/topics/gpt)  
**3.** [3B1B Chapter on LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M)  
**4.** [Hugging Face :hugging_face: Environment for LLMs/VLMs](https://medium.com/@nimritakoul01/a-comprehensive-guide-to-large-language-model-applications-with-hugging-face-7da9085c0c19)  
**5.** [Guide to LaTeX (this is important, you would need LaTeX in your academics too)](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes)

**Task:**  
Write a report (300 words at max) about what you understood from the resources given above. Try to read about **AGI (Artificial General Intelligence)** and integrate it in your writing.  
You are supposed to submit a **LaTeX PDF file of the report, cite all references properly.**
**Make this the Appendix of your documentation.**
