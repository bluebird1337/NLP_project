# Academic Word Sense Disambiguation 
    
<!-- Badges -->
<p>
  <a href="">
    <img src="https://img.shields.io/badge/contributors-3-yellow" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/last%20update-January%202021-green" alt="last update" />
  </a>
</p>

## Abstract
Employed BERT and NLTK to classify the exact meanings of 15 commonly used academic keywords from the Semcor corpus.

## Introduction
In practice, extended the Bert-Chinese pre-train model to process the dissertation data obtained by web crawler. 

The input of this model is a sentence containing search keywords, and output synonyms, word senses, usage examples and confidence levels, which is cross-validated using Wordnet. 

Finally, use the Semcor corpus to test the model's accuracy, and create a website for the demo.

> WSD_Model.ipynb

Including data pre-processing, searching for alternative word candidates, designing weighting methods, similarity calculation.

> WSD_demo.py

Using streamlit to create an user interface for project demo

## Methodology
As shown in the flow chart, first we conduct the pre-process to the dissertation corpus, this step includes capitalization, punctuation, stop words and part-of-speech restoration, etc.

In the next stage, we use the Wordnet weighting method and BERT-Chinese pre-train model to calculate the closest alternative word in semantic meaning.

After finding the closest alternative word, compare it with the original target word, select the closest meaning between the two and output it as the classification result.

## Result
The final prediction accuracy rate was 70%, with a standard deviation of 0.25.

The prediction accuracy rate using the above method is highly related to the popularity of the word. It can be seen that interest with the highest accuracy rate appears 382 times in more than 37,000 sentences in corpus. The more popular and commonly used words are easier to correctly distinguish their meanings.

## Conclusion
This project is a downstream task using the BERT-pretrain model to analyze the meaning of specific AKL academic words.

The implementation method includes: data pre-processing, searching for alternative word candidates, designing weighting methods, similarity calculation, web demo, etc.

In the future, the word meaning disambiguation function designed by this research can be extended to distinguish the meaning of academic papers.