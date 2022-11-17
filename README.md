# A Data-Driven Analysis of Romance in Movies

**ToeStewBrr** 🍲 🦶: Antoine Bonnet, Hugo Bordereaux, Alexander Sternfeld & Marguerite Thery

## Abstract 📰

The [CMU Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/) contains plot summaries and metadata of 42,306 movies and 450,669 characters ranging from 1888 to 2012. Love takes a central place in movies; the word love appears as the third most common noun in these summaries (after father and man… #YOUGOGIRL). We aim to assess how romantic relationships are depicted in movies, by studying which two characters are coupled in a movie. Movies reflect the culture at the time of creation, therefore this enquiry can provide insights on how views on romance differ across time and across the world.

## Research questions ❓

To gain a comprehensive understanding of the characters paired in romantic relationships in movies, we aim to answer the following questions:

1. How do the demographics differ between characters in a couple (i.e. age, ethnicity, religion, gender)?
2. Which type of personalities are coupled together?
3. Are there recurrent personality types among lovers for each gender?
4. Are there different types of love couples?
5. Has the cinematic couple evolved over time?

## Methods ✒️

### 1. General analysis
To gain a better understanding of the provided datasets, we first performed an exploratory analysis. We show here a single finding from this analysis, while a thorough description and many more results can be found in `general_analysis.ipynb`. 


#### 1.1. Romantic movies

The figure below shows the runtime of romantic movies and non-romantic movies over time. From this graph, we first note that the runtime of movies increases over time. This illustrates that movies from around 1900 are often short, such as the [Dickson Experimental Sound Film](https://en.wikipedia.org/wiki/The_Dickson_Experimental_Sound_Film). Second, we find that, on average, romantic movies are longer than non-romantic movies.  

<p align="center" width="100%">
    <img width="80%" src="Images/Runtime.png">
</p>

#### 1.2. Character personalities

As a first step to discovering the personalities that are matched together in a couple, we used the tv trope personality types that were part of the CMU dataset. Characters from approximately 500 movies were classified into 72 character types. When considering romantic movies, we obtained the top 5 character types that are displayed in the histogram below. For those wondering: the defining characteristics of a "ditz" are [profound stupidness or quirkiness](https://tvtropes.org/pmwiki/pmwiki.php/Main/TheDitz). 

<p align="center" width="100%">
    <img width="80%" src="Images/Tv_trope_clusters.png">
</p>

Although this gives a rough sketch of the personalities, the classification of 500 movies is rather limited. Therefore, we will conduct our own analysis directly on the plot summaries to extract couples and character roles. 

### 2. CoreNLP analysis

[**CoreNLP**](https://nlp.stanford.edu/software/) is an incredible natural language processing toolkit created at Stanford University. CoreNLP analysis is run through a **pipeline** of sequential analysis steps called annotators. The full list of available annotators is available [here](https://stanfordnlp.github.io/CoreNLP/annotators.html). We will use CoreNLP to extract couples and lovers' persona from the plot summaries. 

#### 2.1. Exploring pre-existing analysis

The authors of the dataset had performed a preliminary analysis using CoreNLP on the plot summaries. This data was useful to extract the main character as the one with the highest number of mentions for each movie. Moreover, we extracted the main pair of interacting characters by using the number of common mentions within a sentence as a proxy for interaction. 

However, our end goal is to extract love relationships as well as the lovers' persona. Using common mentions as a proxy for love relationships is a vulgar approximation and so we must run our own NLP analysis on the plot summaries to extract more accurate information. We therefore decided to build our own customized CoreNLP pipeline on the plot summaries. 

#### 2.2. Custom CoreNLP pipeline

We now use a **custom CoreNLP pipeline** to analyze the plot summaries. For now, due to the weakness of our available computing power, we only analyze romantic comedy movies. A complete description of our NLP pipeline is available in the `coreNLP_analysis.ipynb` notebook. 

Our custom pipeline consists of the following annotators: 

1. [Tokenization (tokenize)](https://stanfordnlp.github.io/CoreNLP/tokenize.html): Turns the whole text into tokens. 

2. [Parts Of Speech (POS)](https://stanfordnlp.github.io/CoreNLP/pos.html): Tags each token with part of speech labels (e.g. determinants, verbs and nouns). 

3. [Lemmatization (lemma)](https://stanfordnlp.github.io/CoreNLP/lemma.html): Reduces each word to its lemma (e.g. *was* becomes *be*). 

4. [Named Entity Recognition (NER)](https://stanfordnlp.github.io/CoreNLP/ner.html): Identifies named entities from the text, including characters, locations and organizations. 

5. [Constituency parsing (parse)](https://stanfordnlp.github.io/CoreNLP/parse.html): Performs a syntactic analysis of each sentence in the form of a tree. 

6. [Coreference resolution (coref)](https://stanfordnlp.github.io/CoreNLP/coref.html): Aggregates mentions of the same entities in a text (e.g. when 'Harry' and 'he' refer to the same person). 

7. [Dependency parsing (depparse)](https://stanfordnlp.github.io/CoreNLP/depparse.html): Syntactic dependency parser. 

8. [Natural Logic (natlog)](https://stanfordnlp.github.io/CoreNLP/natlog.html): Identifies quantifier scope and token polarity. Required as preliminary for OpenIE. 

9. [Open Information Extraction (OpenIE)](https://stanfordnlp.github.io/CoreNLP/openie.html): Identifies relation between words as triples *(subject, relation, object of relation)*. We use this to extract relationships between characters, as well as character traits. 

10. [Knowledge Base Population (KBP)](https://stanfordnlp.github.io/CoreNLP/kbp.html): Identifies meaningful relation triples. 



#### 2.3. Character roles among genders

#### 2.2. Lover personas

#### 2.3. Gender differences

#### 2.4. Evolution in time
Explain how we will analyze it; show difference from gen analysis

## Proposed timeline ⏲️
* 19-11-2022: Submit the second milestone
* 23-11-2022: Run coreNLP augmented pipeline on all the plot summaries. 
* 25-11-2022: Extract love pairs characters with their corresponding characteristics. 
* 02-12-2022: Perform analysis on demographics and personality types between characters in a romantic relationship. 
* 09-12-2022: Run temporal analysis. Begin developing a rough draft of the datastory.
* 16-12-2022: Complete code implementation and interactive visualizations. 
* 20-12-2022: Complete datastory. 
* 23-12-2022: Final submission.

## Organization within the team 💪
|            | **Task**                                                                                             |
|------------|------------------------------------------------------------------------------------------------------|
| Antoine | Develop core NLP pipeline with Marguerite <br /> Use core NLP to describe relationships between characters |
| Marguerite | Develop core NLP pipeline with Antoine <br /> Cluster characters by main characteristics                |
| Hugo | Refine classification for romantic words  <br /> Set up the website and learn about interactive viz with Alexander                                         |
| Alexander | Continue exploration of the dataset <br /> Set up the website and learn about interactive viz with Hugo             |

## Questions for the TA ❔
* Find a way to label the characters or the relationships? 

