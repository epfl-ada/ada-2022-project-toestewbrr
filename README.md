# How are romantic relationships depicted in movies? 

## Abstract 📰

The CMU dataset contains plot summarie and metadata on movies between 1888 and 2012. In these plot summaries, love appears as the third most common noun (after father and man… #YOUGOGIRL). We aim to assess how romantic relationships are depicted in movies, by studying the two characters that are coupled in a movie. Movies reflect the culture at the time of creation, therefore this research can provide insights on how views on romance differ across time and across the world. PART REGARDING THE RESEARCH METHODS

## Research questions ❓

To gain a comprehensive understanding of the romantic relationships shown in movies, we aim to answer the following questions:

1. Can we cluster characters in movies by their main characteristics?
2. Which types of characters are coupled in a romantic relationship in movies?
3. How is the relationship characterized (happy / violent / short)?
4. Are romantic relationships depicted differently in older movies?

## Methods ✒️

### Test
* Type of characters and type of relationships 
  * ex: are there more personas with x characteristic in y type of relation
* Find plot summaries which contain most love related words 
* Extract information about each character (job, physical details, qualities, actions) using NLP Core on plot summaries 
* Create pairs dataframe which contain the pairs of characters which are in a romantic relationship in a movie 
* Plots: clouds/clusters. Clustering of the characters based on their features. Map each character in a loving relationship on a map based on the textual vector. Get a vector from word-to-vec from all characteristics and map it. Show the link between two characters. 
* Find dataset online which maps main components of each character
* Find type of relationships. 

## Proposed timeline 
* November 19: Redefine subject ideas and submission of milestone and wordvec working
* November 25: Relationship pairs and corresponding attributes dataset
* December 2: Analysis on the relationship dataset
* December 9: Imagine and create graphs 
* December 16: 
  * Code finished and commented 
  * Full draft of the website (website is working, summary of how we are going to write the story, having all graphs available) 
  * (bonus) Interactive plots
* December 23: Hand-in the project 

## Organization within the team 
|            | **Task**                                                                                             |
|------------|------------------------------------------------------------------------------------------------------|
| Teammate 1 | Develop core NLP pipeline with teammate 2 <br /> Use core NLP to describe relationships between characters |
| Teammate 2 | Develop core NLP pipeline with teammate 1 <br /> Cluster characters by main characteristics                |
| Teammate 3 | Refine classification for romantic words  <br /> Set up the website                                         |
| Teammate 4 | Continue exploration of the dataset <br /> Develop a list of visualizations for on the website             |

## Questions 
* Find a way to label the characters or the relationships? 

## TODO: 
* General analysis: 
  * character meta-analysis
  * add first plot for the name clusters 
  * Move description to load data 
* CoreNLP 
  * example of what we can extract from core nlp directly 
  * extract characteristic from an example (marg)
  * extract relationship from an example (ant)
  * look at all words in all summaries : verbs (subj and obj), nouns
* Word2Vec (hugo)
  * get all love related words and gives you how significant 

