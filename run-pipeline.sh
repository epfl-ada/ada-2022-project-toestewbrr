#/bin/bash

# IMPORTANT: 
# Make sure you have 5g of heap memory to run it. 
# Make sure you have enough space on your computer for the outputs: 6-8g of space will be safe. 

# PREREQ: 
# I assume you have downloaded java and coreNLP, instructions can be find there: https://stanfordnlp.github.io/CoreNLP/download.html#steps-1
# Make sure you have the latest vers standford-corenlp-4.5.1

# INSTRUCTIONS: 
# Run the data_preparation.ipynb notebook. It will create a folder Plots containing all the files you need to run. 

# PIPELINE: 
# TODO: Assume your stanford-core-nlp is just one directory up your repo, CHANGE the path if that is not the case: 
cp -r Plots ../stanford-corenlp-4.5.1/Plots
cd ../stanford-corenlp-4.5.1

# This will create a file name filelist.txt which contain all the files that you want to process
find Plots/*.txt > filelist.txt

# Be careful this takes 5g of heap memory to run. 
# For 6000 files, I needed 4g of memory on my laptop to store the outputs. Make sure you have enough place on your computer. 
java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,pos,lemma,ner,parse,coref,depparse,natlog,openie,kbp -coref.md.type RULE -coref.resolve TRUE -filelist filelist.txt -outputDirectory PlotsOutputs/ -outputFormat xml

# While this run, check if you have output in PlotsOutputs folder under stanford-corenlp-4.5.1.

# THE PIPELINE FINISHED: 
# Check that you have the same number of elements in PlotsOutputs and in Plots. 
# Rename the files to keep just the xml suffix
for file in PlotsOutputs/*.txt.xml; do
    mv "$file" "${file%.txt.xml}.xml"
done

# Compress the PlotsOutputs folder and add it to CoreNLP/PlotsOutputs_your_name.zip
