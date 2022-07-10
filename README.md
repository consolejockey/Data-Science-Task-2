
# Locality-Sensitive Hashing (LSH) and Jaccard Similarity

The commited code and dataset is part of a task for the module *Data Science I*,
conducted by Dr. Lena Wiese at Goethe-University Frankfurt am Main.

## Task
The task was to compare textual data using Jaccard similarity, LSH and minhashing.
First the documents had to be preprocessed, which means removing stop words, 
punctuation, lowercasing, and tokenizing the text. The Jaccard similarity based 
on the shingles and based on the signatures resulting from minhashing are both
calculated. Furthermore, a characteristic matrix and a matrix that contains shingles 
that appear in both documents are generated. The results are displayed visually using
the Streamlit module.

## Dataset
The dataset can be found at following websites:  
[Github page](https://github.com/socd06/medical-nlp)  
[Official MTSamples website](https://www.mtsamples.com/)  
[Kaggel repository](https://www.kaggle.com/tboyle10/medicaltranscriptions#mtsamples.csv=)

## How to run the project
To run the project, simply start the \_\_init__.py file or open a command prompt and type

**streamlit run task.py**


You might need to define the absolute path to the file.  
Please make sure to have the dataset in the same folder as the python file.

## Documentation
[Streamlit Documentation](https://docs.streamlit.io/)  
[NLTK](https://www.nltk.org/)
[Datasketch](http://ekzhu.com/datasketch/index.html)

## License
The project is published under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)