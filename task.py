import csv
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
from datasketch import MinHash, MinHashLSH
import statistics
import streamlit as st
import random
import os


filepath = os.path.dirname(__file__)
path_to_dataset = filepath + "mtsamples.csv"

stop = list(stopwords.words("english"))


def extract_data_from_file(path):
    with open(path, "r", encoding="utf8") as file:
        data = []
        reader = csv.reader(file)
        for row in reader:
            if len(row[4]) == 0:
                continue
            else:
                pass
            data.append(row[4])
        data.pop(0)
    return data


def properties(dataset: list):
    '''
    Parsing a list with texts as items, this function calculates and returns
    basic statistic information about it.
    '''

    amount = len(dataset)
    size = [len(text) for text in dataset]
    maximum = max(size)
    minimum = min(size)
    mean = statistics.mean(size)
    median = statistics.median(size)
    shorter_100 = len([text for text in size if text < 100])
    btwn_1k_5k = len([text for text in size if 1000 <= text <= 5000])
    btwn_5k_10k = len([text for text in size if 5000 <= text <= 10000])
    above_10k = len([text for text in size if text > 10000])

    print(f"Amount of texts: {amount}\n"
          f"Maximum length: {maximum}\n"
          f"Minimum length: {minimum}\n"
          f"Text length mean: {mean}\n"
          f"Text length median: {median}\n"
          f"Text length < 100: {shorter_100}\n"
          f"Text length 1,000-5,000: {btwn_1k_5k}\n"
          f"Text length 5,000-10,000: {btwn_5k_10k}\n"
          f"Text length > 10,000: {above_10k}"
          )
    return (size, maximum, minimum, mean, median, shorter_100, btwn_1k_5k,
            btwn_5k_10k, above_10k)


def preprocess(text: str) -> list:
    '''
    The preprocess-function takes a string and formats it for further use.
    For a list of stop words the nltk module has been used.

    Parameters
    ----------
    text : str
        A single string representing the text that gets preprocessed.

    Returns
    -------
    tokens : list
        A list containing the formatted tokens of the parsed text.
    '''

    text = "".join([i for i in text if not i.isdigit()])
    text = text.translate(str.maketrans("", "", string.punctuation))
    token_lst = list(nltk.word_tokenize(text.lower()))
    tokens = [word for word in token_lst if word not in stop]
    return tokens


def jaccard(*args) -> float:
    '''
    The jaccard-function takes an arbitrary amount of lists and
    calculates their Jaccard similarity.

    Parameters
    ----------
    *args : list
        An arbitrary amount of lists containing tokenized texts.

    Returns
    -------
    float
        A float representin the jaccard similarity of the parsed lists.
    '''

    start = [set(s) for s in args]
    inter = len(start[0].intersection(*start))
    union = len(start[0].union(*start))

    return inter / union


def shingling(items: list, length: int) -> list:
    '''
    The shingling-function takes a list and an integer as it's parameters.
    The integer sets the shingle size (based on words) and shingles
    the list content accordingly.

    Parameters
    ----------
    items : list
        A tokenized text.
    length : int
        Determines the shingle size.

    Raises
    ------
    ValueError
        If the shingle size is larger than the amount of tokens in the given
        list, a ValueError is raised.

    Returns
    -------
    s : list
        A list containing shingles based on the determined shingle size.
    '''

    if length == 1:
        return items
    elif length > len(items):
        raise ValueError("The shingle size is larger than the amount of items "
                         "in the document.\nPlease choose a smaller "
                         "shingle size.")
    else:
        shingles = list()
        index = 0
        for i in items:
            if index + length > len(items):

                return shingles
            new = " ".join(items[index:index + length])
            shingles.append(new)
            index += 1
        # Removing duplicates.
        shingles = list(dict.fromkeys(shingles))
        return shingles


def gen_matrix(doc_1, doc_2):
    '''
    This function takes two documents and generates a characteristic matrix
    based on the appearing shingles. If a shingle from the universal set
    appears in the document, a 1 is placed in the same row. Otherwise a 0
    gets noted.

    Parameters
    ----------
    doc_1 : list
        A list containing shingles.
    doc_2 : list
        A list containing shingles.

    Returns
    -------
    matrix : dataframe
        A matrix representing the characteristic matrix based on two documents.
    '''

    combined_shingles = doc_1 + doc_2
    combined_shingles = list(dict.fromkeys(combined_shingles))

    document_1 = []
    document_2 = []

    for shingle in combined_shingles:
        if shingle in doc_1:
            document_1.append(1)
        else:
            document_1.append(0)

        if shingle in doc_2:
            document_2.append(1)
        else:
            document_2.append(0)

    matrix = pd.DataFrame({
        "Shingles": combined_shingles,
        "Document 1": document_1,
        "Document 2": document_2})

    return matrix


def match_matrix(dataframe):
    '''
    The match_matrix-function takes a pandas dataframe as it's only parameter
    and returns a filtered version of it. In the filtered dataframe only the
    rows that have a 1 in the Document 1 and Document 2 column are returned.

    Parameters
    ----------
    dataframe : dataframe
        A pandas dataframe.

    Returns
    -------
    match : dataframe
        A dataframe where both columns contain a 1.
    '''

    match = dataframe.loc[(dataframe['Document 1'] == 1) & dataframe
                          ['Document 2'] == 1]
    return match


if __name__ == "__main__":

    st.title("Locality-Sensitive Hashing and Jaccard Similarity")
    st.caption("You can determine the shingle size by using the slider. The "
               "matrix and Jaccard similarity will change accordingly. The "
               "first Jaccard similarity is based on the document shingles "
               "and the second Jaccard similarity is calculated by using the "
               "minhashed-based signatures. If you want to compare two other "
               "documents, please press the 'Get new documents'-button. "
               "Documents are always chosen randomly.")

    data = extract_data_from_file(path_to_dataset)

    slider = st.slider("Shingle size", min_value=1, max_value=100)

    # Preserve documents during slider change.
    if "doc1" not in st.session_state:
        doc1 = random.choice(data)
        doc2 = random.choice(data)
        st.session_state.doc1 = doc1
        st.session_state.doc2 = doc2

    if st.button("Get new documents."):

        doc1 = random.choice(data)
        doc2 = random.choice(data)
        st.session_state.doc1 = doc1
        st.session_state.doc2 = doc2

    pre_doc1 = preprocess(st.session_state.doc1)
    pre_doc2 = preprocess(st.session_state.doc2)

    shing_doc1 = shingling(pre_doc1, slider)
    shing_doc2 = shingling(pre_doc2, slider)

    matrix = gen_matrix(shing_doc1, shing_doc2)
    match = match_matrix(matrix)

    real_jaccard = jaccard(shing_doc1, shing_doc2)

    st.header("Characteristic Matrix")
    st.write(matrix)

    st.header("Match Matrix")
    st.caption("The Match Matrix shows all shingles that appear in both "
               "documents.")
    st.write(match)

    st.header("Jaccard Similarity #1")
    st.caption("The Jaccard Similarity #1 is based on the document shingles.")
    st.write(real_jaccard)

    m_doc1 = MinHash(num_perm=256)
    m_doc2 = MinHash(num_perm=256)

    for shingle in shing_doc1:
        m_doc1.update(shingle.encode("utf8"))

    for shingle in shing_doc2:
        m_doc2.update(shingle.encode("utf8"))

    lsh = MinHashLSH(threshold=0.1, num_perm=256)
    lsh.insert("m_doc1", m_doc1)
    result = lsh.query(m_doc2)

    sig_jaccard = m_doc1.jaccard(m_doc2)
    st.header("Jaccard Similarity #2")
    st.caption("The Jaccard Similarity #2 is based on the signatures "
               "created by the minhash functions.")
    st.write(sig_jaccard)

    if result:
        st.write("Documents considered a candidate pair.")
    else:
        st.write("Documents not considered a candidate pair.")
