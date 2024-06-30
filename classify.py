import sys
import numpy as np
from math import sqrt
import os
import csv

def cosine_similarity(v1, v2):
    num = np.dot(v1, v2)
    den_a = np.dot(v1, v1)
    den_b = np.dot(v2, v2)
    return num / (sqrt(den_a) * sqrt(den_b)) if den_a > 0 and den_b > 0 else 0.0

def read_vocab():
    with open('/home/yolan00/Desktop/nlp/data/vocab_file.txt', 'r') as f:  
        vocab = f.read().splitlines()
    return vocab

def read_queries(query_file):
    with open(query_file) as f:
        queries = f.read().splitlines()
    return queries

def read_ground_truth(ground_truth_file):
    ground_truth = {}
    with open(ground_truth_file) as f:
        for line in f:
            query, category = line.strip().split('\t')
            ground_truth[query] = category
    return ground_truth

def read_category_vectors():
    vectors = {}
    with open('/home/yolan00/Desktop/nlp/data/category_vectors.txt', 'r') as f:  # Absolute path to the category vectors file
        for l in f:
            l = l.rstrip('\n')
            fields = l.split()
            cat = fields[0].replace('_', ' ')
            vec = np.array([float(v) for v in fields[1:]])
            vectors[cat] = vec
    return vectors

def get_ngrams(text, n):
    text = text.lower()
    ngrams = {}
    for i in range(0,len(text) - n + 1):
        ngram = text[i:i + n]
        if ngram in ngrams:
            ngrams[ngram] += 1
        else:
            ngrams[ngram] = 1
    return ngrams

def normalise_tfs(tfs,total):
    for k,v in tfs.items():
        tfs[k] = v / total
    return tfs

def mk_vector(vocab, tfs):
    vec = np.zeros(len(vocab))
    for t, f in tfs.items():
        if t in vocab:
            pos = vocab.index(t)
            vec[pos] = f
    return vec

def get_titles(category):
    cat_dir = category.replace(' ', '_')
    if not os.path.exists(cat_dir):
        raise FileNotFoundError(f"Directory does not exist: {cat_dir}")
    titles = []
    with open(os.path.join(cat_dir, "linear.txt"), 'r') as f:
        for line in f:
            if line.startswith('<doc id'):
                title = line.split('title="')[1].split('">')[0]
                titles.append(title)
    return titles

def classify_queries(ngram_size, ground_truth):
    vocab = read_vocab()
    vectors = read_category_vectors()
    queries = read_queries(sys.argv[1])

    results = []
    y_true = []
    y_pred = []

    for q in queries:
        ngrams = get_ngrams(q, ngram_size)
        qvec = mk_vector(vocab, ngrams)
        
        cosines = {cat: cosine_similarity(vec, qvec) for cat, vec in vectors.items()}
        sorted_categories = sorted(cosines.items(), key=lambda item: item[1], reverse=True)
        
        top_categories = sorted_categories[:3]
        top_category = top_categories[0][0]

        try:
            titles = get_titles(top_category)
            results.append((q, top_categories, titles[:5]))
            y_true.append(ground_truth[q])
            y_pred.append(top_category)
        except FileNotFoundError as e:
            print(e)

    # Save classification results
    with open(f"classification_results_{ngram_size}.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query", "Top Category 1", "Score 1", "Top Category 2", "Score 2", "Top Category 3", "Score 3", "Top Titles"])
        for result in results:
            top_categories = result[1]
            top_category_1, score_1 = top_categories[0]
            top_category_2, score_2 = top_categories[1]
            top_category_3, score_3 = top_categories[2]
            titles_str = "\n".join(result[2])
            writer.writerow([result[0], top_category_1.split('/')[-1].replace('_', ' '), score_1, top_category_2.split('/')[-1].replace('_', ' '), score_2, top_category_3.split('/')[-1].replace('_', ' '), score_3, titles_str])
    
    # Save true and predicted labels
    with open(f"true_vs_predicted_{ngram_size}.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query", "True Label", "Predicted Label", "Correct"])
        for query, true, pred in zip(queries, y_true, y_pred):
            pred_category = pred.split('/')[-1].replace('_', ' ')  # Extract category name from path
            correct = "Yes" if true == pred_category else "No"
            writer.writerow([query, true, pred_category, correct])

if __name__ == "__main__":
    ground_truth = read_ground_truth("ground_truth.txt")
    for ngram_size in range(2, 8):  # Evaluate for n-gram sizes 2 to 7
        classify_queries(ngram_size, ground_truth)
