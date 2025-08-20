import pandas as pd
import numpy as np
import spacy
import scipy
from gensim.models import Word2Vec
import multiprocessing
import logging
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt

# acknowledgements
# preprocessing code adapted from https://towardsdatascience.com/turbo-charge-your-spacy-nlp-pipeline-551435b664ad
# word2vec code adapted from https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial
# clustering adapted from https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/
# evaluation (elbo, silhouette) from https://medium.com/@nirmalsankalana/k-means-clustering-choosing-optimal-k-process-and-evaluation-methods-2c69377a7ee4

# logging settings for w2v stdout
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

# Load data as dataframe
df = pd.read_csv("data/abstracts-cleaned.csv")
# df.shape
# df.head()

# Perform preprocessing: mk lowercase, rm stop words and punctuation, lemmatize
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
def preprocess(text):
    doc = nlp(text)
    lemma_list = [str(tok.lemma_).lower() for tok in doc if (not tok.is_stop and not tok.is_punct)]
    return lemma_list
# call preprocess function, add to df column
df['abstracts_preproc'] = df['abstracts'].apply(preprocess)
# also save as a list, which w2v model prefers
abs_preproc = df['abstracts_preproc'].values.tolist()

# Run word2vec model
# define parameters
w2v_model = Word2Vec(min_count=7,
                     window=4,
                     vector_size=30,
                     workers=6,
                     compute_loss=True)
# build the vocabulary, given the parameters
w2v_model.build_vocab(abs_preproc, progress_per=10000)
# train the model
w2v_model.train(abs_preproc, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

# Extract the vectors from the model
# define function
def vectorize(list_of_docs, model):
    features = []
    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features
# call function and write to list
vectorized_docs = vectorize(abs_preproc, model=w2v_model)

# Option to output vectors to pickle
# with open('data/w2v_vectors.pkl', 'wb') as f:
#    pickle.dump(vectorized_docs, f)
# with open('data/w2v_vectors.pkl', 'rb') as f:
#    vectorized_docs = pickle.load(f)


# Note: I experimented with scaling, but ultimately chose not to, 
# as https://arxiv.org/abs/1508.02297 showed that scaling word embedding 
# vectors reduces the amount of useful word significance information in 
# domain-specific corpora.
# scale data prior to PCA
# std_scaler = StandardScaler()
# scaled_df = std_scaler.fit_transform(vectorized_docs)

# Perform PCA dimension reduction
# put vectors into dataframe
vector_df = pd.DataFrame(vectorized_docs)
# put vectors into np.array
# vectorized_docs_arr = np.array(vectorized_docs)

# Dimension reduction
# initialize PCA with the number of components equal to the 
# number of word2vec features (i.e., vector_size from w2v model)
full_components = vector_df.shape[1]
pca = PCA(n_components=full_components)
# fit PCA to the scaled data and save to dataframe
# this data (with all PCs, 30-d) will be used at the end to extract 
# most representative tokens/documents, but is not used for clustering. see note below.
pca_vec_size = pd.DataFrame(pca.fit_transform(vector_df), columns=[f"PC{i+1}" for i in range(full_components)])

# Use cumulative variance to determine the optimal number of components
# calculate the explained variance and cumulative explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)
# select the number of components based on the desired explained variance threshold
# note: 0.90 and 0.95 are more ideal, but they required keeping more PCs, which resulted
# in poorer clusters
desired_explained_variance = 0.80
num_components = np.argmax(cumulative_explained_variance >= desired_explained_variance) + 1
# output the selected number of components
print(f"Number of components to explain {desired_explained_variance:.2f} variance: {num_components}")

# re-do the PCA, using the desired variance level and number of components
# fit PCA with the selected number of components and save to dataframe
pca = PCA(n_components=num_components)
PCA_ds = pd.DataFrame(pca.fit_transform(vector_df), columns=[f"PC{i+1}" for i in range(num_components)])
PCA_ds.to_csv('PCA_ds.csv', index=False)

# Create sub-corpora for 4 year increments (e.g. 1996-1999, 2000-2003, etc)
# Note: I ran out of time to create an elegant function to generate the sub-corpora,
# so I manually extracted the vectors from PCA_ds.csv and created a CSV file
# for each 4-year increment, which was used for clustering.
# Also note: I performed clustering using the PCA reduced dimension 
# vectors (PCA_ds). However, I got an error when running the later functions to 
# extract the most representative tokens/documents because the matrix sizes
# do not match. That is, the cluster centroids are taken from the PCA data (12-d) and
# the tokens/documents that were closest to the centroid were calculated using
# the original word2vec model (30-d). My solution was to extract the 
# most representative tokens/documents using the full PCA data (30-d). While
# this approach is not ideal, the additional 18-d of components contain relatively 
# minimal information (20%), and this information should have a minimal impact on  
# which tokens and documents appear closest to each cluster's centroid.


# Define k-means clustering function, with parameters:
# X: dataset, use vectors from 4 year increments
# k: number of clusters, as determined by ELBO method
# mb: minibatches, keep at 500
# print_silhouette_values: keep as True; outputs evaluation score for each cluster
def mbkmeans_clusters(
    X, 
    k, 
    mb, 
    print_silhouette_values, 
):
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia:{km.inertia_}")
    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km, km.labels_,


# The following code (load data, ELBO, kmeans, davies-bouldin index, plot 2-d PCA) is
# to be run on the CSV files containing vectorized embeddings for each of 4-year increments. 
# Be sure to generate the following outputs each time:
# 1. Save the ELBO plot as an image file
# 2. Save the Silhouette values from stdout to a text file
# 3. Add the Davies-Bouldin index to the text file
# 4. Save the 2-d PCA plot as an image file

# Load vectorized embeddings for a 4-year increment
df1996_1999 = pd.read_csv("data/vectors_by_year/9699vec.csv")

# Use ELBO to determine number of clusters
# initialize the KElbowVisualizer with the KMeans estimator and a range of K values
Elbow_M = KElbowVisualizer(KMeans(), k=30)
# fit the visualizer to the PCA-transformed data
Elbow_M.fit(df1996_1999)
# display the elbow method plot
Elbow_M.show()

# Call the clustering function, add parameters
# update X and k each time, keep mb and print_silhouette_values constant
clustering, cluster_labels = mbkmeans_clusters(
    X=df1996_1999,
    k=11,
    mb=500,
    print_silhouette_values=True,
)

# Add cluster data to dataframe
# create empty dataframe, to be reused
df_years = pd.DataFrame()
df_years['km_labels'] = cluster_labels

# K-means evaluation metric: Davies-Bouldin index
davies_bouldin_score = davies_bouldin_score(df1996_1999, df_years['km_labels'])
print(davies_bouldin_score)

# Plot k-means clustering, using first 2 PCA components
plt.scatter(df1996_1999['PC1'], df1996_1999['PC2'],  
           c = df_years['km_labels'], cmap =plt.cm.winter)
plt.show()


# Lastly, once the clustering is done, get the 40 most representative tokens for all
# of the clusters, and the 10 most representative abstracts for the top 3 clusters in
# each 4-year increment. I am still working on extracting and analyzing the most 
# representative abstracts for all clusters in order to validate the topics from the most
# representative tokens.

# Functions return the most representative tokens and documents are returned to stdout. 
# Copy this into a text document (one for each cluster) and use to label the cluster
# using content analysis.

# Extract topics from clusters for analysis
# get most representative tokens per cluster, ie topics
# replace k (in the for loop) with # of clusters for the 4-year increment
# topn: adjust the number of most representative tokens. there is some noise
# in the topics, so topn definitely need to be > 10
for i in range(11):
    tokens_per_cluster = ""
    most_representative = w2v_model.wv.most_similar(positive=[clustering.cluster_centers_[i]], topn=40)
    for t in most_representative:
        tokens_per_cluster += f"{t[0]} "
    print(f"Cluster {i}: {tokens_per_cluster}")

# Return the most representative documents for a particular cluster to stdout
# replace best_k with the number corresponding to the cluster with
# the highest silhouette score. do this 3 times, to get the 10 abstracts
# associated with the 3 best clusters. note: for the conference paper, this will
# need to be done for every cluster!
test_cluster = best_k
most_representative_docs = np.argsort(
    np.linalg.norm(df1996_1999 - clustering.cluster_centers_[test_cluster], axis=1)
)
abs_ls = df['abstracts'].values.tolist()
for d in most_representative_docs[:10]:
    print(abs_ls[d])
    print("-------------")
