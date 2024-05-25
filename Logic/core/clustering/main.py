import numpy as np
import os
from tqdm import tqdm
import json
import fasttext
import wandb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.fasttext_model import FastText
from Logic.core.word_embedding.fasttext_model import preprocess_text
from Logic.core.clustering.dimension_reduction import DimensionReduction
from Logic.core.clustering.clustering_metrics import ClusteringMetrics
from Logic.core.clustering.clustering_utils import ClusteringUtils


# Main Function: Clustering Tasks
def main():
    # 0. Embedding Extraction
    # TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
    embeddings = []
    model = fasttext.load_model(
        '/Users/maryamtavasoli/Desktop/IMDb-IR-System-main-phase2/Logic/core/word_embedding/FastText_model.bin')
    with open('/Users/maryamtavasoli/Desktop/IMDb-IR-System-main-phase2/Logic/core/indexes/preprocessed_index4.json',
              'r') as file:
        data = json.load(file)
    ft_model = FastText(preprocessor=preprocess_text, method='skipgram')
    ft_model.model = model
    true_labels = []
    for id in tqdm(data["documents"].keys()):
        if data["documents"][id]["summaries"]:
            final_embedding = None
            summaries = data["documents"][id]["summaries"]
            id_embeddings = []
            for summary in summaries:
                embedding = ft_model.get_query_embedding(summary, False)
                id_embeddings.append(embedding)
            final_embedding = np.mean(id_embeddings, axis=0)
            genre = 'Crime'
            if data["documents"][id]["geners"]:
                true_labels.append(data["documents"][id]["geners"][0])
                genre = data["documents"][id]["geners"][0]
            else:
                true_labels.append(genre)
            embeddings.append(final_embedding)
    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(true_labels)
    # 1. Dimension Reduction
    # TODO: Perform Principal Component Analysis (PCA):
    #     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
    #     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
    #     - Draw plots to visualize the results.
    dim_reduction = DimensionReduction()
    reduced_embeddings = dim_reduction.pca_reduce_dimension(embeddings, n_components=50)
    project_name = "clustering_project"
    run_name = "clustering_run"
    wandb.login(key='6bcfbb6b7bdfbcfe174074d1fa9b8a76576082ec')
    dim_reduction.wandb_plot_explained_variance_by_components(np.array(embeddings), project_name, run_name)
    dim_reduction.plot_singular_values(np.array(embeddings), project_name, run_name)
    # TODO: Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
    #     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
    #     - Use the output vectors from this step to draw the diagram.
    twod_embeddings = dim_reduction.convert_to_2d_tsne(np.array(embeddings))
    dim_reduction.wandb_plot_2d_tsne(np.array(embeddings), project_name, run_name)
    # 2. Clustering
    ## K-Means Clustering
    # TODO: Implement the K-means clustering algorithm from scratch.
    # TODO: Create document clusters using K-Means.
    # TODO: Run the algorithm with several different values of k.
    # TODO: For each run:
    #     - Determine the genre of each cluster based on the number of documents in each cluster.
    #     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
    #     - Check the implementation and efficiency of the algorithm in clustering similar documents.
    # TODO: Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
    # TODO: Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)
    clustering_utils = ClusteringUtils()
    k_values = list(range(2, 9))
    project_name = "kmeans_clustering_project"
    run_name = "kmeans_clustering_run"
    reduced_embeddings = list(reduced_embeddings)
    clustering_utils.plot_kmeans_cluster_scores(reduced_embeddings, true_labels, k_values, project_name, run_name)
    clustering_utils.visualize_elbow_method_wcss(reduced_embeddings, k_values, project_name, run_name)
    for k in k_values:
        reduced_embeddings = np.array(reduced_embeddings)
        clustering_utils.visualize_kmeans_clustering_wandb(reduced_embeddings, k, project_name, run_name)

    ## Hierarchical Clustering
    # TODO: Perform hierarchical clustering with all different linkage methods.
    # TODO: Visualize the results.
    clustering_utils = ClusteringUtils()
    linkage_method = 'ward'
    run_name = "hierarchical_clustering_ward"
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(reduced_embeddings, project_name, linkage_method,
                                                                   run_name)
    run_name = "hierarchical_clustering_average"
    linkage_method = 'average'
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(reduced_embeddings, project_name, linkage_method,
                                                                   run_name)
    run_name = "hierarchical_clustering_complete"
    linkage_method = 'complete'
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(reduced_embeddings, project_name, linkage_method,
                                                                   run_name)
    run_name = "hierarchical_clustering_single"
    linkage_method = 'single'
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(reduced_embeddings, project_name, linkage_method,
                                                                   run_name)

    # 3. Evaluation
    # TODO: Using clustering metrics, evaluate how well your clustering method is performing.
    reduced_embeddings=list(reduced_embeddings)
    kmeans_clustering = clustering_utils.cluster_kmeans(reduced_embeddings, 2)
    clustering_metrics = ClusteringMetrics()
    print("silhouette_score for kmeans clustering: ", clustering_metrics.silhouette_score(reduced_embeddings,
                                                                                          kmeans_clustering[1]))
    print("purity_score for kmeans clustering: ", clustering_metrics.purity_score(true_labels, kmeans_clustering[1]))
    print("adjusted_rand_score for kmeans clustering: ", clustering_metrics.adjusted_rand_score(true_labels,
                                                                                                kmeans_clustering[1]))
    hierarchical_clustering_single = clustering_utils.cluster_hierarchical_single(reduced_embeddings)
    print("silhouette_score for hierarchical clustering with single linkage: ",
          clustering_metrics.silhouette_score(reduced_embeddings,
                                              hierarchical_clustering_single))
    print("purity_score for hierarchical clustering with single linkage: ", clustering_metrics.purity_score(true_labels,
                                                                                                            hierarchical_clustering_single))
    print("adjusted_rand_score for  hierarchical clustering with single linkage: ", clustering_metrics.adjusted_rand_score(true_labels,
                                                                                                hierarchical_clustering_single))
    hierarchical_clustering_average = clustering_utils.cluster_hierarchical_average(reduced_embeddings)
    print("silhouette_score for hierarchical clustering with average linkage: ",
          clustering_metrics.silhouette_score(reduced_embeddings,
                                              hierarchical_clustering_average))
    print("purity_score for hierarchical clustering with average linkage: ", clustering_metrics.purity_score(true_labels,
                                                                                                            hierarchical_clustering_average))
    print("adjusted_rand_score for  hierarchical clustering with average linkage: ",
          clustering_metrics.adjusted_rand_score(true_labels,
                                                 hierarchical_clustering_average))
    hierarchical_clustering_complete = clustering_utils.cluster_hierarchical_complete(reduced_embeddings)
    print("silhouette_score for hierarchical clustering with complete linkage: ",
          clustering_metrics.silhouette_score(reduced_embeddings,
                                              hierarchical_clustering_complete))
    print("purity_score for hierarchical clustering with complete linkage: ", clustering_metrics.purity_score(true_labels,
                                                                                                            hierarchical_clustering_complete))
    print("adjusted_rand_score for  hierarchical clustering with complete linkage: ",
          clustering_metrics.adjusted_rand_score(true_labels,
                                                 hierarchical_clustering_complete))
    hierarchical_clustering_ward = clustering_utils.cluster_hierarchical_ward(reduced_embeddings)
    print("silhouette_score for hierarchical clustering with ward linkage: ",
          clustering_metrics.silhouette_score(reduced_embeddings,
                                              hierarchical_clustering_ward))
    print("purity_score for hierarchical clustering with ward linkage: ", clustering_metrics.purity_score(true_labels,
                                                                                                            hierarchical_clustering_ward))
    print("adjusted_rand_score for  hierarchical clustering with ward linkage: ",
          clustering_metrics.adjusted_rand_score(true_labels,
                                                 hierarchical_clustering_ward))


if __name__ == "__main__":
    main()
