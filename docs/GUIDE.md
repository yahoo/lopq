## Guide to using deep features for similarity applications

This is a short guide through the design and evaluation process for using features extracted from deep neural networks for similarity and deduplication applications. While it is targeted at image similarity applications, the guidelines outlined here are relevant to other types of data as well.

There are many possible similarity applications, so different use cases may require different evaluation approaches. Part of the flexibility of nearest neighbor systems is their potential for multipurpose application. Here are a handful of the most common use cases:
- nearest neighbor and approximate nearest neighbor retrieval and ranking as an end goal in itself
- classification with kNN, possibly utilizing approximate nearest neighbor retrieval at classification time
- clustering of data in the feature space
- duplicate pair detection when pairs are known explicitly
- deduplication of collections when all pairs in a collection should be considered


### 1. Assess raw features

For any similarity application, the result will be no better than the quality of the features used. A good first step is to assess the suitability of your features.

#### Similarity search

In the case of similarity search, you should obtain features for an evaluation dataset. Create a set of queries from the dataset and allow the remaining data to comprise the search index for evaluation purposes. Exhaustively compute the top nearest neighbors in the index set for each query. If you already have queries and groundtruth results for your use case you should use those in the evaluation. Otherwise you can get some insight by manually inspecting the ranked results. If your evaluation dataset has labels, you can compute performance of a kNN classifier with the features. This will particulary be useful if you expect to do nearest neighbor classification in your application or if you expect to be able to cluster the features.

#### Deduplication

In the case of deduplication, you should construct an evaluation dataset of duplicate pairs and non-duplicate pairs in equal proportion, and then compute the feature distance for each pair. Sweeping a threshold on the distance and counting accuracy, false positive rate, false negative rate, etc. will allow you to prove that the features are good for your deduplication task. You might wish to focus on some performance metrics more than others depending on your use case.

#### Dimensionality reduction

Depending on the dimension of the feature in consideration, it can be beneficial to first reduce the dimensionality of the feature with PCA or PCA+whitening. This is, firstly, to reduce the complexity of downstream distance computations and, secondly, it may improve task performance. For example, in our experience with high-dimensional feature vectors (e.g. 4096 dimensions) extracted from convolutional networks, we find that PCA'ing the feature to a lower dimension (e.g. 128) actually improves the results of similiary applications, most likely be removing some noise from the representation.

#### Getting new features

If the features prove unsuitable for the task, the proper step is to get new features. In the "worst" case this means training a new model that produces features that better capture the information relevant to the task. For images, you might try features extracted at different layers of a deep model. For example, usually in deep vision models lower layers capture more spatial or literal information (e.g. that left-facing red car is in front of a tree) and deeper layers capture more abstract information (e.g. a red car appears with foliage).

### 2. Design quantization model

After finding effective features, you may consider quantizing the features to save storage space and runtime complexity.

#### Similarity search

For similarity search applications, LOPQ is a great choice because it allows fast retrieval in large databases and it can produce accurate (though approximate) rankings at lower computational cost by amortizing computations over the distance computations in the candidate result set.

There are two primary hyperparameters in an LOPQ model to consider. The first is the number of coarse clusters in the model. A large number will increase ranking accuracy, but may also result in a lower average number of index points in each cell. This, in turn, will result in many coarse cells being retrieved and ranked per query to meet a quota of candidate results. A handful of cells can be ranked efficiently, but too many cells will impose a prohibitive overhead. Secondly, the number of subquantizers can be set relatively independently - the more subquantizers, the more accurate ranking will be, but the more memory will be required for each index item.

General advice is to set the coarse clusters as appropriate for expectations about retrieval and then set enough subquantizers for suitable ranking quality in experimentation.

#### Pairwise deduplication with hashing

If an application requires comparison of explicit pairs of items to determine whether they are duplicates, a simple approach is to threshold exact feature distance or approximate LOPQ distance between pairs. But if all that is required is determining duplicates, this is unnecessarily expensive. Instead, the LOPQ codes can be use as hashes in a scheme called Locally Optimized Hashing, or LOH. In this use case, it can be desirable to tune the LOPQ model for the best performance by hash collisions.

Since LOPQ codes are comprised of coarse codes and fine codes, it is important to note that collisions must be computed first between the coarse codes. If any coarse code matches for the pair, next compute the number of collisions for fine codes corresponding to colliding coarse codes. With this two-phase comparison procedure, there tends to be a high false negative rate when, for instance, items are close but happen by chance to be assigned to different coarse codes, thus precluding a finer-grain comparison with fine codes.

This false negative problem can be mitigated by choosing fewer coarse clusters for the LOPQ model. This allows more pairs to make it to the fine-grained phase. The next consideration is the number of subquantizers. More subquantizers will result in more gradations of collision, which could be desirable for some applications; for instance, determining a multilevel duplicate "score".

Another relevant factor for this use case is the number of subquantizer clusters. Intuitively, the larger the number of subquantizer clusters, the smaller each cluster will be. Take point A and point B that fall into all the same clusters. Now consider moving B farther from A such that each fine code changes until, finally, there are no collisions in the fine codes. Setting the number of subquantizer clusters higher will on average make this distance smaller and vice versa. Thus, the number of subquantizer clusters is important in determining the total range of distances in the feature space that can be distinguished by LOH. Similarly, the number of subquantizers is important in determining the number of distinguishable distances within this range.

#### Collection deduplication

If the use case is to detect duplicates in a set of photos, there is an effective algorithm for clustering based on LOH collisions. This is a graph-based clustering algorithm that operates on a graph constructed such that neighboring nodes have an LOH collision of at least some threshold. Finding connected components in this graph can discover groups of duplicates in the collection, even if some pairs in the group have no collisions. For details, please see the paper [here](https://arxiv.org/abs/1604.06480).

This approach can help mitigate the false negative problem described above without special tuning of the LOPQ model. In particular, the same LOPQ model optimized for similarity search can be used effectively for collection deduplication. This is a particularly great solution for search applications where the same codes can be used for similarity search in one use context and used for search result deduplication in another use context.
