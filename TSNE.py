import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(24, 24))  #in inches

  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y, color="green")
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig("imageTSNE/"+filename)

def saveTSNE(final_embeddings, reverse_dictionary):
	try:
		from sklearn.manifold import TSNE

		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
		plot_only = 500
		low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
		labels = [reverse_dictionary[i] for i in xrange(plot_only)]
		plot_with_labels(low_dim_embs, labels, filename="tsne.png")

	except ImportError:
		print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
