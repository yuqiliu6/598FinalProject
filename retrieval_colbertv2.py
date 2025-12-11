from ragatouille import RAGPretrainedModel

def colbertv2_retrieve(docs, query, k=5, index_name="colbertv2_index"):
	"""
	Builds a ColBERTv2 index over `docs`,
	runs retrieval for `query`,
	and returns the retrieved texts as a list of strings.
	"""
	# Load pretrained ColBERTv2 retriever
	rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

	# Build the index
	rag.index(
		collection=docs,
		index_name=index_name
	)

	# Perform retrieval
	results = rag.search(query=query, k=k)

	# Extract only the text fields
	return [hit["text"] for hit in results]


# Usage Example:
#     docs = [
# 	"ColBERT is a late interaction neural retrieval model.",
# 	"ColBERTv2 improves efficiency and space via residual compression.",
# 	"BM25 is a lexical retrieval method based on TF-IDF-like scoring.",
# 	"PLAID is an efficient engine for ColBERTv2-style retrieval.",
# 	"ColBERT uses MaxSim over token embeddings to compute similarity."
# ]

# query = "How does ColBERTv2 compare query and document embeddings?"

# retrieved = colbertv2_retrieve(docs, query, k=3)

# print(retrieved)