
from sentence_transformers import util

def flatten_context_sentences(example):
    """
    Turn example["context"] into a flat list of sentences with metadata. 
    Each returned item: {title, sent_id, text}.
    """
    flat = []
    titles = example["context"]["title"]
    sentences_per_title = example["context"]["sentences"]

    for title, sent_list in zip(titles, sentences_per_title):
        for i, sent in enumerate(sent_list):
            flat.append({
                "title": title,
                "sent_id": i,
                "text": sent,
            })
    return flat


def rank_sentences(question, sentences, model, top_k=8):
    """
    Rank sentences by semantic similarity to the question.
    """
    # Encode question and all candidate sentences
    q_emb = model.encode([question], convert_to_tensor=True)           # shape (1, d)
    s_texts = [s["text"] for s in sentences]
    s_embs = model.encode(s_texts, convert_to_tensor=True)            # shape (N, d)

    # Compute cosine similarity (1 x N) and get top_k indices
    scores = util.cos_sim(q_emb, s_embs)[0]                           # tensor (N,)
    top_k = min(top_k, len(sentences))
    top_scores, top_indices = scores.topk(top_k)

    # Collect top sentences with scores
    ranked = []
    for score, idx in zip(top_scores, top_indices):
        s = sentences[idx.item()]
        ranked.append({
            "title": s["title"],
            "sent_id": s["sent_id"],
            "text": s["text"],
            "score": float(score.item()),
        })
    return ranked



