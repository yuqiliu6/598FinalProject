from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from llmquery import decompose_query, answer_subquestions, answer_query
from retrieval import rank_sentences, flatten_context_sentences
import pandas as pd



def main():
    # Load HotpotQA (distractor) via Hugging Face
    print("Loading HotpotQA (distractor)...")
    dataset = load_dataset("hotpot_qa", "distractor")
    dev = dataset["validation"]
    print(len(dev))
    dev = dev.select(range(10))

    # Load a small sentence-transformers model
    print("Loading sentence-transformers model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    example = dev[1]
    question = example["question"]
    expected_answer = example["answer"] 
    flat_sentences = flatten_context_sentences(example)
    subquestions = decompose_query(question)
    print("Subquestions:")
    print(subquestions)
    ans_list = []
    for i in range(len(subquestions)):
        print("Ranking sentence for subquestion...")
        top_k = 4
        ranked_sentences = rank_sentences(subquestions[i], flat_sentences, model, top_k=top_k)
        print(f"\nTop {top_k} sentences:")
        for j, s in enumerate(ranked_sentences, start=1):
            print(f"{j}. (score={s['score']:.4f}) [{s['title']} #{s['sent_id']}] {s['text']}")
        ans = answer_subquestions(subquestions[i], ranked_sentences)
        ans["subquestion"] = subquestions[i]
        ans_list.append(ans)    
    fianl_ans = answer_query(question, ans_list)
    print(fianl_ans)

    # df = pd.DataFrame(columns=["question", "expected_answer", "answer"])
    # idx = 0
    # for q in dev:
    #     question = q["question"]
    #     expected_answer = q["answer"]  
    #     flat_sentences = flatten_context_sentences(q)
    #     subquestions = decompose_query(question)
    #     #print("Subquestions:")
    #     #print(subquestions)
    #     ans_list = []
    #     for i in range(len(subquestions)):
    #         #print("Ranking sentence for subquestion...")
    #         top_k = 4
    #         ranked_sentences = rank_sentences(subquestions[i], flat_sentences, model, top_k=top_k)
    #         #print(f"\nTop {top_k} sentences:")
    #         # for j, s in enumerate(ranked_sentences, start=1):
    #         #     print(f"{j}. (score={s['score']:.4f}) [{s['title']} #{s['sent_id']}] {s['text']}")
    #         ans = answer_subquestions(subquestions[i], ranked_sentences)
    #         ans["subquestion"] = subquestions[i]
    #         # print(ans)
    #         ans_list.append(ans)
    #     fianl_ans = answer_query(question, ans_list)
    #     print(f"Final answer for question: {idx}")
    #     print(fianl_ans)
    #     df.loc[idx] = [question, expected_answer, fianl_ans]
    #     df.to_csv("output.csv", index=False)
    #     idx += 1


if __name__ == "__main__":
    main()
