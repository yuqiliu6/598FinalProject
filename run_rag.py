from datasets import load_dataset
import pandas as pd
from treenode import *
from load_classifier import *

# ds = load_dataset("musique", "musique_open")  # includes passages!
# example = ds["train"][0]
def load_hotpot(num_samples=-1):
    dataset = load_dataset("hotpot_qa", "distractor")
    dev = dataset["validation"]
    if num_samples != -1:
        dev = dev.select(range(num_samples))
    return dev
def load_musique(num_samples=-1):
    data_files = {
        "train": "D:/CS598JH/finalproject/data/musique_ans_train.jsonl",
        "validation": "D:/CS598JH/finalproject/data/musique_ans_dev.jsonl",
    }
    ds = load_dataset("json", data_files=data_files)
    dev = ds["validation"]
    if num_samples != -1:
        dev = dev.select(range(2000, 2000+num_samples))
    # for key in dev.features:
    #     print(key)
    return dev
def load_2wiki():
    pass

def debug_compelx():
    df = pd.read_csv("complex_questions.csv")
    llm = LLM()
    stmodel = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # retriever = Retriever(stmodel, 'musique')
    retriever = Retriever(stmodel, 'hotpotqa')

    df1 = pd.DataFrame(columns=["question", "expected_answer", "answer"])
    idx = 0
    for idx, row in df.iterrows():
        question = row["question"]
        expected_answer = row["expected_answer"]
        flattened_context = row["flattened_context"]
        print(type(flattened_context))
        print(flattened_context)
        break
        print("question:", question)
        print("expected_answer:", expected_answer)
        constructor = TreeConstructor(llm)
        tree_root = constructor.build_tree(question)

        print("====================Tree====================")
        def print_tree(node: ReasoningNode, indent: int = 0):
            pref = " " * indent
            print(f"{pref}{node.id} [{node.kind}] Q: {node.question}")
            if node.answer:
                print(f"{pref}  A: {node.answer}")
            for c in node.children:
                print_tree(c, indent + 2)
        print_tree(tree_root)

        executor = TreeExecutor(retriever, llm)
        final_answer = executor.answer_tree(tree_root, flattened_context)
        print("Final answer:", final_answer)
        df1.loc[idx] = [question, expected_answer, final_answer]
        df1.to_csv("output.csv", index=False)
        idx += 1

def main():
    # dataset = load_musique(2)
    dataset = load_hotpot(500)
    #dataset = [dataset[15]]

    print("Loading sentence-transformers model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    llm = LLM()
    stmodel = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # retriever = Retriever(stmodel, 'musique')
    retriever = Retriever(stmodel, 'hotpotqa')
    classifier =Classifier("musique-hop-classifier")

    df = pd.DataFrame(columns=["question", "expected_answer", "answer"])
    idx = 0
    for q in dataset:
        # run classifier
        question = q["question"]
        expected_answer = q["answer"]  
        flat_sentences = retriever.flatten_context_sentences(q)
        pred_class, pred_prob = classifier.predict(question)
        hops = pred_class + 2
        # print("Predicted hops:", hops)
        # print("question:", question)
        # print("expected_answer:", expected_answer)
        final_answer = ""
        if hops <= 2:
            # run simple strategy
            pass
        else:
            constructor = TreeConstructor(llm)
            tree_root = constructor.build_tree(question)
            print("====================Tree====================")
            def print_tree(node: ReasoningNode, indent: int = 0):
                pref = " " * indent
                print(f"{pref}{node.id} [{node.kind}] Q: {node.question}")
                if node.answer:
                    print(f"{pref}  A: {node.answer}")
                for c in node.children:
                    print_tree(c, indent + 2)
            print_tree(tree_root)
            
            executor = TreeExecutor(retriever, llm)
            final_answer = executor.answer_tree(tree_root, flat_sentences)
            print("Final answer:", final_answer)

            df.loc[idx] = [question, expected_answer, final_answer]
            df.to_csv("output.csv", index=False)
            idx += 1


if __name__ == "__main__":
    main()
