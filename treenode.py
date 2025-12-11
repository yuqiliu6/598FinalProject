from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Protocol, Iterable
from utils import LLM, Retriever, normalize_outer_template
from sentence_transformers import SentenceTransformer, util
import random
from verifier import run_full_verifier_pipeline

class NodeKind(str, Enum):
    ROOT = "ROOT"
    BRANCH = "BRANCH"   # parallel subquestions
    NEST = "NEST"       # inner → outer
    LEAF = "LEAF"       # directly answerable

@dataclass
class ReasoningNode:
    id: str
    question: str
    kind: NodeKind
    parent: Optional["ReasoningNode"] = None
    children: List["ReasoningNode"] = field(default_factory=list)
    answer: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    accuracy: Optional[float] = None
    answered_count: int = 0

    def add_child(self, child: "ReasoningNode") -> None:
        child.parent = self
        self.children.append(child)

    def is_leaf(self) -> bool:
        return len(self.children) == 0 and self.kind == NodeKind.LEAF
    

class TreeConstructor:
    def __init__(self, llm: LLM):
        self.llm = llm
        self._next_id = 0

    def _new_id(self) -> str:
        self._next_id += 1
        return f"n{self._next_id}"

    def build_tree(self, root_question: str) -> ReasoningNode:
        root = ReasoningNode(
            id=self._new_id(),
            question=root_question,
            kind=NodeKind.ROOT,
        )
        self._expand_node(root)
        return root

    def _expand_node(self, node: ReasoningNode) -> None:
        """ decompose question into subquestion"""
        plan = self.llm.decompose_question(node.question)

        if plan["type"] == "leaf":
            node.kind = NodeKind.LEAF
            return

        if plan["type"] == "branch":
            node.kind = NodeKind.BRANCH
            for sub_q in plan["sub_questions"]:
                child = ReasoningNode(
                    id=self._new_id(),
                    question=sub_q,
                    kind=NodeKind.LEAF,  # may get expanded further
                )
                node.add_child(child)
                self._expand_node(child)

        elif plan["type"] == "nest":
            node.kind = NodeKind.NEST
            # inner question that must be answered first
            inner_q = plan["inner_question"]
            outer_template = plan["outer_template"]
            outer_template = normalize_outer_template(outer_template)
            child = ReasoningNode(
                id=self._new_id(),
                question=inner_q,
                kind=NodeKind.LEAF,
                meta={"outer_template": outer_template},
            )
            print(child.meta)
            node.add_child(child)
            self._expand_node(child)

    


class TreeExecutor:

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        threshold: float = 0.8,   # "panic" threshold at current aggregation
    ):
        self.retriever = retriever
        self.llm = llm
        self.threshold = threshold

        self._all_nodes: List[ReasoningNode] = []
        self._root: Optional[ReasoningNode] = None

    def answer_tree(self, root: ReasoningNode, sentences: List[str]) -> str:
        self._root = root
        self._all_nodes.clear()
        self._answer_node(root, sentences)
        return root.answer or ""

    def _answer_node(self, node: ReasoningNode, sentences: List[str]) -> None:
        # Keep track of all nodes in traversal order
        self._all_nodes.append(node)

        # LEAF → directly answerable via retrieval
        if node.kind == NodeKind.LEAF:
            retrieved = self.retriever.rank_sentences(node.question, sentences)
            node.answer = self.llm.answer_subquestions(node.question, retrieved)["answer"]
            # you can also verify leaves if you want
            verify_result = run_full_verifier_pipeline(node.question, node.answer, retrieved)
            node.accuracy = verify_result['accuracy']
            node.answered_count += 1
            return

        # First answer children
        for child in node.children:
            self._answer_node(child, sentences)

        # Then aggregate according to node kind
        if node.kind == NodeKind.BRANCH:
            branch_answer = self._branch_aggregate(node)
            verify_result = run_full_verifier_pipeline(node.question, branch_answer["answer"], branch_answer["child_info"])
            node.answer = branch_answer["answer"]
            node.accuracy = verify_result['accuracy']
        elif node.kind == NodeKind.NEST:
            nest_answer = self._nest_aggregate(node, sentences)
            verify_result = run_full_verifier_pipeline(node.question, node.answer, nest_answer["retrieved"])
            node.answer = nest_answer["answer"]
            node.accuracy = verify_result['accuracy']
            
        elif node.kind == NodeKind.ROOT:
            if len(node.children) == 1:
                node.answer = node.children[0].answer
                node.accuracy = node.children[0].accuracy
            else:
                branch_answer = self._branch_aggregate(node)
                verify_result = run_full_verifier_pipeline(node.question, branch_answer["answer"], branch_answer["child_info"])
                node.answer = branch_answer["answer"]
                node.accuracy = verify_result['accuracy']
        

        # If very low confidence, trigger repair / re-answer logic
        if verify_result['confidence_score'] != 1 and node.answered_count < 3:
            self._fix_answer(node, sentences)


    # ---- BRANCH_AGGREGATION ----
    def _branch_aggregate(self, node: ReasoningNode) -> str:
        """
        Example: combine answers from children and
        ask LLM to produce the entity that satisfies all of them.

        In the paper's example:
        A1': 'BBC Radio 1, KEXP'
        A2': 'BBC Radio 1, NPR'
        → BRANCH_AGGREGATION → 'BBC Radio 1'
        """
        child_info = "\n".join(
            f"Sub-question: {c.question}\nAnswer: {c.answer}"
            for c in node.children
        )

        prompt = (
            "You are a reasoning module that aggregates answers from multiple "
            "sub-questions.\n"
            "You are given several (sub-question, answer) pairs that are "
            "all about the SAME underlying entity.\n"
            "Return a short phrase describing the entity that satisfies ALL "
            "of the sub-answers, e.g. their intersection.\n\n"
            f"{child_info}\n\n"
            "Return ONLY the final entity phrase."
        )

        agg_answer = self.llm.chat(prompt)
        node.answered_count += 1
        return {
            "answer": agg_answer.strip(),
            "child_info": child_info
        }

    # ---- NEST_AGGREGATION ----
    def _nest_aggregate(self, node: ReasoningNode, sentences:List[str]) -> str:
        """
        Example: inner child answers 'BBC Radio 1'; we then ask
        'Who owns BBC Radio 1?' to get 'UK government'.
        """
        if not node.children:
            raise ValueError("NEST node without children")

        inner = node.children[0]
        inner_answer = inner.answer or ""

        # outer question template is stored on the child (from planner)
        outer_template = inner.meta.get(
            "outer_template",
            "{inner}"
        )

        outer_q = outer_template.format(inner=inner_answer)

        # Ask retriever/LLM again
        retrieved = self.retriever.rank_sentences(outer_q, sentences)
        final_answer = self.llm.answer_subquestions(outer_q, retrieved)
        print(final_answer)
        node.answered_count += 1
        return {
            "answer": final_answer["answer"],
            "retrieved": retrieved
        }
    
    def _fix_answer(self, trigger_node: ReasoningNode, sentences: List[str]) -> None:
        """
        Called when 'trigger_node' has confidence < threshold1.
        We look for a previously visited node with confidence < threshold,
        re-answer that node, and then re-aggregate up to the root.
        """
        # 1. Find candidate nodes with low confidence
        candidates = [
            n for n in self._all_nodes
            if n.accuracy is not None and n.accuracy < self.threshold
        ]

        if not candidates:
            # nothing obvious to fix
            return

        # 2. Choose the "worst" node (lowest confidence)
        target = min(candidates, key=lambda n: n.accuracy)

        # 3. Re-answer that node
        self._reanswer_node(target, sentences)

        # 4. Re-aggregate from that node upward to the root
        self._propagate_up(target, sentences)

    def _reanswer_node(self, node: ReasoningNode, sentences: List[str]) -> None:
        """
        Recompute the answer and confidence for a single node,
        without redoing the entire tree.
        """
        #print("reanswer node: \n", node.question)
        if node.kind == NodeKind.LEAF:
            retrieved = self.retriever.rank_sentences(node.question, sentences)
            node.answer = self.llm.answer_subquestions(node.question, retrieved)["answer"]
            verify_result = run_full_verifier_pipeline(node.question, node.answer, retrieved)
            node.accuracy = verify_result['accuracy']
            node.answered_count += 1
            return

        # Non-leaf: recompute children first, then aggregate
        for child in node.children:
            self._reanswer_node(child, sentences)

        if node.kind == NodeKind.BRANCH:
            branch_answer = self._branch_aggregate(node)
            verify_result = run_full_verifier_pipeline(node.question, branch_answer["answer"], branch_answer["child_info"])
            node.answer = branch_answer["answer"]
            node.accuracy = verify_result['accuracy']
        elif node.kind == NodeKind.NEST:
            nest_answer = self._nest_aggregate(node, sentences)
            verify_result = run_full_verifier_pipeline(node.question, node.answer, nest_answer["retrieved"])
            node.answer = nest_answer["answer"]
            node.accuracy = verify_result['accuracy']
            
        elif node.kind == NodeKind.ROOT:
            if len(node.children) == 1:
                node.answer = node.children[0].answer
                node.accuracy = node.children[0].accuracy
            else:
                branch_answer = self._branch_aggregate(node)
                verify_result = run_full_verifier_pipeline(node.question, branch_answer["answer"], branch_answer["child_info"])
                node.answer = branch_answer["answer"]
                node.accuracy = verify_result['accuracy']

    def _propagate_up(self, node: ReasoningNode, sentences: List[str]) -> None:
        """
        After fixing 'node', recompute all ancestors' answers and confidence
        up to the root.
        """
        #print("propagate up: \n", node.question)
        cur = node.parent
        while cur is not None:
            if cur.kind == NodeKind.BRANCH:
                branch_answer = self._branch_aggregate(cur)
                verify_result = run_full_verifier_pipeline(cur.question, branch_answer["answer"], branch_answer["child_info"])
                cur.answer = branch_answer["answer"]
                cur.accuracy = verify_result['accuracy']
            elif cur.kind == NodeKind.NEST:
                nest_answer = self._nest_aggregate(cur, sentences)
                verify_result = run_full_verifier_pipeline(cur.question, nest_answer["answer"], nest_answer["retrieved"])
                cur.answer = nest_answer["answer"]
                cur.accuracy = verify_result['accuracy']
            elif cur.kind == NodeKind.ROOT:
                if len(cur.children) == 1:
                    cur.answer = cur.children[0].answer
                else:
                    branch_answer = self._branch_aggregate(cur)
                    verify_result = run_full_verifier_pipeline(cur.question, branch_answer["answer"], branch_answer["child_info"])
                    cur.answer = branch_answer["answer"]
                    cur.accuracy = verify_result['accuracy']
            cur = cur.parent
def main():
    q = (
        "Which Oscar-nominated film was written by the screenwriter who wrote a 1991 romantic drama based upon a screenplay by Sooni Taraporevala?")
    context = {
"title": [
"Mississippi Masala",
"Part 2, Sounder",
"Sooni Taraporevala",
"Howards End (film)",
"Little Zizou",
"Duel of Hearts",
"The Namesake (film)",
"Lone Scherfig",
"Parineeta (2005 film)",
"Such a Long Journey (film)"
],
"sentences": [
[
"Mississippi Masala is a 1991 romantic drama film directed by Mira Nair, based upon a screenplay by Sooni Taraporevala, starring Denzel Washington, Sarita Choudhury, and Roshan Seth.",
" Set primarily in rural Mississippi, the film explores interracial romance between African Americans and Indian Americans in the United States."
],
[
"Part 2, Sounder is a 1976 American drama film directed by William A. Graham.",
" It is the sequel to the 1972 Oscar-nominated film \"Sounder\", which in turn is based on William H. Armstrong's Newbery Award-winning novel of the same name.",
" Although Lonne Elder III and Robert B. Radnitz returned as screenwriter and producer respectively, neither Martin Ritt nor any of the cast members from the first film participated in the sequel, with the exception of Taj Mahal, who reprised his role as Ike and returned as composer.",
" According to Bob McCann, the film was \"barely released.\""
],
[
"Sooni Taraporevala (born 1957) is an Indian screenwriter, photographer and filmmaker who is best known as the screenwriter of \"Mississippi Masala\", \"The Namesake\" and Oscar-nominated \"Salaam Bombay\" (1988), all directed by Mira Nair."
],
[
"Howards End is a 1992 British romantic drama film based upon the novel of the same name by E. M. Forster (published in 1910), a story of class relations in turn-of-the-20th-century England.",
" The film—produced by Merchant Ivory Productions as their third adaptation of a Forster novel (following \"A Room with a View\" in 1985 and \"Maurice\" in 1987)—was the first film to be released by Sony Pictures Classics.",
" The screenplay was written by Ruth Prawer Jhabvala, directed by James Ivory and produced by Ismail Merchant."
],
[
"Little Zizou is an 2008 Indian film in Hindi, Gujarati, and English, written and directed by Sooni Taraporevala.",
" \"Little Zizou\" is a fast-paced, exuberant, yet poignant comedy about how two battling Mumbai families finally come to terms."
],
[
"Duel of Hearts is a 1991 romantic television film directed by John Hough.",
" Terence Feely penned the screenplay, based on the Barbara Cartland novel, \"A Duel of Hearts\".",
" The film stars Alison Doody, Michael York, Geraldine Chaplin and Benedict Taylor."
],
[
"The Namesake is a 2006 Indian-American drama film which was released in the United States on 9 March 2007, following screenings at film festivals in Toronto and New York City.",
" It was directed by Mira Nair and is based upon the novel of the same name by Jhumpa Lahiri, who appeared in the movie.",
" Sooni Taraporevala wrote the screenplay.",
" The film received positive reviews from American critics.",
" The film stars Tabu, Irrfan Khan, Kal Penn and Sahira Nair."
],
[
"Lone Scherfig (born 2 May 1959) is a Danish film director and screenwriter who has been involved with the Dogme 95 film movement and who has been widely critically acclaimed for several of her movies, including the Oscar-nominated film \"An Education\" (2009).",
" Scherfig's movies are generally romantic comedies, including her film \"One Day\" (2011), based on the David Nicholls novel.",
" Through both experimenting with creative constraints and her astute attention to detail, she has come to be recognized as a blossoming talent in the film industry."
],
[
"Parineeta (\"The Married Woman\") is a 2005 Indian musical romantic drama film adaptation of the 1914 Bengali novella, \"Parineeta\" by Sarat Chandra Chattopadhyay.",
" Directed by debutant Pradeep Sarkar, it was based upon a screenplay by the film's producer, Vidhu Vinod Chopra.",
" The film featured Vidya Balan (In her Bollywood Debut), Saif Ali Khan and Sanjay Dutt in the lead roles.",
" Raima Sen plays the supporting role of Lalita's chirpy cousin.",
" Sabyasachi Chakrabarty plays the pivotal role of Shekhar's father.",
" Diya Mirza, with a cameo appearance as Shekhar's fiancé and Rekha, with a cameo performance of a night club (Moulin Rouge) singer, are other notable performances."
],
[
"Such a Long Journey is a 1998 Indo-Canadian english language film based on the novel of the same name written by Rohinton Mistry.",
" The film is directed by Sturla Gunnarsson with a screenplay by Sooni Taraporevala.",
" The film received twelve Genie Awards nominations including the Best Picture, Best Director, and Best Actor.",
" The film was screened at the Toronto International Film Festival."
]
]
}

    llm = LLM()
    stmodel = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    retriever = Retriever(stmodel)
    def flatten_context_sentences(example):
        """
        Turn example["context"] into a flat list of sentences with metadata. 
        Each returned item: {title, sent_id, text}.
        """
        flat = []
        titles = example["title"]
        sentences_per_title = example["sentences"]

        for title, sent_list in zip(titles, sentences_per_title):
            for i, sent in enumerate(sent_list):
                flat.append({
                    "title": title,
                    "sent_id": i,
                    "text": sent,
                })
        return flat
    flat_sentences = flatten_context_sentences(context)

    constructor = TreeConstructor(llm)
    tree_root = constructor.build_tree(q)
    # def print_tree(node: ReasoningNode, indent: int = 0):
    #     pref = " " * indent
    #     print(f"{pref}{node.id} [{node.kind}] Q: {node.question}")
    #     if node.answer:
    #         print(f"{pref}  A: {node.answer}")
    #     for c in node.children:
    #         print_tree(c, indent + 2)
    # print_tree(tree_root)
    

    executor = TreeExecutor(retriever, llm)
    final_answer = executor.answer_tree(tree_root, flat_sentences)

    print("Final answer:", final_answer)

if __name__ == "__main__":
    main()