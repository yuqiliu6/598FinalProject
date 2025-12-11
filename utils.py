import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import json
from sentence_transformers import util
from sentence_transformers import SentenceTransformer, util

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print(api_key)


class LLM:
    # Configure Gemini API
    def __init__(self):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def chat(self, prompt):
        """
        Call the LLM and handle results safely.
        """
        try:
            response = self.model.generate_content(prompt)

            # Case 1: response is plain string
            if isinstance(response, str):
                return response.strip()

            # Case 2: response has `.text` attribute (LLM frameworks vary)
            if hasattr(response, "text"):
                return str(response.text).strip()

            # Case 3: response is dict-like
            if isinstance(response, dict):
                if "text" in response:
                    return str(response["text"]).strip()
                if "content" in response:
                    return str(response["content"]).strip()

            return str(response).strip()

        except Exception as e:
            print(f"Error calling LLM: {e}")
            return f"[LLM_CALL_FAILED] Error: {e}\nPrompt was:\n{prompt}"
        
    def extract_json(self, text, fallback=None):
        """
        Extract and parse a JSON object from the LLM's response text.
        """

        if text is None:
            return fallback

        # Try to extract JSON inside ```json ... ``` blocks
        codeblock_match = re.search(
            r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE
        )
        if not codeblock_match:
            codeblock_match = re.search(
                r"```(.*?\{.*?\}.*?)```", text, flags=re.DOTALL
            )  # generic code block

        if codeblock_match:
            json_part = codeblock_match.group(1).strip()
            try:
                return json.loads(json_part)
            except Exception:
                pass  # fall through to other strategies

        # Try to find the first {...} JSON object anywhere in the output
        brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if brace_match:
            json_part = brace_match.group(0)
            try:
                return json.loads(json_part)
            except Exception:
                pass

        # try direct parse
        text_stripped = text.strip()
        try:
            return json.loads(text_stripped)
        except Exception:
            pass

        # Failed: return fallback object for debugging
        return fallback if fallback is not None else {
            "parse_error": True,
            "raw_text": text
        }
    def decompose_question(self, question):
        """
        Call the LLM to classify: leaf / branch / nest and give subquestions.

        Return format (examples):

        - {"type": "leaf"}
        - {"type": "branch", "sub_questions": [q1, q2, ...]}
        - {"type": "nest",
           "inner_question": q_inner,
           "outer_template": "Who owns {inner}?"}
        """
        system_prompt = (
    "You are a Question Planning module for multi-hop QA. "
    "Given a user question, your task is ONLY to determine the correct "
    "reasoning structure needed to answer it. You do NOT answer the question. "
    "You classify the question into exactly ONE of the following types:\n\n"

    "1. 'leaf' — The question is directly answerable with a single retrieval "
    "or single-hop reasoning. No decomposition is necessary.\n\n"

    "2. 'branch' — The question must be decomposed into multiple independent "
    "sub-questions. Each sub-question can be solved separately. The final "
    "answer is obtained by aggregating these independently-computed answers. "
    "No sub-question depends on another.\n\n"

    "3. 'nest' — The question contains a dependency. You must answer an "
    "inner sub-question first, and use its answer to complete an outer "
    "question. The outer question cannot be resolved without the inner answer.\n\n"

    "OUTPUT FORMAT (STRICT):\n"
    "You MUST return a JSON object with:\n"
    "- 'type': one of ['leaf','branch','nest']\n"
    "- If type='branch': include 'sub_questions' (a list of strings)\n"
    "- If type='nest': include:\n"
    "      'inner_question': a single string\n"
    "      'outer_template': a string containing a placeholder '{x}' where "
    "the inner answer will be inserted\n"
    "Do NOT include any text outside the JSON object.\n"
    "Do NOT answer any sub-question.\n\n"

    "CLARITY RULES:\n"
    "- For 'branch', ensure each sub-question is independent.\n"
    "- For 'nest', ensure the inner question fills in a missing entity "
    "required by the outer question.\n"
    "- Sub-questions must be logically entailed by the original question.\n"
    "- Keep the decomposition minimal and meaningful.\n\n"

    "EXAMPLES:\n\n"

    "Example 1 (leaf):\n"
    "Input: 'Who wrote Pride and Prejudice?'\n"
    "Output:\n"
    "{\n"
    "  \"type\": \"leaf\"\n"
    "}\n\n"

    "Example 2 (branch):\n"
    "Input: 'Which city is the capital of France, and what river runs through it?'\n"
    "Output:\n"
    "{\n"
    "  \"type\": \"branch\",\n"
    "  \"sub_questions\": [\n"
    "    \"What is the capital of France?\",\n"
    "    \"What river runs through the capital of France?\"\n"
    "  ]\n"
    "}\n\n"

    "Example 3 (nest):\n"
    "Input: 'What is the population of the city where Albert Einstein was born?'\n"
    "Output:\n"
    "{\n"
    "  \"type\": \"nest\",\n"
    "  \"inner_question\": \"Where was Albert Einstein born?\",\n"
    "  \"outer_template\": \"What is the population of {x}?\"\n"
    "}\n\n"

    "Now analyze the user's question and return ONLY the JSON object."
)

        user_prompt = f"\nQuestion:\n{question}\n\nReturn only JSON."

        raw = self.chat(system_prompt+user_prompt)

        extracted = self.extract_json(raw)
        print("extracted:", extracted)
        return extracted
    
    def answer_subquestions(self, subquestion, retrieved_docs):
        """
        Answer subquestions based on retrieved documents
        """
        # concat the evidence block
        evidence_blocks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            evidence_blocks.append(f"[Doc {i}]\n{doc}")
        evidence_text = "\n\n".join(evidence_blocks)

        prompt = f"""
    You are a careful, evidence-based assistant.

    Subquestion:
    {subquestion}

    You are given several evidence snippets:

    {evidence_text}

    Using ONLY the information in these snippets, answer the subquestion.

    Important instructions:
    - If the evidence clearly supports an answer, give it.
    - If the evidence is ambiguous or insufficient, answer "unknown".
    - Give a brief reasoning (1–3 sentences) that cites which Doc numbers you used.
    - If you use Doc 1 and Doc 3, for example, include [1, 3] in "supporting_docs".

    Respond in STRICT JSON with the following keys:
    {{
    "answer": string,             // short answer or "unknown"
    "reasoning": string,          // 1–3 sentences, cite docs like "Doc 1, Doc 3"
    "supporting_docs": [int, ...] // doc indices starting from 1
    }}
    """

        raw_output = self.chat(prompt)
        #print(raw_output)
        parsed_output = self.extract_json(raw_output)
        #print(parsed_output)
        if "answer" not in parsed_output:
            parsed_output["answer"] = "unknown"
        if "reasoning" not in parsed_output:
            parsed_output["reasoning"] = ""
        if "supporting_docs" not in parsed_output or not isinstance(parsed_output["supporting_docs"], list):
            parsed_output["supporting_docs"] = []
        return parsed_output



class Retriever():
    def __init__(self, model, dataset_name):
        self.model = model  
        self.dataset_name = dataset_name

    def flatten_context_sentences(self, example):
        if self.dataset_name == "hotpotqa":
            return self.flatten_context_sentences_hotpotqa(example)
        elif self.dataset_name == "musique":
            return self.flatten_context_sentences_musique(example)
        
    def flatten_context_sentences_hotpotqa(self, example):
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

    def flatten_context_sentences_musique(self, example):
        """
        Turn example["paragraphs"] into a flat list of sentences with metadata.
        Each returned item: {title, sent_id, text}.
        
        MuSiQue structure:
        example["paragraphs"] = [
            {
                "title": "...",
                "paragraph_text": "Sentence 1. Sentence 2. ...",
                "is_supporting": True/False
            },
            ...
        ]
        """
        flat = []
        
        for paragraph in example["paragraphs"]:
            title = paragraph["title"]
            # Split paragraph into sentences (simple split by '. ')
            # You might want to use a more sophisticated sentence splitter
            sentences = paragraph["paragraph_text"].split('. ')
            
            # Clean up sentences and add period back if needed
            for i, sent in enumerate(sentences):
                sent = sent.strip()
                if sent:  # Skip empty strings
                    # Add period back if it was removed by split (except for last sentence)
                    if i < len(sentences) - 1 and not sent.endswith('.'):
                        sent = sent + '.'
                    
                    flat.append({
                        "title": title,
                        "sent_id": i,
                        "text": sent,
                    })
        
        return flat


    def rank_sentences(self, question, sentences, top_k=8):
        """
        Rank sentences by semantic similarity to the question.
        """
        # Encode question and all candidate sentences
        q_emb = self.model.encode([question], convert_to_tensor=True)           # shape (1, d)
        s_texts = [s["text"] for s in sentences]
        s_embs = self.model.encode(s_texts, convert_to_tensor=True)            # shape (N, d)

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

import re
import string

PLACEHOLDER_CANONICAL = "{inner}"

def normalize_outer_template(raw: str) -> str:
    """
    Normalize various LLM styles to a single placeholder {inner}.

    Handles things like:
    - 'Who owns [ANSWER]?'       -> 'Who owns {inner}?'
    - 'Who owns {inner_answer}?' -> 'Who owns {inner}?'
    - 'Who owns {x} and {y}?'    -> only the first becomes {inner}
    """
    if not raw:
        return PLACEHOLDER_CANONICAL

    # 1) Convert [ANSWER]-style placeholders to {inner}
    #    e.g. "Who owns [ANSWER]?" -> "Who owns {inner}?"
    raw = raw.replace("[ANSWER]", PLACEHOLDER_CANONICAL)

    # 2) If there is already {inner}, we're done
    if "{inner}" in raw:
        return raw

    # 3) Find any {...} field names and normalize the *first* one
    formatter = string.Formatter()
    field_names = [fname for _, fname, _, _ in formatter.parse(raw) if fname]

    if field_names:
        first = field_names[0]
        # replace {first} with {inner}
        raw = raw.replace("{" + first + "}", PLACEHOLDER_CANONICAL)
    
    if "{}" in raw:
        raw = raw.replace("{}", PLACEHOLDER_CANONICAL)

    # 4) If we still don't have any placeholder, fall back to just appending one
    if "{inner}" not in raw:
        # Example: original template was just "Who owns this station?"
        # We'll append the answer: "Who owns this station? {inner}"
        raw = raw + " {inner}"

    return raw