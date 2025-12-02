import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import json

# Configure Gemini API
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print(api_key)
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

def call_llm(prompt, model):
    """
    Call the LLM and handle results safely.
    """
    try:
        response = model.generate_content(prompt)

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
    
def extract_json(text, fallback=None):
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



            
def parse_subquestions(response: str) -> list[str]:
    """
    Parse the LLM response to extract individual subquestions
    """
    subquestions = []
    
    # Split by newlines
    lines = response.strip().split('\n')
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        line_lower = line.strip().lower()
        
        # Skip header lines like "Subquestions:", "Questions:", "Here are the subquestions:", etc.
        if (line_lower.startswith('subquestion') or 
            line_lower.startswith('question') or
            line_lower.startswith('here are') or
            line_lower.startswith('breakdown') or
            line_lower == 'subquestions:' or
            line_lower == 'questions:'):
            continue
        
        # Remove numbering (e.g., "1.", "1)", "a.", etc.)
        cleaned = re.sub(r'^[\d\w]+[\.\)]\s*', '', line.strip())
        
        # Remove bullet points or dashes
        cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
        
        if len(cleaned) < 5:
            continue
        if cleaned:
            subquestions.append(cleaned)
    
    return subquestions


def decompose_query(question: str) -> list[str]:
    """
    Decompose a complex question into simpler subquestions
    """
    prompt = f"""You are an expert at breaking down complex questions into simpler subquestions.

Given a complex question, break it down into 2-4 simpler subquestions that need to be answered in sequence.
Each subquestion should be clear and answerable independently.

Examples:
Question: "Where was the director of Inception born?"
Subquestions:
1. Who directed the movie Inception?
2. Where was this director born?

Question: "What is the population of the capital of France?"
Subquestions:
1. What is the capital of France?
2. What is the population of this city?

Now break down this question:
Question: {question}

Subquestions:"""

    response = call_llm(prompt, model)
    subquestions = parse_subquestions(response)
    
    return subquestions


def answer_subquestions(subquestion, retrieved_docs):
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

    raw_output = call_llm(prompt, model)
    #print(raw_output)
    parsed_output = extract_json(raw_output)
    #print(parsed_output)
    if "answer" not in parsed_output:
        parsed_output["answer"] = "unknown"
    if "reasoning" not in parsed_output:
        parsed_output["reasoning"] = ""
    if "supporting_docs" not in parsed_output or not isinstance(parsed_output["supporting_docs"], list):
        parsed_output["supporting_docs"] = []
    return parsed_output

def answer_query(question, sub_ans):
    """
    Answer the original query based on the subquestion answers
    """
    # concat the evidence block
    answer_block = []
    for a in sub_ans:
        answer_block.append(f"Subquestion: {a['subquestion']} Answer: {a['answer']} Reasoning: {a['reasoning']}\n")
    evidence_text = "\n\n".join(answer_block)

    prompt = f"""
You are an expert multi-hop reasoning assistant.

Your task is to answer the ORIGINAL QUESTION using the results of several SUBQUESTIONS.
Each subquestion has already been answered based on retrieved evidence.

Follow these rules:
- Use the subquestion answers as your primary evidence.
- If subquestion answers conflict, analyze them and decide which are more reliable.
- If some subquestions return "unknown" or incomplete information, acknowledge the gap.
- Base your final answer ONLY on information supported by the subquestion results.
- Be concise and analytically precise.

----------------------------------------
ORIGINAL QUESTION:
{question}

----------------------------------------
SUBQUESTION RESULTS:
{evidence_text}

(Note: Each entry includes subquestion text, its answer, and the reasoning generated from its retrieved evidence.)

----------------------------------------
TASK:
1. Integrate all subquestion results.
2. Determine the best-supported answer to the ORIGINAL QUESTION in short phrases, if it is a yes/no question, only answer yes or no.
3. Provide a short explanation (2–4 sentences) showing how the subanswers combine into the final result.
4. If the information is insufficient or contradictory, answer "unknown".

----------------------------------------
Return your response in the following JSON format:

{{
  "final_answer": string,
  "final_reasoning": string
}}

"""
    raw_output = call_llm(prompt, model)
    parsed_output = extract_json(raw_output)
    if "final_answer" not in parsed_output:
        parsed_output["final_answer"] = "unknown"
    if "final_reasoning" not in parsed_output:
        parsed_output["final_reasoning"] = ""
    return parsed_output