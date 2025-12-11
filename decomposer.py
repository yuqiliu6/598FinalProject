import json
from typing import Any, Dict
import google.generativeai as genai


SYSTEM_PROMPT = """
Decompose the given query into a set of explicit sub-questions, identify dependencies among them, 
	and extract key entities. Follow the instructions below and return only the specified JSON fields.

	Instructions:
	1. Break the original query into a small number of clear, self-contained sub-questions.
	2. Maintain logical order; each sub-question should represent one reasoning step.
	3. Label sub-questions implicitly by order: q1, q2, q3, ...
	4. Specify dependencies using these labels (e.g., "q3": ["q1", "q2"]) when a sub-question requires answers from earlier ones.
	5. Extract key entities referenced or implied in the query.
	6. Return ONLY the following fields in JSON format:
	   - "sub_questions"
	   - "dependencies"
	   - "key_entities"

	Constraints:
	- Sub-questions must be concise but context-preserving.
	- Include no more than 5 sub-questions unless absolutely required.
	- The output must be valid JSON with no additional commentary.

	Output Format (JSON Only):
	{
		"sub_questions": ["Sub-question 1", "Sub-question 2", ...],
		"dependencies": {"q2": ["q1"], "q3": ["q1", "q2"]},
		"key_entities": ["entity1", "entity2", ...]
	}

	Example:
	Input: "Which U.S. state has a capital city whose population is smaller than the state's largest city, given that this state hosted the 1984 Summer Olympics?"
	Output:
	{
		"sub_questions": [
			"Which state hosted the 1984 Summer Olympics?",
			"What is the capital city of that state?",
			"What is the population of that capital city?",
			"What is the largest city in that state?",
			"What is the population of the largest city?"
		],
		"dependencies": {
			"q2": ["q1"],
			"q3": ["q2"],
			"q4": ["q1"],
			"q5": ["q4"]
		},
		"key_entities": ["U.S. state", "capital city", "largest city", "1984 Summer Olympics"]
	}
"""


def decompose_query(model: genai.GenerativeModel, query: str) -> Dict[str, Any]:
	"""
	Uses Gemini to decompose a query into structured JSON.
	Returns the full parsed JSON object.
	"""
	prompt = f"""{SYSTEM_PROMPT}

Now decompose the following query and respond with JSON only:

Query: "{query}"
"""

	# Generate response
	resp = model.generate_content(
		prompt,
		generation_config=genai.GenerationConfig(
			temperature=0.0,
			max_output_tokens=5000,
		),
	)

	# Extract text from Gemini response
	text = ""
	try:
		if getattr(resp, "candidates", None):
			cand = resp.candidates[0]
			content = getattr(cand, "content", None)
			if content:
				parts = getattr(content, "parts", None) or []
				chunks = []
				for p in parts:
					t = getattr(p, "text", "")
					if t:
						chunks.append(t)
				text = "".join(chunks).strip()
	except Exception as e:
		print(f"Error extracting Gemini text: {e}")
		return {}

	# Attempt JSON parsing
	try:
		return json.loads(text)
	except json.JSONDecodeError:
		# Try to extract a JSON block if wrapped in extra text
		import re
		match = re.search(r"\{.*\}", text, re.DOTALL)
		if match:
			try:
				return json.loads(match.group(0))
			except json.JSONDecodeError:
				print("Failed to parse extracted JSON block.")
				return {}

	print("Failed to parse JSON from model output.")
	return {}
