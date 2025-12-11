import os
import math
import textwrap
from typing import List, Dict, Any

from datasets import load_dataset
import google.generativeai as genai


class Evaluator:
	def __init__(self, client):
		self.client = client

	def score_accuracy(self, question: str, answer: str, evidences: List[str]) -> float:
		"""
		Use LLM-as-a-judge to score accuracy of the answer
		based on the question and provided evidences.
		Output: float between 0 and 1.
		"""
		evidence_text = "\n\n".join([f"[Evidence {i+1}]\n{ev}" for i, ev in enumerate(evidences)])

		system_prompt = (
			"""
			You are an accuracy evaluator. Your task is to score how well the ANSWER is supported by the EVIDENCE with respect to the QUESTION.
			Score the answer on a continuous scale from 0.0 to 1.0, where:
			- 1.0 means fully correct, fully supported, and complete.
			- 0.0 means completely incorrect, contradictory, or unsupported.
			- Values between 0 and 1 should reflect partial correctness or incomplete support.

			Use this rubric:
			• 0.8–1.0: Mostly or fully correct; strong evidence support; no major omissions.
			• 0.5–0.79: Partially correct; some support exists but answer is incomplete or slightly inaccurate.
			• 0.2–0.49: Weak correctness; fragments match but major parts are missing or unsupported.
			• 0.01–0.19: Barely correct; only trivial overlap with evidence.
			• 0.0: Entirely incorrect, contradicted, or hallucinated.

			Do NOT round to 0 or 1 unless the case is clearly extreme.
			Return ONLY the numeric score as a decimal between 0 and 1.
			"""
		)

		user_prompt = f"""
				Question:
				{question}

				Answer:
				{answer}

				Evidences:
				{evidence_text}

				What is the score?
				"""

		response = self.client.chat.completions.create(
			model="gpt-4o-mini",
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt}
			],
			max_tokens=10,
			temperature=0.0
		)

		score_str = response.choices[0].message.content.strip()
		print('score', score_str)

		try:
			score = float(score_str)
		except:
			import re
			match = re.search(r"0\.\d+|1\.0|1|0", score_str)
			if match:
				score = float(match.group(0))
			else:
				score = 0.0

		return max(0.0, min(score, 1.0))


class AttrScoreModel:
	def __init__(self, client, model_name: str = "gpt-4o-mini"):
		self.client = client
		self.model_name = model_name

	def score_attr(self, question: str, answer: str, evidences: List[str]) -> str:
		claim = f"Question: {question}\nAnswer: {answer}"
		reference = "\n".join(f"- {e}" for e in evidences)

		system_prompt = textwrap.dedent("""
			You are an Attribution Validator.
			Your task is to verify whether a given reference can support a given claim.

			You must classify the relationship between the claim and the reference as exactly ONE of:
			- Attributable
			- Extrapolatory
			- Contradictory

			Respond with EXACTLY one word.
		""")

		user_prompt = textwrap.dedent(f"""
			Claim:
			{claim}

			Reference:
			{reference}

			What is the relationship?
		""")

		response = self.client.chat.completions.create(
			model=self.model_name,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			temperature=0.1,
			max_tokens=30,
		)

		raw = response.choices[0].message.content.strip()
		print('attr_label_raw', raw)

		try:
			label = raw.split()[0]
		except Exception:
			return "Extrapolatory"

		label_norm = label.lower()

		if "attrib" in label_norm:
			return "Attributable"
		if "contrad" in label_norm:
			return "Contradictory"
		if "extra" in label_norm:
			return "Extrapolatory"

		return "Extrapolatory"


class Verifier:
	def __init__(self, llm_judge: Evaluator, attr_model: AttrScoreModel, threshold: float = 0.75):
		self.llm_judge = llm_judge
		self.attr_model = attr_model
		self.threshold = threshold

	@staticmethod
	def _map_attr_label_to_score(label: str) -> float:
		label = label.strip().lower()
		if label == "attributable":
			return 1.0
		elif label == "contradictory":
			return 0.0
		elif label == "extrapolatory":
			return 0.5
		return 0.0

	def verify(self, question: str, answer: str, evidences: List[str]) -> Dict[str, Any]:
		accuracy = self.llm_judge.score_accuracy(question, answer, evidences)
		accuracy = float(max(0.0, min(1.0, accuracy)))

		attr_label = self.attr_model.score_attr(question, answer, evidences)
		credibility = self._map_attr_label_to_score(attr_label)

		final_score = (accuracy ** 0.6) * (credibility ** 0.4)
		print('final_score', final_score)

		confidence_score = 1 if final_score >= self.threshold else 0

		return {
			"accuracy": accuracy,
			"credibility": credibility,
			"attr_label": attr_label,
			"final_score": final_score,
			"confidence_score": confidence_score,
		}


class GeminiChatAdapter:
	def __init__(self, model_name: str = "gemini-2.5-pro"):
		genai.configure(api_key="AIzaSyDu0I916eEzyNF6dbvNgBXHhcG2rzYao-E")
		self._model = genai.GenerativeModel("gemini-2.5-pro")
		self.chat = self._Chat(self)

	class _Chat:
		def __init__(self, outer: "GeminiChatAdapter"):
			self.completions = outer._Completions(outer)

	class _Completions:
		def __init__(self, outer: "GeminiChatAdapter"):
			self.outer = outer

		def create(self, model: str, messages: List[Dict[str, str]],
		           max_tokens: int = 10, temperature: float = 0.0):

			parts = []
			for m in messages:
				role = m.get("role", "user")
				content = m.get("content", "")
				if role == "system":
					prefix = "System: "
				elif role == "user":
					prefix = "User: "
				else:
					prefix = ""
				parts.append(prefix + content)
			prompt = "\n\n".join(parts)

			resp = self.outer._model.generate_content(
				prompt,
				generation_config=genai.GenerationConfig(
					temperature=temperature,
					max_output_tokens=5000,
				),
			)

			text = ""
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

			class _Message:
				def __init__(self, content): self.content = content

			class _Choice:
				def __init__(self, message): self.message = message

			class _Response:
				def __init__(self, choices): self.choices = choices

			return _Response([_Choice(_Message(text))])


def main(max_samples: int = 100):
	client = GeminiChatAdapter(model_name="gemini-2.5-pro")

	llm_judge = Evaluator(client)
	attr_model = AttrScoreModel(client, model_name="gemini-2.5-pro")
	verifier = Verifier(llm_judge, attr_model, threshold=0.75)

	dataset = load_dataset("osunlp/AttrScore", "attreval_gensearch")
	data_split = dataset.get("test") or dataset.get("validation") or dataset.get("train")

	print("Example keys:", data_split[0].keys())

	total = len(data_split)
	correct_attr_label = 0
	correct_confidence = 0
	false_positives = 0
	false_negatives = 0
	for i in range(total):
		ex = data_split[i]

		question = ex.get("query", "")
		answer = ex.get("answer", "")
		reference = ex.get("reference", "")
		evidences = [reference] if reference is not None else []
		gold_label = ex.get("label")

		result = verifier.verify(question, answer, evidences)
		pred_label = result["attr_label"]
		pred_conf = result["confidence_score"]

		if str(pred_label).lower() == str(gold_label).lower():
			correct_attr_label += 1
		elif str(pred_label) == "attributable" and str(pred_label).lower() != "attributable":
			false_positives += 1
		elif str(pred_label) != "attributable" and str(pred_label).lower() == "attributable":
			false_negatives +=1
		gold_conf = 1 if str(gold_label).lower() == "attributable" else 0

		print("gold-label", gold_label)
		print("pred-label", pred_label)
		print("pred-conf-score", pred_conf)
		print('---------------------------------')

		if pred_conf == gold_conf:
			correct_confidence += 1

		if (i + 1) % 10 == 0:
			print(f"Processed {i+1}/{total} examples...")
			print(f"Attr label accuracy:      {correct_attr_label / total:.3f}")
			print(f"ConfidenceScore accuracy: {correct_confidence / total:.3f}")
			print(f"False negative rate: {false_negatives / total:.3f}")
			print(f"False positive rate: {false_positives / total:.3f}")

	attr_accuracy = correct_attr_label / total
	conf_accuracy = correct_confidence / total

	print("\n=== Evaluation on AttrEval-Simulation (Gemini 2.5 Pro) ===")
	print(f"Attr label accuracy:      {attr_accuracy:.3f}")
	print(f"ConfidenceScore accuracy: {conf_accuracy:.3f}")
	print(f"False negative rate: {false_negatives / total:.3f}")
	print(f"False positive rate: {false_positives / total:.3f}")


if __name__ == "__main__":
	main(max_samples=100)
