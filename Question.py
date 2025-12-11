from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SubQueryDatum:
	subquery: str								# One decomposed subquery
	answer: str = ""							# Answer to this subquery
	evidences: List[str] = field(default_factory=list)
	confidence: List[Optional[float]] = field(default_factory=list)
	acc_score: Optional[float] = None			# e.g., combined score

	def to_dict(self):
		return {
			"subquery": self.subquery,
			"answer": self.answer,
			"evidences": self.evidences,
			"confidence": self.confidence,
			"acc_score": self.acc_score,
		}


@dataclass
class MultiHopDatum:
	query: str									# The full multi-hop question
	subqueries: List[SubQueryDatum] = field(default_factory=list)

	def add_subquery(self, subquery: SubQueryDatum):
		self.subqueries.append(subquery)

	def to_dict(self):
		return {
			"query": self.query,
			"subqueries": [sq.to_dict() for sq in self.subqueries]
		}



from typing import Optional

def find_backtrack_index(example: MultiHopDatum, current_idx: int) -> Optional[int]:
	"""
	Backtracking rule:
	1) If current subquery's acc_score >= 0.75, return None (no backtracking).
	2) If current subquery's acc_score < 0.75, scan backward from current_idx - 1 to 0.
	3) Return the first index encountered whose acc_score < 0.8.
	4) If no previous subquery satisfies acc_score < 0.8, return current_idx
	   (since it's the earliest subquery that satisfies the condition).
	"""
	# Validate index
	if current_idx < 0 or current_idx >= len(example.subqueries):
		raise IndexError("current_idx out of range")

	current_sq = example.subqueries[current_idx]
	current_score = current_sq.acc_score if current_sq.acc_score is not None else 0.0

	# Only backtrack if current subquery is weak (< 0.75)
	if current_score >= 0.75:
		return None

	# Scan backward from current_idx - 1 down to 0
	for i in range(current_idx - 1, -1, -1):
		sq = example.subqueries[i]
		score = sq.acc_score
		if score is None:
			print("Score is None!!!!!!")
			score = 0.0
		if score < 0.8:
			return i

	# No earlier subquery satisfies acc_score < 0.8
	# → current subquery is the earliest that does
	return current_idx