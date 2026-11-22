"""Pure prewrite selection. Calibration/evaluator outcomes are never inputs."""
def action(candidate,score,threshold=0.5):
 if candidate is None:return "insufficient"
 return "attempt" if score>=threshold else "reject"
def select(rows,coverage=0.5,score="candidate_change"):
 # Fixed budget counts all eligible; missing candidates are never written.
 k=int(len(rows)*coverage)
 ordered=sorted(rows,key=lambda r:r["uid"])
 ordered.sort(key=lambda r:r[score] if r.get("candidate") else float("-inf"),reverse=True)
 chosen={r["uid"] for r in ordered[:k] if r.get("candidate")}
 return {r["uid"]:("attempt" if r["uid"] in chosen else "insufficient" if not r.get("candidate") else "reject") for r in rows}
