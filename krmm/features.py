"""Pre-update features only, shared by offline analysis and answer-free deployment."""
import numpy as np
DERIVED_FEATURE='candidate_changes_direct_answer'
def vector(diagnosis):
    direct='ABCD'[int(np.argmax(diagnosis['direct'][0]))]
    return list(diagnosis['features'])+[float(diagnosis['candidate']!=direct)]
