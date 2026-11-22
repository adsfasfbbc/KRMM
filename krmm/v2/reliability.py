from .reconstruct import reconstruct
from .ontology import Fact
FEATURES=["support","ambiguity","conflict","key_sensitivity","sham_stability","provenance"]
def features(facts):
 full=reconstruct(facts);c=full["candidate"]
 reduced=[reconstruct(facts[:i]+facts[i+1:]) for i in range(len(facts))]
 key=max([float(c is not None and r["candidate"]!=c) for r in reduced] or [0])
 # Adding a known irrelevant material observation is a real recomputation, not a constant mock.
 sham=Fact(facts[0].entity if facts else "unknown","material","wood","control","sham","control","wood","The unrelated storage box is wood.")
 with_sham=reconstruct(facts+[sham])
 independent=len(set(f.provenance_group for f in facts))
 return [float(c is not None),max(0,len(full["compatible"])-1)/3,float(full["conflict"]),key,float(with_sham["candidate"]==c),independent/max(1,len(facts))]
def fit_ere(X,y):
 from sklearn.linear_model import LogisticRegression
 from sklearn.dummy import DummyClassifier
 if len(set(y))<2:return DummyClassifier(strategy="prior").fit(X,y)
 return LogisticRegression(C=1,class_weight=None,random_state=107,max_iter=1000).fit(X,y)
def probability(model,X):
 import numpy as np
 p=model.predict_proba(X)
 return p[:,list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(X))
