import logging
from typing import List
import numpy as np

from deepcem.utils import bucket, cluster_pair
from matcher import DittoModel

from deepcem.config import AppConfig
from deepcem.data_structures import Cluster, Reference, get_cluster
from deepcem.ditto_utils import serialize_record
from deepcem.similarity import nbr
from deepcem.strategies.base import Strategy
from matcher import classify, load_model

logger = logging.getLogger("cem.inline_encoded")

class InlineEncode(Strategy):
    def __init__(self):
        self.ditto_model: DittoModel = None
        self._cache = {}

    def get_ditto_model(self, cfg: AppConfig):
        if not self.ditto_model: 
            _, self.ditto_model = load_model(f"{cfg.finetune.task}", str(cfg.run.checkpoints_root), cfg.finetune.encoder, cfg.algo.use_gpu, cfg.algo.fp16) #load model concatenates root + task

        return self.ditto_model
    
    def calculate_cluster_similarity(self, clusters: dict[str, Cluster], parents, ci: str, cj: str, references: dict[str, Reference], hyperedges: dict[str, List[str]], cfg: AppConfig ):

        # compare all references of ci with all references of cj
        lines = []
        # iterate over all references of ci
        c_i = get_cluster(ci, clusters,parents)
        c_j = get_cluster(cj, clusters,parents)

        # add neighborhoods to dictionary and create lines for ditto 
        # get neighborhood for each

        #TODO: ORDER? FIRST AUTHOR? 
        # TODO: ADD MORE METADATA FOR TOP NEIGHBORS? AUTHOR: - EMAIL; AGE;
        # TODO: MULTIPLICITY, FREQUENCY
        # TODO TWO HOP SUMMARIES? 
        # TODO: SHARED NEIGHBORS (AUTHORS) as attribute
        nbr_ci: List[str] = nbr(c_i, hyperedges)
        nbr_cj: List[str] = nbr(c_j, hyperedges)

        def top_k(nbr, k=5):
            seen = set()
            result = []
            for x in nbr: 
                if x not in seen:
                    seen.add(x)
                    result.append(x)
                if len(result) >= k:
                    break
            return result

        nbr_ci_list = top_k(nbr_ci, 5)
        nbr_cj_list = top_k(nbr_cj, 5)

        A = set(nbr_ci)
        B = set(nbr_cj)

        inter = len(A & B)
        union = len(A | B)
        minsz = min(len(A), len(B))

        jaccard = (inter / union) if union else 0.0
        overlap = (inter / minsz) if minsz else 0.0
        containL = (inter / len(A)) if len(A) else 0.0
        containR = (inter / len(B)) if len(B) else 0.0
                
        weighted_jaccard = 0 # TODO
        cosine_sim = 0 # TODO
        rare_overlap_count = 0 # TODO
        
        nbr_ci_str =  f"{' | '.join(nbr_ci_list)}"
        nbr_cj_str =  f"{' | '.join(nbr_cj_list)}"

        # Bucket
        jac = bucket(jaccard)
        # pairwise_summary = f"jac={jaccard:.3f} overlap={overlap:.3f} containL={containL:.3f} containR={containR:.3f}"
        pairwise_summary = f"jac={jac}"

        for ri in c_i.references:
            # iterate over all references in cj
            for rj in c_j.references:
                if ri != rj:
                    # format: 
                    ri_fields = dict(references[ri].attrs)
                    rj_fields = dict(references[rj].attrs)

                    ri_fields[cfg.dataset.context_field] = nbr_ci_str
                    rj_fields[cfg.dataset.context_field] = nbr_cj_str

                    ri_fields["pairwise_summary"] = pairwise_summary
                    rj_fields["pairwise_summary"] = pairwise_summary

                    # [ref_left.attrs, nbr_left_cluster] [ref_right.attrs, nbr_right_cluster]
                    left = serialize_record(ri_fields)
                    right = serialize_record(rj_fields)

                    lines.append(f"{left}\t{right}\t0")
 
        labels, scores = classify(lines, self.get_ditto_model(cfg), lm=cfg.finetune.encoder)

        # find highest score in scores
        highest_score = 0
        for score in scores:
            probs = np.exp(score) / np.sum(np.exp(score))
            # scores[1]: The logit for class 1 "match"
            if probs[1].item() >= highest_score:
                highest_score = probs[1].item()
        sim = round(highest_score,3)
        logger.debug(f"Similarity Score for {ci} - {cj}: {sim}")

        return sim