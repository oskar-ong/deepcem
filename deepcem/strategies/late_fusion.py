import logging
import numpy as np

from deepcem.config import PipelineConfig
from deepcem.data_structures import Cluster, Hyperedge, Reference, get_cluster
from deepcem.ditto_utils import serialize_ditto
from deepcem.similarity import RelationalSimilarity, choose_rel_similarity_measure
from deepcem.strategies.base import Strategy
from deepcem.utils import cluster_pair
from matcher import load_model, classify, DittoModel 

logger = logging.getLogger("cem.late_fusion")

class LateFusion(Strategy):
    def __init__(self):
        self.ditto_model: DittoModel = ""
        self.rel_sim: RelationalSimilarity = ""
        self._cache: dict[tuple[str, str], float] = {}

    def get_ditto_model(self, cfg: PipelineConfig):
        if not self.ditto_model: 
            _, self.ditto_model = load_model(f"{cfg.dataset}", cfg.ckpt, cfg.lm, cfg.use_gpu, cfg.fp16)

        return self.ditto_model

    def get_rel_sim(self, cfg: PipelineConfig):
        if not self.rel_sim:
            self.rel_sim = choose_rel_similarity_measure(cfg.rel_similarity)
        return self.rel_sim

    def calculate_cluster_similarity(self, clusters: dict[str, Cluster], parents, ci: str, cj: str, references: dict[str, Reference], hyperedges: dict[str, Hyperedge], cfg: PipelineConfig ):

        # compare all references of ci with all references of cj
        lines = []
        comparisons = []
        old_scores = []
        # iterate over all references of ci

        c_i = get_cluster(ci, clusters,parents)
        c_j = get_cluster(cj, clusters,parents)
        for ri in c_i.references:
            # iterate over all references in cj
            for rj in c_j.references:
                logger.debug(f"serialize Ditto: ci: {ci}, ri: {ri}, cj: {cj}, rj: {rj}")

                if ri != rj:
                    if cluster_pair(ri,rj) in self._cache:
                        old_scores.append(self._cache[cluster_pair(ri,rj)])
                    else:
                        lines.append(serialize_ditto(
                            references[ri], references[rj]))
                        comparisons.append(cluster_pair(ri,rj))
                
        labels, scores = classify(lines, self.get_ditto_model(cfg), lm=cfg.lm)

        for key,value in zip(comparisons,scores):
            self._cache[key] = value

        scores.extend(old_scores)
        # find highest score in scores
        highest_score = 0
        for score in scores:
            probs = np.exp(score) / np.sum(np.exp(score))
            # scores[1]: The logit for class 1 "match"
            if probs[1].item() >= highest_score:
                highest_score = probs[1].item()
        sim_a = highest_score
        logger.debug(f"ditto score: {sim_a}")

        # sim.r(ci,cj)
        sim_r = self.get_rel_sim(cfg).calculate(
            get_cluster(ci, clusters, parents), get_cluster(cj, clusters, parents), hyperedges, references)
        logger.debug(f"sim_r score: {sim_r}")
        # calculate sim(ci,cj)
        sim_ci_cj = (1-cfg.alpha) * sim_a + cfg.alpha * sim_r
        return sim_ci_cj