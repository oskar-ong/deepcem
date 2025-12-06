import numpy as np

from deepcem.config import PipelineConfig
from deepcem.data_structures import Cluster, Hyperedge, Reference, get_cluster
from deepcem.ditto_utils import serialize_ditto
from deepcem.similarity import nbr
from deepcem.strategies.base import Strategy

class InlineEncode(Strategy):
    def __init__(self):
        self.ditto_model: DittoModel

    def get_ditto_model(self, cfg: PipelineConfig):
        if not self.ditto_model: 
            _, self.ditto_model = load_model(f"{cfg.dataset}_extracted", cfg.ckpt, cfg.lm, cfg.use_gpu, cfg.fp16)

        return self.ditto_model
    
    def calculate_record_similarity(self, clusters: dict[str, Cluster], parents, ci: str, cj: str, ri: Reference, rj: Reference, references: dict[str, Reference], hyperedges: dict[str, Hyperedge], cfg: PipelineConfig):
        inline_encode_option = ""

        if inline_encode_option == "top-k":

            # get neighborhood of ci

            neighbors_ci = nbr(ci, hyperedges, references)

            neighbors_ci_sorted = sorted(neighbors_ci, key=lambda c: c.id)
            top_k_neighbors_ci = neighbors_ci_sorted[cfg.top_k]



            # get top-k references of cluster 
            for ri in get_cluster(ci, clusters, parents).references:
            # iterate over all references in cj
                for rj in get_cluster(cj, clusters, parents).references:
                    break 

        return True

    def calculate_cluster_similarity(self, clusters: dict[str, Cluster], parents, ci: str, cj: str, ri: Reference, rj: Reference, references: dict[str, Reference], hyperedges: dict[str, Hyperedge], cfg: PipelineConfig ):
        
        neighbors_ci = nbr(ci, hyperedges, references)

        neighbors_ci_sorted = sorted(neighbors_ci, key=lambda c: c.id)
        top_k_neighbors_ci = neighbors_ci_sorted[cfg.top_k]

        # get top-k references of cluster 
        for ri in get_cluster(ci, clusters, parents).references:
        # iterate over all references in cj
            for rj in get_cluster(cj, clusters, parents).references:
                break 


        # compare all references of ci with all references of cj
        lines = []
        # iterate over all references of ci
        for ri in get_cluster(ci, clusters, parents).references:
            # iterate over all references in cj
            for rj in get_cluster(cj, clusters, parents).references:

                lines.append(serialize_ditto(
                    references[ri], references[rj]))
                
        labels, scores = classify(lines, self.get_ditto_model(cfg), lm=cfg.lm)

        # find highest score in scores
        highest_score = 0
        for score in scores:
            probs = np.exp(score) / np.sum(np.exp(score))
            # scores[1]: The logit for class 1 "match"
            if probs[1].item() >= highest_score:
                highest_score = probs[1].item()
        
        sim_ci_cj = highest_score

        return sim_ci_cj