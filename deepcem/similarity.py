from deepcem.data_structures import Cluster, Reference, Hyperedge

from collections import defaultdict
from typing import Dict, Set


class RelationalSimilarity():
    def __init__(self):
        pass

    def calculate(self, neighborhood_a, neighborhood_b):
        similarity = 0
        return similarity


class JaccardCoefficient(RelationalSimilarity):
    def calculate(self, ci: Cluster, cj: Cluster, hyperedges):
        # JaccardCoef f (ci, cj ) = |N br(ci) ⋂ N br(cj )|  |N br(ci) ⋃ N br(cj )|
        neighborhood_a = nbr(ci, hyperedges)
        neighborhood_b = nbr(cj, hyperedges)

        # if any of the neighborhoods is empty return 0
        if not neighborhood_a or not neighborhood_b:
            return 0.0

        intersection_nbr_a_b = neighborhood_a & neighborhood_b
        union_nbr_a_b = neighborhood_a | neighborhood_b

        similarity = len(intersection_nbr_a_b) / len(union_nbr_a_b)
        return similarity

def nbr(c: Cluster, hyperedges: dict[str, Hyperedge]):

    neighborhood = set()
    for r_id in c.references: 
        for a in hyperedges.get(r_id, []):
            neighborhood.add(a)
    return neighborhood


def typed_neighbor_clusters(
    ci: Cluster,
    hyperedges: Dict[str, Hyperedge],
    references: Dict[str, Reference],
) -> Dict[str, Set[Cluster]]:
    """
    Group neighbor clusters of ci by hyperedge relation_type.
    Returns: {relation_type -> set of neighbor Clusters}
    """
    typed_neighbors: Dict[str, Set[Cluster]] = defaultdict(set)

    for h_id in ci.hyperedges:
        h = hyperedges[h_id]
        rel_type = h.relation_type 

        for r_id in h.references:
            cj = references[r_id].cluster
            if cj is ci:
                continue  

            typed_neighbors[rel_type].add(cj)

    return typed_neighbors

def cluster_stats(
    ci: Cluster,
    hyperedges: Dict[str, Hyperedge],
    references: Dict[str, Reference],
) -> Dict[str, float]:
    stats = {}

    stats["cluster_size"] = len(ci.references)
    stats["hyperedges_count"] = len(ci.hyperedges)

    typed_neighbors = typed_neighbor_clusters(ci, hyperedges, references)

    # total degree
    all_neighbors = set()
    for clusters in typed_neighbors.values():
        all_neighbors.update(clusters)
    stats["cluster_degree_total"] = len(all_neighbors)

    # per-type degrees
    for rel_type, clusters in typed_neighbors.items():
        stats[f"cluster_degree_{rel_type}"] = len(clusters)

    return stats

def pairwise_neighbor_stats(
    ci: Cluster,
    cj: Cluster,
    hyperedges: Dict[str, Hyperedge],
    references: Dict[str, Reference],
) -> Dict[str, float]:
    stats: Dict[str, float] = {}

    ti = typed_neighbor_clusters(ci, hyperedges, references)
    tj = typed_neighbor_clusters(cj, hyperedges, references)

    # all neighbors (regardless of type)
    ni_all = set().union(*ti.values()) if ti else set()
    nj_all = set().union(*tj.values()) if tj else set()

    inter_all = ni_all & nj_all
    union_all = ni_all | nj_all

    stats["neigh_overlap_total"] = float(len(inter_all))
    stats["neigh_jaccard_total"] = float(len(inter_all)) / len(union_all) if union_all else 0.0

    # per-type overlaps/Jaccards
    all_types = set(ti.keys()) | set(tj.keys())
    for rel_type in all_types:
        ni_t = ti.get(rel_type, set())
        nj_t = tj.get(rel_type, set())
        inter_t = ni_t & nj_t
        union_t = ni_t | nj_t

        stats[f"neigh_overlap_{rel_type}"] = float(len(inter_t))
        stats[f"neigh_jaccard_{rel_type}"] = float(len(inter_t)) / len(union_t) if union_t else 0.0

    return stats

def choose_rel_similarity_measure(name):
    if name == "jaccard_coefficient":
        return JaccardCoefficient()
    else:
        raise ValueError(f"Relational similarity error {name} not implemented")

# def calculate_cluster_similarity(lines, model, conf, ci, cj, rel_sm, clusters, parents, hyperedges, references, alpha) -> float:
#     # sim.a
#     # randomseed? runid is seed
#     #print(lines)

#     labels, scores = classify(
#         lines, model, lm=conf.lm)
#     # find highest score in scores
#     highest_score = 0
#     for score in scores:
#         probs = np.exp(score) / np.sum(np.exp(score))
#         # scores[1]: The logit for class 1 "match"
#         if probs[1].item() >= highest_score:
#             highest_score = probs[1].item()
#     sim_a = highest_score

#     # sim.r(ci,cj)
#     sim_r = rel_sm.calculate(
#         get_cluster(ci, clusters, parents), get_cluster(cj, clusters, parents), hyperedges, references)

#     # calculate sim(ci,cj)
#     sim_ci_cj = (1-alpha) * sim_a + alpha * sim_r
#     return sim_ci_cj
