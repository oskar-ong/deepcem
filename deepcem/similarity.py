from typing import List
from deepcem.data_structures import Cluster, Hyperedge

class RelationalSimilarity():
    def __init__(self):
        pass

    def calculate(self, neighborhood_a, neighborhood_b, hyperedges):
        similarity = 0
        return similarity


class JaccardCoefficient(RelationalSimilarity):
    def calculate(self, ci: Cluster, cj: Cluster, hyperedges: dict[str, List[str]]):
        # JaccardCoef f (ci, cj ) = |N br(ci) ⋂ N br(cj )|  |N br(ci) ⋃ N br(cj )|
        neighborhood_a = set(nbr(ci, hyperedges))
        neighborhood_b = set(nbr(cj, hyperedges))

        # if any of the neighborhoods is empty return 0
        if not neighborhood_a or not neighborhood_b:
            return 0.0

        intersection_nbr_a_b = neighborhood_a & neighborhood_b
        union_nbr_a_b = neighborhood_a | neighborhood_b

        similarity = len(intersection_nbr_a_b) / len(union_nbr_a_b)
        return similarity

def nbr(c: Cluster, hyperedges: dict[str, List[str]]):

    neighborhood = []
    for r_id in c.references: 
        for a in hyperedges.get(r_id, []):
            neighborhood.append(a)
    return neighborhood

def choose_rel_similarity_measure(name):
    if name == "jaccard_coefficient":
        return JaccardCoefficient()
    else:
        raise ValueError(f"Relational similarity error {name} not implemented")

