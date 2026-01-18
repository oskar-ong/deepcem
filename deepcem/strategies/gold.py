from deepcem.data_structures import get_cluster
from deepcem.strategies.base import Strategy

# class Gold(Strategy):
#     def __init__(self):
#         self.pairs = {}

#     def set_pairs(self, pairs):
#         for line in pairs:
#             self.pairs[(line[0], line[1])] = line[2]

#     def calculate_cluster_similarity(self, clusters, parents, ci, cj):
        
#         c_i = get_cluster(ci, clusters, parents)
#         c_j = get_cluster(cj, clusters, parents)
#         for ri in c_i.references:
#             for rj in c_j.references:
#                 if self.pairs[(ri,rj)] == 1:
#                     return 1
#         return 0
        
class Gold(Strategy):
    def __init__(self):
        # Using a set of frozensets for bi-directional lookup and speed
        self.positive_pairs = set()

    def set_pairs(self, pairs_list):
        """
        Expects a list of tuples: (id_a, id_b, label)
        Only stores positive matches to keep the lookup set small.
        """
        for left_dict, right_dict, label in pairs_list:
            if label == 1:
                # Use a frozenset so (A, B) is the same as (B, A)
                pair_key = frozenset([left_dict['id'], right_dict['id']])
                self.positive_pairs.add(pair_key)

    def calculate_cluster_similarity(self, clusters, parents, ci, cj):
        # 1. Fetch clusters ONCE outside the loops
        cluster_i_refs = get_cluster(ci, clusters, parents).references
        cluster_j_refs = get_cluster(cj, clusters, parents).references
        
        # 2. Short-circuit: Exit as soon as a single match is found
        for ri in cluster_i_refs:
            for rj in cluster_j_refs:
                if frozenset([ri, rj]) in self.positive_pairs:
                    return 1
                    
        return 0



        