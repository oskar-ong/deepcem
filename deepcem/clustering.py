from collections import defaultdict
import logging

import pandas as pd
from tqdm import tqdm

from deepcem.config import PipelineConfig
from deepcem.data_structures import Reference, Hyperedge, Cluster, get_cluster, make_cluster, merge_clusters, find
from deepcem.strategies.factory import strategy_factory
from deepcem.utils import cluster_pair

logger = logging.getLogger("cem.clustering")
deleted_pq_entries = set() # debug 
class PriorityQueue:
    def __init__(self):
        self.heap = []               # list of tuples: (priority, item)
        self.position_map = {}       # item -> index in heap

    def add(self, priority, item):
        """Add a new item or update priority if it exists."""
        if item in self.position_map:
            self.update_priority(item, priority)
        else:
            entry = [priority, item]
            self.heap.append(entry)
            idx = len(self.heap) - 1
            self.position_map[item] = idx
            self._sift_up(idx)

    def update_priority(self, item, new_priority):
        """Update the priority of an existing item."""
        if item not in self.position_map:
            logger.info(item)
            print(deleted_pq_entries)
            raise KeyError(f"Item {item} not found in priority queue")
        idx = self.position_map[item]
        old_priority = self.heap[idx][0]
        self.heap[idx][0] = new_priority
        if new_priority < old_priority:
            self._sift_up(idx)
        else:
            self._sift_down(idx)

    def remove(self, item):
        """Remove an item from the priority queue."""
        if item not in self.position_map:
            raise KeyError(f"Item {item} not found in priority queue")
        idx = self.position_map[item]
        last_idx = len(self.heap) - 1
        if idx != last_idx:
            self._swap(idx, last_idx)
        _, removed_item = self.heap.pop()
        del self.position_map[removed_item]
        if idx < len(self.heap):
            # Re-heapify
            self._sift_up(idx)
            self._sift_down(idx)

    def pop(self):
        """Remove and return the item with the lowest priority."""
        if not self.heap:
            raise KeyError("pop from an empty priority queue")
        last_idx = len(self.heap) - 1
        self._swap(0, last_idx)
        priority, item = self.heap.pop()
        del self.position_map[item]
        if self.heap:
            self._sift_down(0)
        return priority, item
    
    def peek(self, n=1):
        """Return the next n (priority, item) pairs without removing them."""
        if n <= 0:
            return []
        if not self.heap:
            return []

        # Copy heap and pop from copy
        import heapq
        temp = self.heap.copy()
        heapq.heapify(temp)

        result = []
        for _ in range(min(n, len(temp))):
            priority, item = heapq.heappop(temp)
            result.append((priority, item))
        return result

    def _sift_up(self, idx):
        while idx > 0:
            parent = (idx - 1) // 2
            if self.heap[idx][0] < self.heap[parent][0]:
                self._swap(idx, parent)
                idx = parent
            else:
                break

    def _sift_down(self, idx):
        n = len(self.heap)
        while True:
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest != idx:
                self._swap(idx, smallest)
                idx = smallest
            else:
                break

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.position_map[self.heap[i][1]] = i
        self.position_map[self.heap[j][1]] = j

    def __bool__(self):
        return bool(self.heap)


def exact_match(df):

    # # FOR TESTING - TODO: REMOVE
    # df = pd.DataFrame([

    #     [{'Id': 1, 'Name': 'Max', 'Age': 30}, {'Id': 99, 'Name': 'Max', 'Age': 30}, {0}],
    #     [{'Id': 3, 'Name': 'Lia', 'Age': 25}, {'Id': 3, 'Name': 'Lia', 'Age': 26}, {0}],
    #     [{'Id': 5, 'Name': 'Tom', 'Age': 40}, {'Id': 5, 'Name': 'Tom', 'Age': 40}, {0}]
    # ])
    # # END TESTING
    df
    def compare_dicts(row):
        d1, d2, _ = row

        #keys = set(d1) & set(d2) - {'Id', 'Year', 'RatingValue'}
        keys = set(d1) & set(d2) 
        return all(d1[k] == d2[k] for k in keys)
    
    df['is_equal'] = df.apply(compare_dicts, axis = 1)

    result = df[df['is_equal']].copy()

    result = pd.DataFrame([
        {'r1_id': d[0]['Id'], 'r2_id': d[1]['Id']}
        for _, d in result.iterrows()
    ])

    return result


def hyperedge_k_match(candidates, references: dict[str, Reference], hyperedges: dict[str, Hyperedge], k):

    matches = []
    for _, row in candidates.iterrows():
        # get r1,r2:
        r1, r2 = row[0]['Id'], row[1]['Id']
        matched = False

        # get all hyperedges associated with r1
        # if r1 is not associated with any hyperedges, return []
        reference_r1 = references.get(r1)
        hyperedges_r1 = reference_r1.hyperedges

        # get all hyperedges associated with r2
        reference_r2 = references.get(r2)
        hyperedges_r2 = reference_r2.hyperedges

        # iterate over each edge of each hyperedge r1 participates in
        for edge_id1 in hyperedges_r1:
            edge_hr1 = hyperedges[edge_id1]
            # iterate over each edge of each hyperedge r2 participates in
            for edge_id2 in hyperedges_r2:
                edge_hr2 = hyperedges[edge_id2]
                # count amount of references that match all attributes
                amt_exact_matches = 0
                # iterate over each reference in edge_hr1
                for n_ehr1 in edge_hr1.references:
                    # get attrs of n_ehr1
                    attrs_nehr1 = references[n_ehr1].attrs
                    # attrs_nehr1 = H.nodes[n_ehr1].properties
                    # iterate over each node in edge_hr2
                    for n_ehr2 in edge_hr2.references:
                        # check origin of node nehr1
                        # attrs_nehr2 = H.nodes[n_ehr2].properties
                        attrs_nehr2 = references[n_ehr2].attrs
                        # if all attributes of n_ehr1 match all attributes of n_ehr2
                        if do_all_attributes_match(n_ehr1, n_ehr2, attrs_nehr1, attrs_nehr2):
                            amt_exact_matches += 1

                        if amt_exact_matches >= k:
                            matches.append((r1, r2))
                            matched = True
                            break

                    if matched == True:
                        break
                if matched == True:
                    break
            if matched == True:
                break

    return matches


def do_all_attributes_match(r1: str, r2: str, attrs_nehr1, attrs_nehr2):
    if r1 != r2 and attrs_nehr1 == attrs_nehr2:
        return True
    else:
        return False


class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py

    def get_sets(self):
        groups = defaultdict(list)
        for x in self.parent:
            root = self.find(x)  # find representative
            groups[root].append(x)
        return list(groups.values())


def bootstrap_clusters(df, references: dict[str, Reference], hyperedges: dict, author_to_pubs) -> tuple[dict[str, Cluster], dict]:
    uf = UnionFind()

    # Every record is a distinct cluster
    for r in references:
        uf.add(r)

    # Initial Algorithm reduces search room by trying to merge clusters based on exact matches and hyperedge k-match
    # We intentionally DO NOT do this to ensure that every record is compared to every other record
    # Uncomment the following lines to activate exact match and hyperedge k-match, for example if the search room is very large
    # Merge all Exact matches

    # exact_matches = exact_match(df)
    # logger.debug(f"Length exact matches: {exact_matches.shape[0]}")
    # logger.debug(f"Length exact matches: {exact_matches.shape[0]}")
    # for _, row in exact_matches.iterrows():
    #     uf.union(row['r1_id'], row['r2_id'])

    # # Merge all Hyperedge k-exact matches
    # hyper_matches = hyperedge_k_match(df, references, hyperedges, k)
    # logger.debug(f"Length hyper matches: {len(hyper_matches)}")
    # logger.debug(f"Length hyper matches: {len(hyper_matches)}")
    # for hm in hyper_matches:
    #     uf.union(hm[0], hm[1])

    all_sets = uf.get_sets()

    # create cluster for each set
    clusters: dict[str, Cluster] = {}
    parents = {}

    for i, s in tqdm(enumerate(all_sets)):
        c_refs: set[str] = set()
        c_hyperedges: set[str] = set()
        for ref_id in s:
            c_refs.add(ref_id)
            #c_hyperedges.update(references[ref_id].hyperedges)
            references[ref_id].cluster = f"c{i}"

        _, parents, clusters = make_cluster(f"c{i}", parents, clusters)
        clusters[f"c{i}"].references.update(c_refs)
        clusters[f"c{i}"].hyperedges.update(c_hyperedges)

    for c in tqdm(clusters):
        # find similar clusters
        similar_clusters: set[str] = set()

        similar_refs: set[str] = set()
        c_hyperedges: set[str] = set()

        # similar clusters: Clusters that appear as potential matches in the candidate set

        for ref in clusters[c].references:
            similar_refs.update(references[ref].similar_references)
            #c_hyperedges.update(references[ref].hyperedges)
            if hyperedges.get(ref) != None:
                c_hyperedges.update(hyperedges.get(ref)) # list

        # logger.debug(f"Similar Refs: {similar_refs}")
        for sr in similar_refs:
            #logger.debug(f"Similar Ref: {sr}, Cluster: {references[sr].cluster}, current_cluster: {c}")
            if references[sr].cluster != c:
                similar_clusters.add(references[sr].cluster)
        clusters[c].similar_clusters.update(similar_clusters)
        # logger.debug(clusters[c].similar_clusters)

        # find neighbor cluster
        neighbor_clusters: set[str] = set()
        # clusters containing references that share a hyper-edge with r
        neighbor_refs: set[str] = set()
        # iterate over all hyperedges of the cluster
        for h in c_hyperedges:
            #neighbor_refs.update(hyperedges[h].references)
            # TODO: Debug 
            try:
                neighbor_refs.add(author_to_pubs[h]) # h can be a list
            except KeyError:
                raise Exception(f"Key Error in author_to_pubs for Key {h}")
        # find cluster of each neighbor ref
        for nr in neighbor_refs:
            if references[nr].cluster != c:
                neighbor_clusters.add(references[nr].cluster)
        clusters[c].neighboring_clusters.update(neighbor_clusters)

    return clusters, parents


def init_pq(clusters: dict[str, Cluster], references: dict[str, Reference], hyperedges, parents: dict, cfg: PipelineConfig):

    # Build priority queue of similarities
    # initialize empty list as priqority queue
    # max-heap simulated via negative similarity
    # most similar pair is popped first
    pq = PriorityQueue()

    # Initialize Priority Queue
    # i = index
    # ci = cluster object

    already_compared = set()
    logger.debug("Iterate over all Clusters")
    for ci in clusters:
        #logger.debug(f"ci: {ci}")
        # iterate over all similar clusters:
        for cj in clusters[ci].similar_clusters:
            #logger.debug(f"ci: {ci}, cj: {cj}")
            # dont self-compare and avoid duplicate comparisons (ci, cj) <-> (cj,ci)
            if ci != cj and cluster_pair(ci,cj) not in already_compared:
                
                sim_ci_cj = strategy_factory(cfg).calculate_cluster_similarity(clusters, parents, ci, cj, references, hyperedges, cfg)

                # Insert tuple into the heap
                # -sim : store negative similarity
                pq.add(-sim_ci_cj, cluster_pair(ci,cj))
                #logger.debug(f"cluster similarity: {-sim_ci_cj}, {ci}, {cj}")

                # store queue entries on each cluster
                clusters[ci].pq_entries[cluster_pair(ci,cj)] = -sim_ci_cj
                clusters[cj].pq_entries[cluster_pair(ci,cj)] = -sim_ci_cj

                # remember already compared
                already_compared.add(cluster_pair(ci,cj))
    return pq, clusters


def iterative_merge(pq: PriorityQueue, clusters: dict[str, Cluster], parents: dict, hyperedges, references: dict[str, Reference], cfg: PipelineConfig):
    iteration = 0

    strategy = strategy_factory(cfg)

    while pq:
        logger.debug(f"Current Iteration: {iteration}")
        iteration += 1

        neg_sim, (ci, cj) = pq.pop()
        ci, cj = cluster_pair(ci,cj)

        # skip stale entries
        rci = find(ci,parents)
        rcj = find(cj,parents)

        if rci != ci or rcj != cj:
            # stale clusters
            continue
        
        if rci == rcj:
            # stale cause already merged
            continue

        sim = -neg_sim
        logger.debug(f"Current Cluster: {(ci, cj)}, similarity: {sim}")
        if sim < cfg.threshold:
            logger.debug("Threshold reached, break")
            break

        # Merge clusters ci and cj
        cij, clusters, parents = merge_clusters(
            get_cluster(ci, clusters, parents), 
            get_cluster(cj, clusters, parents), 
            clusters, 
            parents)

        # Find all clusters "ck" that are similar to cij
        logger.debug(f"Iterate over all similar clusters of {cij.id}: {len(cij.similar_clusters)} Clusters")

        for ck_id in cij.similar_clusters:
            rk = find(ck_id,parents)
            if rk == cij.id:
                continue

            sim_cij_ck = strategy.calculate_cluster_similarity(clusters, parents, cij.id, rk, references, hyperedges, cfg)

            # insert sim(cij, ck), cij, ck into pq
            pq.add(-sim_cij_ck, cluster_pair(cij.id,rk))
            # add pqentry for cluster cij
            get_cluster(cij.id, clusters, parents).pq_entries[cluster_pair(cij.id,rk)] = -sim_cij_ck
            # add pqentry for cluster ck
            get_cluster(rk, clusters, parents).pq_entries[cluster_pair(cij.id,rk)] = -sim_cij_ck

        # Iterate over each neighbor cn of cij
        logger.debug(f"Iterate over all neighboring clusters of {cij.id}: {len(cij.neighboring_clusters)} Clusters")
        for cn_id in cij.neighboring_clusters:

            rn = find(cn_id, parents)
            if rn == cij.id:
                continue
            
            cn = get_cluster(rn,clusters,parents)

            # find all clusters "ck" that are similar to cn
            for ck_id in cn.similar_clusters:
                rk = find(ck_id,parents)
                if rk == rn:
                    continue

                sim_ck_cn = strategy.calculate_cluster_similarity(clusters, parents, rk, rn, references, hyperedges, cfg)

                # update q sim(ck, cn), ck, cn
                pq.add(-sim_ck_cn, cluster_pair(rn,rk))

                get_cluster(rk, clusters, parents).pq_entries[cluster_pair(rk,rn)] = -sim_ck_cn
                get_cluster(rn, clusters, parents).pq_entries[cluster_pair(rk,rn)] = -sim_ck_cn

    return clusters, references, hyperedges, parents
