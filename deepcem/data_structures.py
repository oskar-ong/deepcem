class Reference:
    def __init__(self, id):
        self.id = id
        self.cluster = '' #self.id
        self.attrs = {} 
        self.hyperedges: set[str] = set()
        self.similar_references: set[str] = set()

class Hyperedge:
    def __init__(self, id, attribute):
        self.id = id
        self.references: set[str] = set()
        self.attributes = dict()
        self.attributes = attribute
        self.relation_type = attribute["relation"] 

class Cluster:
    def __init__(self, id):
        self.id = id
        self.references: set[str] = set()
        self.hyperedges: set[str] = set()
        self.similar_clusters: set[str] = set() # For a cluster that has a single reference r, the similar clusters are those that contain references in the same bucket as r after blocking.
        self.neighboring_clusters: set[str] = set() # neighbor: clusters containing references that share a hyper-edge with r
        self.pq_entries: dict[tuple[str,str],float] = dict()

    def __repr__(self):
        return f"Cluster(id='{self.id}')"

def merge_clusters(c1: Cluster, c2: Cluster, clusters: dict, parents:dict) -> tuple[Cluster, dict, dict]:
    """Merge two clusters into a new one and update mappings recursively."""
    rep1, rep2 = find(c1.id, parents), find(c2.id, parents)
    new_id = rep1 + rep2
    new_cluster = Cluster(new_id)

    #references
    new_cluster.references.update(c1.references)
    new_cluster.references.update(c2.references)
    #hyperedges
    new_cluster.hyperedges.update(c1.hyperedges)
    new_cluster.hyperedges.update(c2.hyperedges)
    #sim clusters
    new_cluster.similar_clusters.update(c1.similar_clusters)
    new_cluster.similar_clusters.update(c2.similar_clusters)
    if c1.id in new_cluster.similar_clusters:
        new_cluster.similar_clusters.remove(c1.id)
    if c2.id in new_cluster.similar_clusters:
        new_cluster.similar_clusters.remove(c2.id)
    #neighbor clusters
    new_cluster.neighboring_clusters.update(c1.neighboring_clusters)
    new_cluster.neighboring_clusters.update(c2.neighboring_clusters)

    parents[rep1] = new_id
    parents[rep2] = new_id
    parents[new_id] = new_id
    # store under its own id
    clusters[new_id] = new_cluster

    return new_cluster, clusters, parents

def find(cid, parents):
    """Find representative id with path compression."""
    if parents[cid] != cid:
        parents[cid] = find(parents[cid], parents)  # path compression
    return parents[cid]

def get_cluster(cid, clusters, parents):
    """Return the current cluster object for a given id."""
    rep = find(cid, parents)
    return clusters[rep]

def make_cluster(cid, parents, clusters):
    """Create a fresh cluster and initialize union-find entry."""
    c = Cluster(cid)
    parents[cid] = cid
    clusters[cid] = c
    return c, parents, clusters