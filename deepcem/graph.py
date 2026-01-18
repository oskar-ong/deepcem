from collections import defaultdict

import pandas as pd

from deepcem.data_structures import Reference, Hyperedge


def create_hyperedges(references: dict[str, Reference], relationships):
    """
    Create all Hyperedges

    Args:
        references (dict[str, Reference]): All References
        relationships (list[tuple(source, relation, target)]): Each tuple is (source, relation, target).

    Returns:
        hyperedges (dict[str, Hyperedge]): All Hyperedges
    """

    # Group sources by (relation, source) to create hyperedges
    grouped_sources = defaultdict(list)
    for source, relation, target in relationships:
        # Actor - actsIn - movie
        edge_id = (source, relation)  # Actor, actsIn
        grouped_sources[(source, relation)].append(target)

        # old
        # edge_id = (source, relation)  # Actor, actsIn
        # grouped_sources[(source, relation)].append(target)

    # Create Hyperedges
    hyperedges: dict[str, Hyperedge]
    hyperedges = {}
    for i, ((source, relation), targets) in enumerate(grouped_sources.items()):
        edge_id = f"e{i}"
        # Hyperedge connects all sources + the target

        edge_references = set()
        for t in targets:
            if t not in references:
                references[t] = Reference(t)

            references[t].hyperedges.add(edge_id)
            edge_references.add(t)

        h = Hyperedge(edge_id, {"relation": relation, "source": source})
        h.references.update(edge_references)
        hyperedges[edge_id] = h
    return hyperedges, references

from collections import defaultdict

def create_hyperedges_from_author_to_pubs(
    references: dict[str, "Reference"],
    author_to_publications: dict[str, list[str]],
    relation: str = "writes",
    min_targets: int = 2,
):
    """
    Create hyperedges from an author -> publications table.

    One hyperedge per author (source), connecting all publications (targets).

    Args:
        references: dict of Reference objects keyed by reference id (publication ids live here).
        author_to_publications: dict[author_id -> list[pub_id]] (author clusters recommended).
        relation: relation name stored in hyperedge attributes (default: "writes").
        min_targets: skip authors with fewer than this many publications (default: 2).

    Returns:
        (hyperedges, references)
    """
    hyperedges: dict[str, "Hyperedge"] = {}

    i = 0
    for author_id, pubs in author_to_publications.items():
        # Deduplicate pubs and (optionally) filter out tiny hyperedges
        targets = list({p for p in pubs if p})
        if len(targets) < min_targets:
            continue

        edge_id = f"e{i}"
        i += 1

        edge_references = set()
        for pub_id in targets:
            if pub_id not in references:
                references[pub_id] = Reference(pub_id)

            references[pub_id].hyperedges.add(edge_id)
            edge_references.add(pub_id)

        h = Hyperedge(edge_id, {"relation": relation, "source": author_id})
        h.references.update(edge_references)
        hyperedges[edge_id] = h

    return hyperedges, references


def extract_references(candidates: pd.DataFrame) -> dict[str, Reference]:
    # All References
    references: dict[str, Reference]
    references = {}

    for _, row in candidates.iterrows():
        # Left node
        left_id = row[0]["id"]
        node_left = row[0]

        # old input format
        #left_id = row["ltable.Id"]
        #node_left = {col.replace("ltable.", ""): row[col]
        #             for col in row.index if col.startswith("ltable.")}

        if left_id not in references:
            references[left_id] = Reference(left_id)
            references[left_id].attrs = node_left

        # Right node
        right_id = row[1]["id"]
        node_right = row[1]

        #old input format
        # right_id = row["rtable.Id"]
        # node_right = {col.replace("rtable.", ""): row[col]
        #               for col in row.index if col.startswith("rtable.")}
        
        if right_id not in references:
            references[right_id] = Reference(right_id)
            references[right_id].attrs = node_right

        references[left_id].similar_references.add(right_id)
        references[right_id].similar_references.add(left_id)

    return references
