from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x], self.rank[x] = x, 0

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while x != root:
            p = self.parent[x]
            self.parent[x], x = root, p
        return root

    def union(self, a: str, b: str) -> None:
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class processedEntity:
    uf: UnionFind
    df: pd.DataFrame
    pairs: Dict[str, List[Tuple[str, str, int]]]
    cp: List
    dfs_by_pollution: Dict[str, pd.DataFrame]
