from abc import ABC, abstractmethod

from deepcem.config import AppConfig
from deepcem.data_structures import Cluster, Hyperedge, Reference
class Strategy(ABC):
   @abstractmethod
   def calculate_cluster_similarity(self, clusters: dict[str, Cluster], parents, ci: str, cj: str, references: dict[str, Reference], hyperedges: dict[str, Hyperedge], cfg: AppConfig) -> float:
      pass