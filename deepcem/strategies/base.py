from abc import ABC, abstractmethod

from deepcem.config import PipelineConfig
from deepcem.data_structures import Cluster, Hyperedge, Reference
class Strategy(ABC):
   @abstractmethod
   def calculate_cluster_similarity(self, clusters: dict[str, Cluster], parents, ci: str, cj: str, ri: Reference, rj: Reference, references: dict[str, Reference], hyperedges: dict[str, Hyperedge], cfg: PipelineConfig) -> float:
      pass