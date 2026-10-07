from dataclasses import dataclass, field
import pandas as pd

@dataclass
class Node:
  obj: object
  edges: list = field(default_factory= lambda: [])

@dataclass
class Edge:
  theta: float # -pi~pi
  d: float
  node: Node

@dataclass
class StructuredVideo:
  start_nodes: list # of nodes
  name: str # name of the video

@dataclass
class VideoRepository:
  videos: list # of StructuredVideo

@dataclass
class TemporalPattern:
  nodes: pd.DataFrame
  edges: pd.DataFrame
  length: int
