from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class AutocompleteResponse(BaseModel):
    genes: List[str]
    drugs: List[str]
    diseases: List[str]

class RecommendationItem(BaseModel):
    rank: int
    drug: str
    score: float
    type: str
    indications: str

class MultiRecommendationItem(BaseModel):
    rank: int
    drug: str
    consensus_score: float
    methods_agreed: int
    type: str
    indications: str
    scores: Dict[str, float]

class GeneRecommendResponse(BaseModel):
    query: str
    method: str
    results: List[RecommendationItem]

class MultiRecommendResponse(BaseModel):
    query: str
    method: str
    results: List[MultiRecommendationItem]

class SimilarDrugItem(BaseModel):
    rank: int
    drug: str
    score: float
    indications: str

class DrugRecommendResponse(BaseModel):
    query: str
    method: str
    results: List[SimilarDrugItem]

class DiseaseGeneItem(BaseModel):
    gene: str
    score: float
    type: str

class DiseaseDrugItem(BaseModel):
    rank: int
    drug: str
    score: float
    type: str
    indications: str

class DiseaseRecommendResults(BaseModel):
    disease: str
    genes: List[DiseaseGeneItem]
    drugs: List[DiseaseDrugItem]

class DiseaseRecommendResponse(BaseModel):
    query: str
    method: str
    results: DiseaseRecommendResults

class NetworkNode(BaseModel):
    id: str
    label: str
    type: str

class NetworkLink(BaseModel):
    source: str
    target: str
    value: float
    type: str

class NetworkResponse(BaseModel):
    nodes: List[NetworkNode]
    links: List[NetworkLink]

class MetricCurve(BaseModel):
    auroc: float
    aupr: float
    recall_10: float
    recall_50: float
    fpr: List[float]
    tpr: List[float]
    recall: List[float]
    precision: List[float]

class ValidationMetricsResponse(BaseModel):
    svd: MetricCurve
    gnn: MetricCurve