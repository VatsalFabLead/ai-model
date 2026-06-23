"""Copyright & plagiarism checker API."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.deps import get_tool_provider
from app.core.security import verify_api_key
from app.services import plagiarism_checker

router = APIRouter(prefix="/plagiarism-check", tags=["plagiarism-check"])


class PlagiarismRequest(BaseModel):
  content: str = Field(..., min_length=40, max_length=50000)


class MatchedSegment(BaseModel):
  type: str
  label: str
  text: str
  match_percent: int
  source: str
  matched_excerpt: str = ""
  url: str = ""
  sentence_index: int = -1
  source_type: str = ""


class WorkflowStep(BaseModel):
  step: str
  status: str
  detail: str = ""


class HighlightedSentence(BaseModel):
  sentence_index: int
  sentence: str
  highlight: str
  match_percent: int
  risk: str
  source: str
  source_type: str = ""
  url: str = ""
  matched_excerpt: str = ""
  label: str = ""


class SimilarityReport(BaseModel):
  similarity_percent: int = 0
  original_percent: int = 100
  risk_level: str = "low"
  sentence_count: int = 0
  chunks_scanned: int = 0
  chunks_matched: int = 0
  high_confidence_matches: int = 0
  medium_confidence_matches: int = 0
  avg_embedding_similarity: float = 0.0
  sources_checked: list[str] = Field(default_factory=list)
  verdict: str = ""


class PlagiarismResponse(BaseModel):
  word_count: int
  unique_words: int
  sentence_count: int
  similarity_percent: int
  original_percent: int
  originality_score: int
  risk_level: str
  likely_original: bool
  content_preview: str
  matched_segments: list[MatchedSegment]
  duplicate_sentences: list[str]
  repeated_sequences: list[str]
  flags: list[str]
  suggestions: list[str]
  summary: str
  overlap_with_original_percent: int = 0
  embedding_available: bool = False
  sources_used: list[str] = Field(default_factory=list)
  avg_embedding_similarity: float = 0.0
  chunks_scanned: int = 0
  chunks_matched: int = 0
  embedding_note: str = ""
  scan_incomplete: bool = False
  workflow: list[WorkflowStep] = Field(default_factory=list)
  highlighted_sentences: list[HighlightedSentence] = Field(default_factory=list)
  similarity_report: SimilarityReport | None = None


class PlagiarismRemoveResponse(BaseModel):
  original_content: str
  rewritten_content: str
  before: PlagiarismResponse
  after: PlagiarismResponse
  similarity_percent_before: int
  similarity_percent_after: int
  original_percent_before: int
  original_percent_after: int
  overlap_with_original_percent: int = 0
  improvement: int
  summary: str


@router.post("/check", response_model=PlagiarismResponse)
async def check(
  payload: PlagiarismRequest,
  _: str = Depends(verify_api_key),
) -> PlagiarismResponse:
  try:
    result = await plagiarism_checker.check_content(content=payload.content)
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  return PlagiarismResponse(**result)


@router.post("/remove", response_model=PlagiarismRemoveResponse)
async def remove(
  payload: PlagiarismRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> PlagiarismRemoveResponse:
  provider = get_tool_provider(request)
  try:
    result = await plagiarism_checker.remove_plagiarism(provider, content=payload.content)
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Rewrite failed: {exc}") from exc
  return PlagiarismRemoveResponse(**result)
