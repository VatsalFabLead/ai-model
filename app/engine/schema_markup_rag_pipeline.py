"""Schema Markup Generator — enterprise JSON-LD pipeline (Schema.org + Google guidelines)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from app.engine import schema_engine
from app.engine.open_data_retrieval import retrieve_from_sources
from app.engine.schema_markup_enrichment import (
  OPEN_DATASET_TREE,
  OPEN_DATA_SOURCES,
  apply_verified_defaults,
  build_linked_schema_graph,
  build_validation_checklist,
  build_validation_report,
  enrich_data_from_open_docs,
  extract_entities,
  map_relationships,
  normalize_schema_type,
  property_requirements,
  resolve_effective_type,
  score_schema,
  slugify,
  validate_google_compliance,
  validate_input,
  validate_nesting,
  validate_property_types,
  validate_schema_structure,
  validate_seo,
  _collect_warnings,
)

GENERATOR_VERSION = "schema-markup-rag-v3.4"

_RAG_TIMEOUT_SEC = 6.0
_AI_ENHANCE_TIMEOUT_SEC = 12.0

ARCHITECTURE_FLOW = [
  "input",
  "input_validator",
  "schema_type_detector",
  "entity_extractor",
  "required_property_loader",
  "recommended_property_loader",
  "open_data_retrieval",
  "relationship_mapper",
  "jsonld_generator",
  "verified_defaults",
  "nested_schema_builder",
  "schema_org_validator",
  "rich_result_validator",
  "quality_checker",
  "seo_analyzer",
  "export_engine",
  "final_output",
]

OPEN_STANDARDS = [
  "Schema.org Vocabulary",
  "Google Search Central (Structured Data)",
  "JSON-LD Specification",
  "Open Knowledge Graph Standards",
  "Rich Results Guidelines",
]


async def _fetch_schema_knowledge(name: str, schema_type: str, language: str | None) -> list[Any]:
  guidance = schema_engine.get_guidance(schema_type, language)
  if not guidance:
    return []
  return [type("OpenDoc", (), {"title": f"{schema_type} schema guidance", "text": guidance, "source": "schema_knowledge"})()]


async def run_schema_markup_pipeline(
  *,
  schema_type: str,
  name: str,
  data: dict[str, Any] | None = None,
  language: str | None = None,
  ai_enhance: bool = False,
  use_rag: bool = False,
  provider: Any = None,
) -> dict[str, Any]:
  """Full schema workflow — verified data only; templates for missing user fields."""
  from app.services import schema_markup as sm

  t0 = time.perf_counter()
  data = dict(data or {})
  stages: dict[str, Any] = {}

  type_map = schema_engine._TYPE_MAP  # noqa: SLF001

  stages["input"] = {"schema_type": schema_type, "name": name, "language": language, "use_rag": use_rag}

  input_val = validate_input(schema_type, name, type_map)
  stages["input_validator"] = input_val
  if not input_val["valid"]:
    raise ValueError("; ".join(input_val["issues"]))

  stype = resolve_effective_type(normalize_schema_type(schema_type, type_map))
  requested = normalize_schema_type(schema_type, type_map)
  category = schema_engine.category_for_type(stype)
  stages["schema_type_detector"] = {
    "requested": schema_type,
    "canonical": requested,
    "effective": stype,
    "category": category,
    "remapped": requested != stype,
  }

  reqs = property_requirements(stype)
  guidance = schema_engine.get_guidance(stype, language)
  stages["required_property_loader"] = {
    "required": reqs.get("required", []),
    "required_count": len(reqs.get("required", [])),
    "guidance_loaded": bool(guidance),
  }
  stages["recommended_property_loader"] = {
    "recommended": reqs.get("recommended", []),
    "recommended_count": len(reqs.get("recommended", [])),
    "rich_results_eligible": reqs.get("rich_results_eligible", False),
  }

  entities = extract_entities(name, data)
  stages["entity_extractor"] = {"entities": entities, "count": len(entities)}

  open_docs: list[Any] = []
  if use_rag:
    seed_keywords = [str(k) for k in (data.get("keywords") or []) if isinstance(k, str)]
    try:
      open_docs = await asyncio.wait_for(
        retrieve_from_sources(name, seed_keywords, OPEN_DATA_SOURCES, per_source=1),
        timeout=_RAG_TIMEOUT_SEC,
      )
    except (asyncio.TimeoutError, Exception):
      open_docs = []
    open_docs = list(open_docs) + await _fetch_schema_knowledge(name, stype, language)
    data = enrich_data_from_open_docs(name, data, open_docs, [])
    stages["open_data_retrieval"] = {
      "sources": OPEN_DATA_SOURCES,
      "datasets": OPEN_DATASET_TREE,
      "doc_count": len(open_docs),
      "keywords_enriched": data.get("keywords", [])[:10],
    }
  else:
    stages["open_data_retrieval"] = {"skipped": True}

  schema = sm._build(stype, name, data, language)  # noqa: SLF001
  stages["jsonld_generator"] = {"field_count": len(schema), "mode": "verified_templates"}

  schema = map_relationships(schema, stype)
  stages["relationship_mapper"] = {"nested_types": reqs.get("recommended", [])}

  schema = apply_verified_defaults(schema, stype, name, data)
  stages["verified_defaults"] = {"uses_templates": True}

  graph = build_linked_schema_graph(schema, stype, name, data)
  stages["nested_schema_builder"] = {"uses_graph": "@graph" in graph, "linked_nodes": len(graph.get("@graph", []))}

  if ai_enhance and provider is not None:
    schema_inner = graph["@graph"][0] if graph.get("@graph") else graph
    try:
      enhanced = await asyncio.wait_for(
        sm._ai_enhance(provider, schema_inner, stype, language),  # noqa: SLF001
        timeout=_AI_ENHANCE_TIMEOUT_SEC,
      )
      if graph.get("@graph"):
        graph["@graph"][0] = enhanced
      else:
        graph = enhanced
      stages["jsonld_generator"]["ai_enhanced"] = True
    except (asyncio.TimeoutError, Exception):
      stages["jsonld_generator"]["ai_enhance_skipped"] = "timeout_or_error"

  if graph.get("@graph"):
    cleaned_nodes: list[dict[str, Any]] = []
    for node in graph["@graph"]:
      clean = schema_engine.sanitize_schema(node, str(node.get("@type", stype)))
      clean.pop("@context", None)
      cleaned_nodes.append(clean)
    final_schema = {"@context": "https://schema.org", "@graph": cleaned_nodes}
  else:
    final_schema = schema_engine.sanitize_schema(graph, stype)

  structure_val = validate_schema_structure(final_schema, stype)
  property_types_val = validate_property_types(final_schema)
  nesting_val = validate_nesting(final_schema, stype)
  stages["schema_org_validator"] = {
    "structure": structure_val,
    "property_types": property_types_val,
    "nesting": nesting_val,
  }

  google_val = validate_google_compliance(final_schema, stype)
  stages["rich_result_validator"] = google_val

  seo_val = validate_seo(final_schema, stype, google_val)
  warnings = _collect_warnings(final_schema, data, stype)
  seo_analysis = {
    "rich_results_eligible": seo_val.get("rich_results_eligible", False),
    "slug": slugify(name),
    "suggested_url": data.get("url") or "{{URL}}",
    "ai_search_friendly": seo_val.get("ai_search_friendly", False),
    "keywords": data.get("keywords", [])[:10],
    "standards": OPEN_STANDARDS,
    "open_datasets": OPEN_DATASET_TREE,
  }
  stages["seo_analyzer"] = seo_analysis

  validation_checklist = build_validation_checklist(
    input_val=input_val,
    structure_val=structure_val,
    property_types_val=property_types_val,
    nesting_val=nesting_val,
    google_val=google_val,
    seo_val=seo_val,
    reqs=reqs,
  )

  scores = score_schema(final_schema, stype, structure_val, google_val)
  validation_report = build_validation_report(final_schema, stype, scores, warnings, google_val)
  stages["quality_checker"] = validation_report

  jsonld_string = schema_engine.pretty_jsonld(final_schema)
  stages["export_engine"] = {"format": "application/ld+json", "bytes": len(jsonld_string.encode("utf-8"))}
  stages["final_output"] = {
    "schema_type": stype,
    "overall_score": scores["overall_score"],
    "needs_user_input": google_val.get("needs_user_input", False),
  }

  elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

  return {
    "generator_version": GENERATOR_VERSION,
    "schema_type": stype,
    "category": category,
    "language": schema_engine.bcp47(language),
    "jsonld": final_schema,
    "jsonld_string": jsonld_string,
    "quality": {
      "completeness_score": scores["completeness_score"],
      "verified_completeness_score": scores.get("verified_completeness_score", 0),
      "schema_score": scores["schema_score"],
      "google_compliance_score": scores["google_compliance_score"],
      "seo_score": scores["seo_score"],
      "overall_score": scores["overall_score"],
      "seo_ready": scores["seo_ready"],
      "missing_recommended_fields": structure_val.get("missing_required", []),
      "field_count": len(final_schema.get("@graph", [final_schema])[0]) if final_schema.get("@graph") else len(final_schema),
    },
    "validation": {
      "input": input_val,
      "structure": structure_val,
      "property_types": property_types_val,
      "nesting": nesting_val,
      "google": google_val,
      "seo": seo_val,
      "checklist": validation_checklist,
      "report": validation_report,
    },
    "entities": entities,
    "requirements": reqs,
    "seo_analysis": seo_analysis,
    "architecture": {
      "version": GENERATOR_VERSION,
      "flow": ARCHITECTURE_FLOW,
      "stages": stages,
      "open_standards": OPEN_STANDARDS,
      "open_datasets": OPEN_DATASET_TREE,
    },
    "elapsed_ms": elapsed_ms,
  }
