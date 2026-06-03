#!/usr/bin/env python3
"""Pull the model<->dataset `trained on` graph from the Wikibase and write graph.json.

Self-contained: the only runtime dependency is `requests`. The Wikibase property/class ids
for this instance are held in the IDS map below. Read-only SPARQL against the public instance.

Usage:
    python pull_kg.py
"""

import os
import json

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = HERE

BASE = "https://public-ai-data-sources.wikibase.cloud"
SPARQL = BASE + "/query/sparql"

# Wikibase property/class ids for THIS instance (public-ai-data-sources.wikibase.cloud).
# These are stable; update them only if the instance is rebuilt/renumbered.
IDS = {
    "P:instance_of":      "P1",
    "P:trained_on":       "P3",
    "P:edition_of":       "P4",
    "P:based_on":         "P23",
    "Q:training_stage":   "P10",
    "P:gpai_summary_url": "P19",
    "P:technical_paper":  "P20",
    "P:publication_date": "P26",
    "P:knowledge_cutoff": "P25",
    "C:ml_model":         "Q1",
    "C:ml_dataset":       "Q2",
}


def pid(local):
    """Resolve a local schema id (e.g. 'P:trained_on') to its Wikibase id, fail loudly."""
    if local not in IDS:
        raise SystemExit("unknown local id %r — add it to IDS" % local)
    return IDS[local]


def query(sparql):
    prefixes = (
        "PREFIX wd:  <%s/entity/>\n"
        "PREFIX wdt: <%s/prop/direct/>\n"
        "PREFIX p:   <%s/prop/>\n"
        "PREFIX ps:  <%s/prop/statement/>\n"
        "PREFIX pq:  <%s/prop/qualifier/>\n" % (BASE, BASE, BASE, BASE, BASE)
    )
    r = requests.get(SPARQL, params={"query": prefixes + sparql},
                     headers={"Accept": "application/sparql-results+json",
                              "User-Agent": "llm-kg-pull/1.0"}, timeout=60)
    r.raise_for_status()
    return r.json()["results"]["bindings"]


def val(binding, key):
    cell = binding.get(key)
    return cell["value"] if cell else None


def qid_of(uri):
    """https://.../entity/Q35 -> Q35"""
    return uri.rsplit("/", 1)[-1] if uri else None


def fetch_datasets():
    """Datasets with `edition of` (parent, for family merging) and `based on` (lineage)."""
    rows = query(
        "SELECT ?x ?xLabel (SAMPLE(?parent) AS ?ed) (SAMPLE(?b) AS ?based) WHERE {\n"
        "  ?x wdt:%(io)s wd:%(ds)s .\n"
        "  OPTIONAL { ?x rdfs:label ?xLabel FILTER(lang(?xLabel)='en') }\n"
        "  OPTIONAL { ?x wdt:%(ed)s ?parent }\n"
        "  OPTIONAL { ?x wdt:%(based)s ?b }\n"
        "} GROUP BY ?x ?xLabel"
        % {"io": pid("P:instance_of"), "ds": pid("C:ml_dataset"),
           "ed": pid("P:edition_of"), "based": pid("P:based_on")})
    nodes = {}
    for b in rows:
        qid = qid_of(val(b, "x"))
        nodes[qid] = {"id": qid, "label": val(b, "xLabel") or qid,
                      "url": BASE + "/entity/" + qid,
                      "edition_of": qid_of(val(b, "ed")),
                      "based_on": qid_of(val(b, "based"))}
    return nodes


def fetch_models():
    """Models with extra attributes (one row per model; SAMPLE collapses multi-values)."""
    rows = query(
        "SELECT ?m ?mLabel\n"
        "  (SAMPLE(?gp) AS ?gpai) (SAMPLE(?tp) AS ?paper)\n"
        "  (SAMPLE(?pd) AS ?pub_date) (SAMPLE(?kc) AS ?cutoff) (SAMPLE(?b) AS ?based)\n"
        "WHERE {\n"
        "  ?m wdt:%(io)s wd:%(model)s .\n"
        "  OPTIONAL { ?m rdfs:label ?mLabel FILTER(lang(?mLabel)='en') }\n"
        "  OPTIONAL { ?m wdt:%(gpai)s ?gp }\n"
        "  OPTIONAL { ?m wdt:%(paper)s ?tp }\n"
        "  OPTIONAL { ?m wdt:%(pub)s ?pd }\n"
        "  OPTIONAL { ?m wdt:%(cut)s ?kc }\n"
        "  OPTIONAL { ?m wdt:%(based)s ?b }\n"
        "} GROUP BY ?m ?mLabel"
        % {"io": pid("P:instance_of"), "model": pid("C:ml_model"),
           "gpai": pid("P:gpai_summary_url"), "paper": pid("P:technical_paper"),
           "pub": pid("P:publication_date"), "cut": pid("P:knowledge_cutoff"),
           "based": pid("P:based_on")})
    nodes = {}
    for b in rows:
        qid = qid_of(val(b, "m"))
        nodes[qid] = {"id": qid, "label": val(b, "mLabel") or qid,
                      "url": BASE + "/entity/" + qid,
                      "gpai": val(b, "gpai"), "paper": val(b, "paper"),
                      "publication_date": val(b, "pub_date"),
                      "knowledge_cutoff": val(b, "cutoff"),
                      "based_on": qid_of(val(b, "based"))}
    return nodes


def fetch_edges():
    rows = query(
        "SELECT ?m ?d ?stageLabel WHERE {\n"
        "  ?m wdt:%(io)s wd:%(model)s . ?m wdt:%(to)s ?d .\n"
        "  OPTIONAL { ?m p:%(to)s ?st . ?st ps:%(to)s ?d .\n"
        "             ?st pq:%(stage)s ?stage .\n"
        "             ?stage rdfs:label ?stageLabel FILTER(lang(?stageLabel)='en') }\n"
        "}" % {"io": pid("P:instance_of"), "model": pid("C:ml_model"),
               "to": pid("P:trained_on"), "stage": pid("Q:training_stage")})
    edges = []
    for b in rows:
        edges.append({"model": qid_of(val(b, "m")),
                      "dataset": qid_of(val(b, "d")),
                      "stage": val(b, "stageLabel")})
    return edges


def main():
    models = fetch_models()
    datasets = fetch_datasets()
    edges = fetch_edges()

    graph = {"models": list(models.values()),
             "datasets": list(datasets.values()),
             "edges": edges}

    with open(os.path.join(SITE, "graph.json"), "w") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    used = {e["dataset"] for e in edges}
    print("models: %d, datasets: %d (%d used in trained_on), edges: %d"
          % (len(models), len(datasets), len(used), len(edges)))
    print("wrote " + os.path.join(SITE, "graph.json"))


if __name__ == "__main__":
    main()
