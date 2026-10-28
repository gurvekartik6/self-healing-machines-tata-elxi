"""Evaluate RAG retrieval behavior and, when gold labels exist, retrieval metrics.

This is an offline retrieval evaluation, not an LLM answer-quality evaluation.
Gold expected_source_ids must be supplied by a domain reviewer before accuracy
metrics are meaningful. The script never treats missing gold labels as success.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.maintenance_rag import retrieve


def main():
    cases=[json.loads(x) for x in (ROOT/'eval'/'rag_eval_dataset.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    results=[]; labeled=0; hit1=[]; recall=[]; reciprocal=[]; abstention_checks=[]
    for case in cases:
        docs=retrieve(case['query'],top_k=int(case.get('top_k',4)),asset_model=case.get('asset_model'))
        sources=[]
        for d in docs:
            if d['source'] not in sources: sources.append(d['source'])
        expected=case.get('expected_source_ids') or []
        metrics=None
        if expected:
            labeled += 1
            hits=[i for i,s in enumerate(sources,1) if s in expected]
            metrics={'hit_at_1': bool(hits and hits[0] == 1 and sources[0] in expected),
                     'recall_at_k': len(set(sources)&set(expected))/len(set(expected)),
                     'reciprocal_rank': (1/min(hits)) if hits else 0.0}
            hit1.append(float(metrics['hit_at_1'])); recall.append(metrics['recall_at_k']); reciprocal.append(metrics['reciprocal_rank'])
        behavior=case.get('expected_behavior','')
        if behavior=='abstain_without_approved_sources':
            abstention_checks.append(not docs)
        results.append({'id':case['id'],'expected_behavior':behavior,'retrieved_count':len(docs),
          'retrieved_sources':[{'source':d['source'],'title':d.get('title'),'revision':d.get('revision'),'page':d.get('page'),'section':d.get('section'),'score':d.get('score')} for d in docs],
          'expected_source_ids':expected,'retrieval_metrics':metrics,
          'automatically_scored': bool(expected),
          'note':case.get('notes','')})
    summary={'case_count':len(cases),'gold_labeled_case_count':labeled,
      'mean_hit_at_1':sum(hit1)/len(hit1) if hit1 else None,
      'mean_recall_at_k':sum(recall)/len(recall) if recall else None,
      'mean_reciprocal_rank':sum(reciprocal)/len(reciprocal) if reciprocal else None,
      'empty_source_abstention_checks':len(abstention_checks),
      'empty_source_abstention_passes':sum(abstention_checks),
      'warning':'Metrics are only computed for cases with reviewer-approved expected_source_ids. This script does not evaluate Gemini answer faithfulness or correctness.'}
    out=ROOT/'reports'/'rag_evaluation.json'; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps({'summary':summary,'cases':results},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2)); print(f'Report: {out}')
if __name__=='__main__': main()
