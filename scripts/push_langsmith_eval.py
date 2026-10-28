"""Optionally upload the RAG evaluation cases as a LangSmith dataset (no model runs)."""
import json, os, sys
from pathlib import Path

def main():
 if not os.getenv('LANGSMITH_API_KEY'): raise SystemExit('Set LANGSMITH_API_KEY first.')
 try: from langsmith import Client
 except ImportError as e: raise SystemExit('Install requirements-llm.txt first.') from e
 root=Path(__file__).resolve().parents[1]
 sys.path.insert(0, str(root))
 cases=[json.loads(x) for x in (root/'eval/rag_eval_dataset.jsonl').read_text().splitlines() if x.strip()]
 client=Client(); name=os.getenv('LANGSMITH_DATASET_NAME','teliport-rag-evaluation')
 try: dataset=client.create_dataset(dataset_name=name,description='TELIPORT RAG evaluation cases. Expected source labels require expert completion.')
 except Exception:
  matches=list(client.list_datasets(dataset_name=name));
  if not matches: raise
  dataset=matches[0]
 examples=[]
 for c in cases:
  examples.append({'inputs':{'query':c['query'],'asset_model':c['asset_model']},'outputs':{'expected_behavior':c['expected_behavior'],'expected_source_ids':c['expected_source_ids']},'metadata':{'case_id':c['id'],'notes':c['notes']}})
 client.create_examples(dataset_id=dataset.id,examples=examples)
 print(f'Uploaded {len(examples)} cases to LangSmith dataset {name!r}. This uploads test text/metadata; review privacy policy first.')
if __name__=='__main__': main()
