"""Register, list, and approve maintenance manuals from the command line."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.maintenance_rag import register_document, approve_document, list_documents

def main():
 p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
 r=sub.add_parser('register'); r.add_argument('path'); r.add_argument('--title',required=True); r.add_argument('--asset-model',required=True,help='Exact model, or * for explicitly cross-model content'); r.add_argument('--revision',required=True); r.add_argument('--notes',default='')
 a=sub.add_parser('approve'); a.add_argument('doc_id'); a.add_argument('--approver',required=True); a.add_argument('--expires-at',required=True,help='ISO-8601 expiry, e.g. 2027-12-31T23:59:00+00:00')
 sub.add_parser('list')
 args=p.parse_args()
 if args.cmd=='register': result=register_document(args.path,title=args.title,asset_model=args.asset_model,revision=args.revision,notes=args.notes)
 elif args.cmd=='approve': approve_document(args.doc_id,args.approver,args.expires_at); result={'approved':args.doc_id}
 else: result=list_documents()
 print(json.dumps(result,indent=2,default=str))
if __name__=='__main__': main()
