from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest
from src.maintenance_rag import _chunks, retrieve, register_document, approve_document


def test_chunks_empty_and_content():
    assert _chunks("   ") == []
    chunks = _chunks("Sentence one. " * 300, max_chars=120, overlap=20)
    assert len(chunks) > 1
    assert all(chunks)


def test_only_registered_approved_documents_are_retrieved(tmp_path: Path):
    manual=tmp_path/'approved_manual.txt'; manual.write_text("For vibration alarms, inspect the documented vibration trend and compare it with the approved baseline.",encoding='utf-8')
    db=tmp_path/'registry.db'
    assert retrieve('vibration alarm baseline',kb_dir=tmp_path,db_path=db)==[]
    info=register_document(manual,title='Vibration Manual',asset_model='FD001',revision='A',db_path=db)
    assert retrieve('vibration alarm baseline',kb_dir=tmp_path,db_path=db)==[]
    expiry=(datetime.now(timezone.utc)+timedelta(days=30)).isoformat()
    approve_document(info['doc_id'],'Responsible Engineer',expiry,db_path=db)
    result=retrieve('vibration alarm baseline',kb_dir=tmp_path,db_path=db,asset_model='FD001')
    assert result and result[0]['source']==str(manual)
    assert result[0]['approved_by']=='Responsible Engineer'
    assert 'vibration' in result[0]['text'].lower()


def test_asset_filter_and_modified_file_exclusion(tmp_path: Path):
    manual=tmp_path/'manual.md'; manual.write_text('Pump inspection vibration baseline.',encoding='utf-8'); db=tmp_path/'registry.db'
    info=register_document(manual,title='Pump Manual',asset_model='PUMP-X',revision='2',db_path=db)
    approve_document(info['doc_id'],'Engineer',(datetime.now(timezone.utc)+timedelta(days=5)).isoformat(),db_path=db)
    assert retrieve('pump vibration',kb_dir=tmp_path,db_path=db,asset_model='FD001')==[]
    manual.write_text('Modified unauthorized procedure content.',encoding='utf-8')
    assert retrieve('pump vibration',kb_dir=tmp_path,db_path=db,asset_model='PUMP-X')==[]


def test_retrieve_empty_knowledge_base(tmp_path: Path):
    assert retrieve('pump vibration',kb_dir=tmp_path,db_path=tmp_path/'empty.db') == []


def test_approved_doc_requires_approver_and_expiry(tmp_path: Path):
    p=tmp_path/'manual.txt'; p.write_text('test manual')
    with pytest.raises(ValueError): register_document(p,title='T',asset_model='X',revision='1',status='approved',db_path=tmp_path/'db')


def test_expired_document_is_excluded(tmp_path: Path):
    manual=tmp_path/'expired.txt'; manual.write_text('Approved inspection procedure for pump vibration.',encoding='utf-8'); db=tmp_path/'registry.db'
    info=register_document(manual,title='Expired',asset_model='PUMP-X',revision='1',db_path=db)
    past=(datetime.now(timezone.utc)-timedelta(days=1)).isoformat()
    # Direct SQL fixture simulates an approval that has since expired.
    import sqlite3
    con=sqlite3.connect(db); con.execute("UPDATE documents SET status='approved',approved_by='Engineer',approved_at=?,expires_at=? WHERE doc_id=?",(datetime.now(timezone.utc).isoformat(),past,info['doc_id'])); con.commit(); con.close()
    assert retrieve('pump vibration inspection',kb_dir=tmp_path,db_path=db,asset_model='PUMP-X') == []


def test_docx_heading_extraction(tmp_path: Path):
    docx=pytest.importorskip('docx')
    from src.maintenance_rag import _extract
    path=tmp_path/'manual.docx'; doc=docx.Document(); doc.add_heading('Inspection',level=1); doc.add_paragraph('Review the approved vibration trend.'); doc.save(path)
    blocks=_extract(path)
    assert blocks and blocks[0]['section']=='Inspection'
    assert 'vibration trend' in blocks[0]['text']
