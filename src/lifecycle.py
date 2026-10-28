"""Model lifecycle utilities: versioned retraining, approval gates, monitoring and drift."""
import json, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import ROOT, REPORTS, MODELS, RAW

APPROVAL = REPORTS / 'model_approval.json'


def dataset_version():
    h=hashlib.sha256()
    for name in ('train_FD001.txt','test_FD001.txt','RUL_FD001.txt'):
        path=RAW/name
        if path.exists():
            h.update(name.encode()); h.update(path.read_bytes())
    return h.hexdigest()[:16]


def record_candidate(metrics):
    """Record a candidate and apply an explicit MAE-based promotion gate."""
    current_path=REPORTS/'champion.json'
    candidate={
      'candidate_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
      'dataset_version': dataset_version(), 'data_source': metrics.get('data_source'),
      'validation_mae': metrics.get('random_forest_validation',{}).get('mae'),
      'validation_rmse': metrics.get('random_forest_validation',{}).get('rmse'),
      'test_mae': metrics.get('test',{}).get('mae'), 'test_rmse': metrics.get('test',{}).get('rmse'),
      'status':'PENDING_APPROVAL', 'approval_required': True,
      'created_at': datetime.now(timezone.utc).isoformat()
    }
    (REPORTS/'candidate_model.json').write_text(json.dumps(candidate,indent=2))
    if not current_path.exists():
        candidate['comparison']='No prior champion; candidate requires human approval.'
    else:
        champion=json.loads(current_path.read_text())
        old=champion.get('validation_mae')
        candidate['comparison'] = 'candidate_better' if old is not None and candidate['validation_mae'] < old else 'candidate_not_better_or_incomparable'
        candidate['previous_champion_validation_mae']=old
    (REPORTS/'candidate_model.json').write_text(json.dumps(candidate,indent=2))
    return candidate


def approve_candidate(approved_by, approve=True, rationale=''):
    path=REPORTS/'candidate_model.json'
    if not path.exists(): raise FileNotFoundError('Run training to create a candidate first.')
    if not str(approved_by).strip(): raise ValueError('Approver name is required.')
    candidate=json.loads(path.read_text())
    if approve and candidate.get('comparison') == 'candidate_not_better_or_incomparable':
        raise ValueError('Promotion blocked: candidate did not beat or could not be compared with the current champion. Reject it or run a comparable evaluation first.')
    candidate.update({'status':'APPROVED' if approve else 'REJECTED','approved_by':approved_by,
                      'approval_rationale':rationale,'approved_at':datetime.now(timezone.utc).isoformat()})
    path.write_text(json.dumps(candidate,indent=2))
    if approve:
        import shutil
        candidate_model = MODELS/'candidate_rul_model.joblib'
        if candidate_model.exists(): shutil.copy2(candidate_model, MODELS/'rul_model.joblib')
        (REPORTS/'champion.json').write_text(json.dumps(candidate,indent=2))
    with (REPORTS/'model_approval_history.jsonl').open('a') as f: f.write(json.dumps(candidate)+'\n')
    return candidate


def monitor_predictions(predictions):
    """Write operational summary and simple feature drift report against train features."""
    from src.config import PROCESSED
    train_path=PROCESSED/'train_features.csv'
    result={'created_at':datetime.now(timezone.utc).isoformat(),'rows':int(len(predictions)),
      'anomaly_count':int(predictions['anomaly'].sum()),
      'anomaly_rate':float(predictions['anomaly'].mean()) if len(predictions) else 0.0,
      'low_rul_count':int((predictions['predicted_rul']<=30).sum()),
      'alerts':[]}
    if result['anomaly_rate'] > 0.10: result['alerts'].append('Anomaly rate exceeds 10%; investigate sensor/data distribution.')
    if train_path.exists():
        train=pd.read_csv(train_path); result['feature_drift']={}
        # Compare current latest-engine feature distribution to training feature distribution.
        for col in [c for c in train.columns if c.endswith('_mean5')][:10]:
            if col in predictions.columns:
                a=pd.to_numeric(train[col],errors='coerce').dropna(); b=pd.to_numeric(predictions[col],errors='coerce').dropna()
                if len(a) and len(b):
                    scale=float(a.std()) or 1.0; z=abs(float(b.mean()-a.mean())/scale)
                    result['feature_drift'][col]={'standardized_mean_shift':round(z,3),'alert':bool(z>=3)}
                    if z>=3: result['alerts'].append(f'Potential feature drift: {col} shifted by {z:.2f} training standard deviations.')
    (REPORTS/'monitoring.json').write_text(json.dumps(result,indent=2))
    return result


def write_sensor_plot():
    """Create sensor-history plots for a representative training engine, when matplotlib exists."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.data_pipeline import read_file
    from src.config import FEATURE_SENSORS
    df=read_file(RAW/'train_FD001.txt')
    engine=int(df.engine_id.iloc[0]); d=df[df.engine_id==engine]
    REPORTS.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(11,5))
    for col in FEATURE_SENSORS[:5]: ax.plot(d.cycle,d[col],label=col,linewidth=1)
    ax.set(title=f'FD001 sensor history — engine {engine}',xlabel='Cycle',ylabel='Raw sensor value'); ax.legend(ncol=3); fig.tight_layout()
    fig.savefig(REPORTS/'sensor_history.png',dpi=140); plt.close(fig)
