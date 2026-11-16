"""Evaluate pcb-3 and rebuild review workspace after preserving pending edits."""
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]


def main():
    weights=ROOT/'runs/02_segmentation/clahe/pcb-3/weights/best.pt'
    data=ROOT/'data/pcb_semantic_clahe/dataset.yaml'
    workspace=ROOT/'reports/manual_semantic_review'
    # Apply before evaluation so metrics and regenerated annotations share one label version.
    subprocess.run([sys.executable,str(ROOT/'scripts/manual_error_review.py'),'apply'],check=True)
    model=YOLO(str(weights),task='semantic')
    model.val(data=str(data),split='test',imgsz=640,batch=4,device=0,save_json=True,
              project=str(ROOT/'runs/03_evaluation/clahe'),name='test-3')
    if workspace.exists():
        archive=ROOT/'reports'/datetime.now().strftime('manual_semantic_review_previous_%Y%m%d_%H%M%S')
        workspace.rename(archive)
        print('Previous review workspace:',archive,flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/manual_error_review.py'),'build','--weights',str(weights)],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/group_manual_review.py')],check=True)


if __name__=='__main__':
    main()
