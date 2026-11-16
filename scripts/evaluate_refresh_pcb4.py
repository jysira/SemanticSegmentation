"""Evaluate latest PCB experiment and rebuild its manual review workspace."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]


def main():
    review=ROOT/'scripts/manual_error_review.py'
    subprocess.run([sys.executable,str(review),'apply','--workspace',str(ROOT/'reports/manual_semantic_review_pcb3'),'--preserve-holes'],check=True)
    weights=ROOT/'runs/02_segmentation/clahe/pcb-4/weights/best.pt'
    data=ROOT/'data/pcb_semantic_clahe/dataset.yaml'
    model=YOLO(str(weights),task='semantic')
    report={}
    for split in ('test','train'):
        metrics=model.val(data=str(data),split=split,imgsz=640,batch=4,device=0,save_json=True,
            project=str(ROOT/'runs/03_evaluation/clahe'),name=split+'_pcb4')
        iou=np.asarray(metrics.per_class_iou,dtype=float)
        report[split]={'miou_including_background':float(metrics.miou),
            'defect_miou':float(np.nanmean(iou[1:])),'per_class_iou':iou.tolist(),
            'weights':str(weights),'data':str(data)}
        (ROOT/'reports/pcb4_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(review),'build','--workspace',str(ROOT/'reports/manual_semantic_review_pcb4'),'--weights',str(weights)],check=True)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
