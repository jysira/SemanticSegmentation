"""Region-level semantic error analysis using greedy class-agnostic IoU matching."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def regions(mask,min_area):
    items = []
    for cls in np.unique(mask):
        if cls in (0,255):
            continue
        count,labels,stats,_ = cv2.connectedComponentsWithStats((mask==cls).astype('uint8'),8)
        for index in range(1,count):
            if stats[index,cv2.CC_STAT_AREA] >= min_area:
                items.append((int(cls),labels==index))
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/pcb_semantic_clahe/dataset.yaml')
    parser.add_argument('--weights',default=str(ROOT/'runs/02_segmentation/clahe/pcb-2/weights/best.pt'))
    parser.add_argument('--split',default='test',choices=['train','val','test'])
    parser.add_argument('--output',type=Path,default=ROOT/'reports/semantic_errors')
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--min-area',type=int,default=5)
    parser.add_argument('--match-iou',type=float,default=.1)
    parser.add_argument('--good-iou',type=float,default=.5)
    args = parser.parse_args()
    if not 0 < args.match_iou <= args.good_iou <= 1 or args.min_area<1:
        parser.error('Require 0 < match-iou <= good-iou <= 1 and min-area >= 1')
    config = yaml.safe_load(args.data.read_text(encoding='utf-8'))
    root = Path(config['path'])
    if not root.is_absolute():
        root = args.data.resolve().parent/root
    folder = root/config[args.split]
    model = YOLO(args.weights,task='semantic')
    rows = []
    totals = {key:0 for key in ['missed','wrong_class','boundary_shift','false_positive','correct']}
    image_counts = dict.fromkeys(totals,0)
    args.output.mkdir(parents=True,exist_ok=True)
    files = sorted(p for p in folder.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'})
    for number,path in enumerate(files):
        relative = path.relative_to(folder)
        gt = cv2.imread(str(root/config.get('masks_dir','masks')/args.split/relative.with_suffix('.png')),0)
        if gt is None:
            raise ValueError(f'Missing ground truth: {path}')
        result = model.predict(str(path),imgsz=args.imgsz,verbose=False)[0]
        pred = result.semantic_mask.data.cpu().numpy().squeeze().astype('uint8')
        pred = cv2.resize(pred,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST)
        pred[gt==255]=255
        truth,predicted = regions(gt,args.min_area),regions(pred,args.min_area)
        pairs = []
        for i,(_,a) in enumerate(truth):
            for j,(_,b) in enumerate(predicted):
                intersection = int(np.logical_and(a,b).sum())
                if intersection:
                    iou = intersection/int(np.logical_or(a,b).sum())
                    if iou>=args.match_iou:
                        pairs.append((iou,i,j))
        used_gt,used_pred = set(),set()
        events = []
        for iou,i,j in sorted(pairs,reverse=True):
            if i in used_gt or j in used_pred:
                continue
            used_gt.add(i)
            used_pred.add(j)
            kind = 'wrong_class' if truth[i][0]!=predicted[j][0] else 'boundary_shift' if iou<args.good_iou else 'correct'
            events.append((kind,i,j,iou))
        events.extend(('missed',i,None,0.) for i in range(len(truth)) if i not in used_gt)
        events.extend(('false_positive',None,j,0.) for j in range(len(predicted)) if j not in used_pred)
        kinds = set()
        for kind,i,j,iou in events:
            totals[kind]+=1
            kinds.add(kind)
            rows.append({'image':str(path),'error_type':kind,'gt_class':truth[i][0] if i is not None else '',
                         'pred_class':predicted[j][0] if j is not None else '', 'iou':round(iou,6)})
        image = cv2.imread(str(path))
        for kind in kinds:
            image_counts[kind]+=1
            if kind=='correct':
                continue
            view = image.copy()
            for event,i,j,iou in events:
                if event!=kind:
                    continue
                for index,items,color in [(i,truth,(0,255,0)),(j,predicted,(0,0,255))]:
                    if index is not None:
                        contour,_ = cv2.findContours(items[index][1].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(view,contour,-1,color,2)
            cv2.putText(view,f'{kind}: GT green / prediction red',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            target = args.output/kind/relative.with_suffix('.jpg')
            target.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(target),view)
        if number%50==0:
            print(number+1,len(files),flush=True)
    fields = ['image','error_type','gt_class','pred_class','iou']
    with (args.output/'errors.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {'images':len(files),'region_counts':totals,'image_counts':image_counts,'weights':args.weights,
              'data':str(args.data),'split':args.split,'min_area':args.min_area,'match_iou':args.match_iou,
              'good_iou':args.good_iou,'matching':'greedy IoU, class agnostic, one-to-one; disconnected or split/merged regions may produce both missed and false positive events'}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
