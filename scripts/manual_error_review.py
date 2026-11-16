"""Build a Labelme review workspace and apply explicitly reviewed annotations."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/pcb_semantic_clahe'
WORK=ROOT/'reports/manual_semantic_review_pcb4'
NAMES=['background','missing_hole','mouse_bite','open_circuit','short','spur','spurious_copper']


def signature(shapes):
    return hashlib.sha256(json.dumps(shapes,sort_keys=True).encode()).hexdigest()


def main():
    global WORK
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['build','apply'])
    parser.add_argument('--weights',default=str(ROOT/'runs/02_segmentation/clahe/pcb-4/weights/best.pt'))
    parser.add_argument('--workspace',type=Path,default=WORK)
    parser.add_argument('--preserve-holes',action='store_true',help='Preserve original enclosed background holes when importing edited polygons')
    args=parser.parse_args()
    WORK=args.workspace.resolve()
    if args.mode=='build':
        if WORK.exists():
            raise FileExistsError(f'Review workspace exists: {WORK}')
        WORK.mkdir(parents=True)
        manifest=[]
        for split in ('train','val','test'):
            report=WORK/'analysis'/split
            subprocess.run([sys.executable,str(ROOT/'scripts/semantic_error_analysis.py'),'--weights',args.weights,'--split',split,'--output',str(report)],check=True)
            with (report/'errors.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            failures={}
            for row in rows:
                if row['error_type']!='correct':
                    failures.setdefault(row['image'],set()).add(row['error_type'])
            for filename,kinds in failures.items():
                image=Path(filename)
                mask_path=DATA/'masks'/split/(image.stem+'.png')
                mask=cv2.imread(str(mask_path),0)
                shapes=[]
                holes=0
                for cls in range(1,7):
                    contours,hierarchy=cv2.findContours((mask==cls).astype('uint8'),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
                    for i,contour in enumerate(contours):
                        if hierarchy[0,i,3]>=0:
                            holes+=1
                            continue
                        points=contour.reshape(-1,2).tolist()
                        if len(points)<3:
                            continue
                        shapes.append({'label':NAMES[cls],'points':points,'group_id':None,'shape_type':'polygon','flags':{}})
                primary=next(k for k in ('missed','wrong_class','boundary_shift','false_positive') if k in kinds)
                folder=WORK/'editable'/primary/split
                folder.mkdir(parents=True,exist_ok=True)
                shutil.copy2(image,folder/image.name)
                record={'version':'5.0.1','flags':{'reviewed':False},'shapes':shapes,
                        'imagePath':image.name,'imageData':None,'imageHeight':mask.shape[0],'imageWidth':mask.shape[1]}
                target=folder/(image.stem+'.json')
                target.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
                manifest.append({'split':split,'image':image.name,'errors':','.join(sorted(kinds)),'primary_error':primary,
                    'json':str(target.relative_to(WORK)),'mask':str(mask_path),
                    'original_shapes_signature':signature(shapes),'holes':holes,
                    'mask_sha256':hashlib.sha256(mask_path.read_bytes()).hexdigest()})
        (WORK/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        with (WORK/'review_list.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=['split','image','errors','holes'])
            writer.writeheader()
            writer.writerows({k:r[k] for k in writer.fieldnames} for r in manifest)
        print('Review images:',len(manifest),WORK,flush=True)
    else:
        manifest=json.loads((WORK/'manifest.json').read_text(encoding='utf-8'))
        prepared=[]
        pending=[]
        for item in manifest:
            record=json.loads((WORK/item['json']).read_text(encoding='utf-8'))
            if signature(record['shapes'])==item['original_shapes_signature']:
                continue
            if item['holes'] and not args.preserve_holes:
                pending.append({'image':item['image'],'json':item['json'],
                                'reason':'Original mask contains holes; polygon import would lose hole topology'})
                continue
            path=Path(item['mask'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['mask_sha256']:
                raise ValueError(f'Dataset changed since review export: {path}')
            original=cv2.imread(str(path),0)
            mask=np.zeros_like(original)
            for shape in record['shapes']:
                if shape['label'] not in NAMES[1:]:
                    raise ValueError(f'Unknown class {shape["label"]} in {item["json"]}')
                points=np.asarray(shape['points'],dtype=np.float32)
                kind=shape['shape_type']
                label=NAMES.index(shape['label'])
                if kind=='circle' and points.shape==(2,2):
                    center=tuple(np.rint(points[0]).astype(int))
                    radius=int(round(float(np.linalg.norm(points[1]-points[0]))))
                    cv2.circle(mask,center,radius,label,-1)
                    continue
                if kind=='rectangle' and points.shape==(2,2):
                    low=np.rint(points.min(0)).astype(int)
                    high=np.rint(points.max(0)).astype(int)
                    cv2.rectangle(mask,tuple(low),tuple(high),label,-1)
                    continue
                if kind not in {'polygon','oriented_rectangle'}:
                    raise ValueError(f'Unsupported shape {kind}: {item["json"]}')
                if points.shape[0]<3 or not np.isfinite(points).all():
                    raise ValueError('Invalid polygon')
                cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],NAMES.index(shape['label']))
            if item['holes'] and args.preserve_holes:
                # Preserve enclosed original background pixels, without restoring
                # original foreground or overwriting a different edited class.
                for class_id in range(1,len(NAMES)):
                    contours,hierarchy=cv2.findContours((original==class_id).astype('uint8'),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
                    if hierarchy is None:
                        continue
                    for index,contour in enumerate(contours):
                        if hierarchy[0,index,3]<0:
                            continue
                        hole=np.zeros_like(original)
                        cv2.drawContours(hole,[contour],-1,1,-1)
                        mask[(hole>0)&(original==0)&(mask==class_id)]=0
            mask[original==255]=255
            prepared.append((path,mask,item))
        for path,mask,item in prepared:
            temporary=path.with_name(path.stem+'.review.png')
            if not cv2.imwrite(str(temporary),mask):
                raise RuntimeError(temporary)
            temporary.replace(path)
            item['mask_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            record=json.loads((WORK/item['json']).read_text(encoding='utf-8'))
            item['original_shapes_signature']=signature(record['shapes'])
        (WORK/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Updated masks:',len(prepared),flush=True)
        (WORK/'pending_imports.json').write_text(json.dumps(pending,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Pending hole-aware imports:',len(pending),flush=True)


if __name__=='__main__':
    main()
