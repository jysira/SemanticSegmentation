"""YOLO semantic FP16/INT8 export, sparse pruning and logit distillation practice."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.nn.tasks import SemanticSegmentationModel

from common import ROOT
from torch.utils.data import Dataset


class Masks(Dataset):
    def __init__(self,root,split,augment=False):
        self.files=sorted((root/'images'/split).glob('*.png'))
        self.mask_dir=root/'masks'/split
        self.augment=augment
        if not self.files:
            raise ValueError(f'No PNG images: {root}/images/{split}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):
        import cv2
        import numpy as np
        path=self.files[index]
        image=cv2.imread(str(path))
        mask=cv2.imread(str(self.mask_dir/(path.stem+'.png')),0)
        if image is None or mask is None:
            raise ValueError(path)
        image=cv2.resize(image,(640,640))
        mask=cv2.resize(mask,(640,640),interpolation=cv2.INTER_NEAREST)
        if self.augment:
            ys,xs=np.where((mask>0)&(mask!=255))
            if len(xs) and np.random.random()<.8:
                k=np.random.randint(len(xs))
                x,y=int(xs[k]),int(ys[k])
            else:
                x,y=np.random.randint(640,size=2)
            x,y=np.clip([x-128,y-128],0,384)
            image,mask=image[y:y+256,x:x+256],mask[y:y+256,x:x+256]
        return torch.from_numpy(image.transpose(2,0,1).copy()).float()/255,torch.from_numpy(mask.copy()).long()


def logits(output):
    return output[0] if isinstance(output,(tuple,list)) else output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['fp16','int8','prune','distill','evaluate'])
    parser.add_argument('--weights',default=str(ROOT/'runs/02_segmentation/clahe/pcb-2/weights/best.pt'))
    parser.add_argument('--teacher',help='Teacher semantic checkpoint, preferably a validated larger model')
    parser.add_argument('--data',type=Path,default=ROOT/'data/pcb_semantic_clahe/dataset.yaml')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/05_compression')
    parser.add_argument('--amount',type=float,default=.1)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--temperature',type=float,default=3.)
    parser.add_argument('--alpha',type=float,default=.3)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--split',choices=['val','test'],default='val')
    args=parser.parse_args()
    if not 0<=args.amount<1 or not 0<=args.alpha<=1 or args.temperature<=0:
        parser.error('Invalid pruning amount, alpha or temperature')
    args.output.mkdir(parents=True,exist_ok=True)
    wrapper=YOLO(args.weights,task='semantic')
    if args.mode=='evaluate':
        metrics=wrapper.val(data=str(args.data.resolve()),split=args.split,imgsz=args.imgsz,
                            batch=args.batch,project=str(args.output.resolve()),name=args.split)
        print(metrics)
        return
    if args.mode in {'fp16','int8'}:
        # Ultralytics calibration loader consumes train images from this YAML.
        path=wrapper.export(format='engine',imgsz=args.imgsz,batch=args.batch,device=0,
            half=args.mode=='fp16',int8=args.mode=='int8',data=str(args.data.resolve()),fraction=1.0)
        (args.output/f'{args.mode}_export.json').write_text(json.dumps({'artifact':str(path),'data':str(args.data),'mode':args.mode},indent=2))
        print(path)
        return
    if args.mode=='prune':
        # Sparse baseline: retain original graph dimensions; no dense-kernel speedup claim.
        layers=[(module,'weight') for module in wrapper.model.modules() if isinstance(module,nn.Conv2d)]
        prune.global_unstructured(layers,pruning_method=prune.L1Unstructured,amount=args.amount)
        for module,_ in layers:
            prune.remove(module,'weight')
        target=args.output/'pruned_sparse.pt'
        wrapper.save(str(target))
        zeros=sum(int((module.weight==0).sum()) for module,_ in layers)
        total=sum(module.weight.numel() for module,_ in layers)
        (args.output/'pruning.json').write_text(json.dumps({'zero_weights':zeros,'conv_weights':total,
            'sparsity':zeros/total,'structural_pruning':False,'note':'Tensor shapes unchanged; dense inference may not accelerate'},indent=2))
        print(target)
        return
    if not args.teacher:
        parser.error('distill requires --teacher')
    import yaml
    config=yaml.safe_load(args.data.read_text(encoding='utf-8'))
    root=Path(config['path'])
    if not root.is_absolute():
        root=args.data.resolve().parent/root
    names=config['names']
    if len(names)!=7:
        raise ValueError('This PCB practice loader requires background + six defect classes')
    teacher=YOLO(args.teacher,task='semantic')
    expected=list(names.values()) if isinstance(names,dict) else names
    if list(teacher.names.values())!=expected:
        raise ValueError('Teacher classes/order differ from dataset')
    # Rebuild the student head at the dataset class count, transfer compatible weights.
    student=SemanticSegmentationModel(wrapper.model.yaml,nc=7,verbose=False)
    student.load(wrapper.model)
    student.names=dict(enumerate(expected))
    wrapper.model=student
    device='cuda' if torch.cuda.is_available() else 'cpu'
    student.to(device).float()
    teacher.model.to(device).float().eval()
    teacher.model.requires_grad_(False)
    train=DataLoader(Masks(root,'train',augment=True),batch_size=args.batch,shuffle=True)
    valid=DataLoader(Masks(root,'val'),batch_size=1)
    optimizer=torch.optim.AdamW(student.parameters(),lr=args.lr)
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)
    best=-1.
    history=[]
    for epoch in range(args.epochs):
        student.train()
        losses=[]
        for images,masks in train:
            # Masks supplies BGR in 0..1; Ultralytics raw modules expect RGB.
            images=images[:,[2,1,0]].to(device)
            masks=masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            output=student(images)
            predicted=F.interpolate(logits(output),size=masks.shape[-2:],mode='bilinear',align_corners=False)
            with torch.no_grad():
                target=F.interpolate(logits(teacher.model(images)),size=masks.shape[-2:],mode='bilinear',align_corners=False)
            temperature=args.temperature
            kd=F.kl_div(F.log_softmax(predicted/temperature,dim=1),F.softmax(target/temperature,dim=1),reduction='none').sum(1)
            # Equal foreground/background contribution prevents background-only KD dominance.
            foreground=(masks>0)&(masks!=255)
            valid_pixels=masks!=255
            ce=F.cross_entropy(predicted,masks,ignore_index=255,reduction='none')
            weights=torch.where(foreground,10.,1.)*valid_pixels
            supervised=(ce*weights).sum()/weights.sum().clamp_min(1)
            distillation=(kd*weights).sum()/weights.sum().clamp_min(1)*temperature**2
            loss=(1-args.alpha)*supervised+args.alpha*distillation
            if isinstance(output,(tuple,list)) and len(output)>1:
                aux=F.interpolate(output[1],size=masks.shape[-2:],mode='bilinear',align_corners=False)
                loss=loss+.2*F.cross_entropy(aux,masks,ignore_index=255)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        student.eval()
        matrix=torch.zeros(7,7,device=device,dtype=torch.int64)
        with torch.no_grad():
            for images,masks in valid:
                masks=masks.to(device)
                prediction=F.interpolate(logits(student(images[:,[2,1,0]].to(device))),size=masks.shape[-2:],mode='bilinear',align_corners=False).argmax(1)
                keep=masks!=255
                matrix+=torch.bincount((masks[keep]*7+prediction[keep]).flatten(),minlength=49).reshape(7,7)
        matrix=matrix.cpu().numpy()
        union=matrix.sum(0)+matrix.sum(1)-matrix.diagonal()
        ious=np.divide(matrix.diagonal(),union,out=np.full(7,np.nan),where=union>0)
        miou=float(np.nanmean(ious[1:]))
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'defect_miou':miou})
        wrapper.save(str(args.output/'student_last.pt'))
        if miou>best:
            best=miou
            wrapper.save(str(args.output/'student_best.pt'))
        print(history[-1],flush=True)
        (args.output/'distillation_history.json').write_text(json.dumps(history,indent=2))


if __name__=='__main__':
    main()
