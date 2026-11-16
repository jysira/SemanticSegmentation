"""Count semantic class images, connected regions, pixels and area quantiles."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = {'min':0, 'p10':.1, 'p25':.25, 'p50':.5, 'p75':.75, 'p90':.9, 'p95':.95, 'max':1}


def summarize(names, accumulators, split, min_area):
    rows = []
    for class_id,name in names.items():
        item = accumulators[class_id]
        areas = np.asarray(item['areas'],dtype=np.float64)
        row = {'split':split,'class_id':class_id,'class_name':name,
               'image_count':item['images'],'pixel_count':item['pixels'],
               'region_count':len(areas),'min_region_area':min_area}
        for key,q in QUANTILES.items():
            row['area_'+key] = float(np.quantile(areas,q)) if len(areas) else None
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/pcb_semantic_merged_grouped/dataset.yaml')
    parser.add_argument('--splits',nargs='+',default=['train','val','test'])
    parser.add_argument('--output',type=Path,default=ROOT/'reports/semantic_stats')
    parser.add_argument('--min-area',type=int,default=1,help='Minimum connected-region area in pixels')
    parser.add_argument('--include-background',action='store_true')
    args = parser.parse_args()
    if args.min_area < 1:
        parser.error('--min-area must be positive')
    config_path = args.data.resolve()
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    root = Path(config['path'])
    if not root.is_absolute():
        root = config_path.parent/root
    raw_names = config['names']
    names = dict(enumerate(raw_names)) if isinstance(raw_names,list) else {int(k):v for k,v in raw_names.items()}
    if not args.include_background:
        names = {k:v for k,v in names.items() if str(v).lower() not in {'background','_background_'}}
    def new_counts():
        return {key:{'images':0,'pixels':0,'areas':[]} for key in names}
    combined = new_counts()
    rows = []
    file_counts = {}
    for split in args.splits:
        if split not in config:
            raise ValueError(f'Split missing from YAML: {split}')
        image_dir = root/config[split]
        mask_component = config.get('masks_dir','masks')
        image_parts = list(image_dir.parts)
        if 'images' not in image_parts:
            raise ValueError(f'Expected an images directory component: {image_dir}')
        index = len(image_parts)-1-image_parts[::-1].index('images')
        mask_dir = Path(*image_parts[:index])/mask_component/Path(*image_parts[index+1:])
        images = sorted(p for p in image_dir.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'})
        if not images:
            raise ValueError(f'No images in {image_dir}')
        counts = new_counts()
        for image in images:
            mask_path = (mask_dir/image.relative_to(image_dir)).with_suffix('.png')
            mask = cv2.imdecode(np.fromfile(mask_path,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2:
                raise ValueError(f'Expected single-channel class-index mask: {mask_path}')
            valid = set(int(k) for k in (raw_names.keys() if isinstance(raw_names,dict) else range(len(raw_names))))|{255}
            if not set(np.unique(mask).tolist()) <= valid:
                raise ValueError(f'Unexpected mask IDs: {mask_path}')
            for class_id in names:
                binary = (mask==class_id).astype(np.uint8)
                pixels = int(binary.sum())
                if not pixels:
                    continue
                _,_,stats,_ = cv2.connectedComponentsWithStats(binary,connectivity=8)
                areas = stats[1:,cv2.CC_STAT_AREA]
                areas = areas[areas>=args.min_area].tolist()
                for bucket in (counts,combined):
                    bucket[class_id]['images'] += 1
                    bucket[class_id]['pixels'] += pixels
                    bucket[class_id]['areas'].extend(areas)
        file_counts[split] = len(images)
        rows.extend(summarize(names,counts,split,args.min_area))
    rows.extend(summarize(names,combined,'all',args.min_area))
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'class_statistics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'dataset':str(config_path),'image_counts':file_counts,'connectivity':8,
              'area_unit':'pixels at stored mask resolution','quantile_method':'linear',
              'note':'Images and pixel counts include all class pixels. min-area filters regions and their quantiles only. Connected regions are not necessarily defect instances.',
              'statistics':rows}
    (args.output/'class_statistics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('split class images regions pixels area_p10 area_p50 area_p90')
    for row in rows:
        print(row['split'],row['class_name'],row['image_count'],row['region_count'],row['pixel_count'],row['area_p10'],row['area_p50'],row['area_p90'])
    print(f'Saved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
