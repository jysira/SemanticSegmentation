"""Move existing review files into primary-error categories, preserving edits."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'reports/manual_semantic_review'


def main():
    path=WORK/'manifest.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    for item in manifest:
        primary=next(k for k in ('missed','wrong_class','boundary_shift','false_positive') if k in item['errors'].split(','))
        original=WORK/item['json']
        folder=WORK/'editable'/primary/item['split']
        target=folder/original.name
        if original!=target:
            folder.mkdir(parents=True,exist_ok=True)
            image=original.parent/item['image']
            for source,destination in ((image,folder/image.name),(original,target)):
                if destination.exists():
                    raise FileExistsError(destination)
                if not source.is_file():
                    raise FileNotFoundError(source)
            image.rename(folder/image.name)
            original.rename(target)
        item['json']=str(target.relative_to(WORK))
        item['primary_error']=primary
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with (WORK/'review_list.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=['split','image','primary_error','errors','holes','json']
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:item[k] for k in fields} for item in manifest)
    for split in ('train','val','test'):
        folder=WORK/'editable'/split
        if folder.exists() and not any(folder.iterdir()):
            folder.rmdir()
    assert all((WORK/r['json']).is_file() for r in manifest)
    assert all((WORK/r['json']).parent.joinpath(r['image']).is_file() for r in manifest)
    print(dict(Counter(r['primary_error'] for r in manifest)))
    print('Verified pairs:',len(manifest))


if __name__=='__main__':
    main()
