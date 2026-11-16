"""Migrate text references from legacy run aliases to numbered directories."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING = {
    'semantic_clahe_eval':'03_evaluation/clahe',
    'semantic_merged_eval':'03_evaluation/merged',
    'semantic_clahe':'02_segmentation/clahe',
    'semantic_merged':'02_segmentation/merged',
    'deploy_verify':'04_deployment/verification',
    'semantic':'02_segmentation/legacy',
    'train':'01_detection/training',
    'val':'01_detection/evaluation',
}


def main():
    files = subprocess.check_output(['rg','--files','--hidden','-g','!.git','-g','!.venv'],cwd=ROOT,text=True,encoding='utf-8').splitlines()
    files += subprocess.check_output(['rg','--files','--no-ignore','runs'],cwd=ROOT,text=True,encoding='utf-8').splitlines()
    changed = []
    for relative in files:
        path = ROOT/relative
        if path.resolve()==Path(__file__).resolve() or path.suffix.lower() not in {'.py','.yaml','.yml','.json','.csv','.md','.txt','.log','.ps1','.cmd'}:
            continue
        try:
            text = path.read_text(encoding='utf-8-sig')
        except (UnicodeError,OSError):
            continue
        original = text
        for old,new in MAPPING.items():
            for separator in ('/','\\\\','\\'):
                text = text.replace('runs'+separator+old+separator,'runs'+separator+new.replace('/',separator)+separator)
            # Handle common pathlib expressions and quoted path endings.
            for quote in ("'",'"'):
                text = text.replace('runs/'+old+quote,'runs/'+new+quote)
                text = text.replace(quote+'runs'+quote+' / '+quote+old+quote,
                                    quote+'runs'+quote+' / '+quote+new+quote)
        if text!=original:
            path.write_text(text,encoding='utf-8')
            changed.append(relative)
    print('Updated files:',len(changed))
    print('\n'.join(changed[:35]))


if __name__=='__main__':
    main()
