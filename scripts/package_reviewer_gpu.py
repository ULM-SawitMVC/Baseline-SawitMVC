"""Package the uncommitted reviewer experiment for a GPU notebook.

Images and trained weights are excluded. Notebook execution is an author action;
packaging does not launch training, reserve hardware or incur cloud charges.
"""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'draft-icic-2026/Accepted-Revise'


def cell(kind,text):
    c={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':
        c.update(execution_count=None,outputs=[])
    return c


def main():
    paths=[ROOT/'pipeline/build_counting_features.py',ROOT/'ground_truth/split_manifest.csv']
    paths += [ROOT/'experiments/revision'/name for name in
        ['revcommon.py','a2_error_decomposition.py','a10_revision_audit.py','a12_second_detector.py']]
    paths += sorted((ROOT/'ground_truth/annotations').glob('*.json'))
    paths += sorted((ROOT/'predictions/y26mv2_per_tree').glob('*.json'))
    bundle=OUT/'reviewer-gpu-bundle.zip'
    with ZipFile(bundle,'w',ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p,p.relative_to(ROOT).as_posix())
    cells=[cell('markdown',
        '# Reviewer 4: additional detector architecture\n\n'
        'This notebook has **not been run** as part of the local revision. Choose a CUDA GPU runtime. '
        'It trains YOLO11m on 716 training trees, selects the checkpoint using 96 validation trees, '
        'then evaluates 141 test trees with the same Ridge+F0 counter used for YOLO26m. '
        'Images require access to the SawitMVC Hugging Face dataset. '
        'Architecture-specific defaults are saved; this is not a claim that every training detail is identical.\n\n'
        'Upload `reviewer-gpu-bundle.zip` in the next cell. The bundle contains revision code, '
        'GT annotations and the reference detector cache; no credentials or images.'),
        cell('code',"%pip install -q ultralytics==8.4.52 scikit-learn==1.8.0 scipy pandas numpy huggingface_hub pyyaml\n"),
        cell('code',"from pathlib import Path\nfrom zipfile import ZipFile\nfrom google.colab import files\nimport os, torch\nassert torch.cuda.is_available(), 'Select a GPU runtime first'\nprint(torch.cuda.get_device_name(0))\nuploaded = files.upload()\nassert 'reviewer-gpu-bundle.zip' in uploaded\nworkspace = Path('/content/sawit-reviewer').resolve()\nworkspace.mkdir(exist_ok=True)\nwith ZipFile('reviewer-gpu-bundle.zip') as archive:\n    for item in archive.infolist():\n        target = (workspace / item.filename).resolve()\n        assert target.is_relative_to(workspace), 'Unsafe archive member'\n    archive.extractall(workspace)\nos.chdir(workspace)\n"),
        cell('code',"from huggingface_hub import login, snapshot_download\nlogin()  # authenticate in this notebook, never put a token in the manuscript\nsnapshot_download('ULM-DS-Lab/SawitMVC-YOLO', repo_type='dataset', local_dir='SawitMVC-YOLO', token=True)\n"),
        cell('code',"!python experiments/revision/a12_second_detector.py --prepare-only\n"),
        cell('markdown','## Train and evaluate\n\nThe next cell starts 60-epoch training (batch 32, imgsz 640, seed 42). '
             'Do not tune on test results. If interrupted after training, resume inference with '
             '`--weights runs/reviewer-yolo11/train/weights/best.pt --device 0`; cached trees are checked against the checkpoint hash.'),
        cell('code',"!python experiments/revision/a12_second_detector.py --device 0\n"),
        cell('code',"import json\nfrom pprint import pprint\nmetrics = Path('results/revision/a12_second_detector_metrics.json')\nassert metrics.exists(), 'Training/evaluation did not finish; inspect the output above'\npprint(json.loads(metrics.read_text()))\n"),
        cell('code',"from zipfile import ZIP_DEFLATED\nresult_zip = Path('/content/reviewer-yolo11-results.zip')\nwith ZipFile(result_zip, 'w', ZIP_DEFLATED) as archive:\n    for directory in ['results/revision', 'predictions', 'runs/reviewer-yolo11']:\n        for path in Path(directory).rglob('*'):\n            if path.is_file() and 'y26mv2_per_tree' not in path.parts:\n                archive.write(path, path.as_posix())\nfiles.download(str(result_zip))\n"),
        cell('markdown','Return the output bundle to update Table VI, the scope discussion, and the response to Reviewer 4. '
             'Do not mark the second architecture request completed before the metrics and checkpoint provenance are available.\n\n'
             'Model documentation: https://docs.ultralytics.com/models/yolo11/')]
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}
    (OUT/'Reviewer-4-GPU.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    print(f'Packaged {len(paths)} files, {bundle.stat().st_size/1048576:.1f} MiB; notebook created, not executed.')


if __name__=='__main__':
    main()
