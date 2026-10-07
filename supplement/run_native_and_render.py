"""Run remaining native CPU jobs sequentially, then render verified local results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NATIVE_PYTHON = Path('D:/pycharm/SketchQL-main/venv/Scripts/python.exe')
DATASETS = ('drtest','drtrain','bdd100kA','bdd100kB','I-24')


def run(strict,native,figures):
    strict,native,figures = map(Path,(strict,native,figures))
    status = json.loads((strict/'suite_status.json').read_text(encoding='utf-8'))
    if status.get('all_strict_groups_measured') is not True:
        raise ValueError('Finish all 80 strict groups before starting native measurements')
    if not (NATIVE_PYTHON.parents[1]/'Lib/site-packages/pandas').exists():
        raise RuntimeError('SketchQL environment needs pandas for the shared QST verifier; install it before native measurements')
    native.mkdir(parents=True,exist_ok=True)
    workers = []
    for dataset in DATASETS:
        command = [str(NATIVE_PYTHON),'-X','utf8','-u','-m','supplement.sketchql_native_adapter',
            '--dataset',dataset,'--output-dir',str(native.resolve()),'--strict-output',str(strict.resolve()),
            '--repetitions','3','--warmups','1','--threads','4']
        print(f'START NATIVE {dataset}',flush=True)
        with (native/f'{dataset}.log').open('a',encoding='utf-8') as log:
            process = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:
                log.write(line); log.flush()
                # Preserve full native progress in the log without flooding the task.
                if 'it/s]' not in line and 's/it]' not in line:
                    print(line,end='',flush=True)
            code = process.wait()
        workers.append(dict(dataset=dataset,returncode=code))
        (native/'worker_status.json').write_text(json.dumps(workers,indent=2),encoding='utf-8')
    command = [sys.executable,'-X','utf8','-m','supplement.render_full_baseline_suite',
        '--source',str(strict.resolve()),'--native-source',str(native.resolve()),'--output-dir',str(figures.resolve())]
    subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict-output',type=Path,default=ROOT/'supplement/out/full_baseline_suite_20260920')
    parser.add_argument('--native-output',type=Path,default=ROOT/'supplement/out/sketchql_native_full_20260920')
    parser.add_argument('--figures',type=Path,default=Path('<LOCAL_USER_HOME>/Desktop/论文/latex-project/figures'))
    args = parser.parse_args()
    run(args.strict_output,args.native_output,args.figures)
