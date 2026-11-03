"""Execute only the predeclared, frozen holdout comparisons; never tune weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from session import BASE, HERE, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args = p.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('Use 1..4 independent execution workers')
    frozen = json.loads((args.session/'freeze.json').read_text())
    plan = json.loads((args.session/'session.json').read_text())
    models = {'v002':frozen['candidate'],'v001':str(BASE/'versions/v001/checkpoint.pt'),
              'M269':str(BASE/'reference-lock.json')}
    existing = [json.loads(p.read_text()) for p in (args.session/'rounds').glob('*/request.json')]
    reserved = {(r.get('comparison'),r.get('chunk')) for r in existing if r['phase']=='holdout'}
    jobs = [(name,chunk) for name in plan['holdout_comparisons'] for chunk in range(8)
            if (name,chunk) not in reserved]
    logs = args.session/'execution-logs'
    logs.mkdir(exist_ok=True)

    def execute(job):
        name,chunk = job
        left,right,mode = name.split('-')
        command = [sys.executable,str(HERE/'session.py'),'arena','--session',str(args.session),
                   '--candidate',models[left],'--opponent',models[right],
                   '--phase','holdout','--comparison',name,'--chunk',str(chunk),
                   '--simulations','64' if mode=='s64' else '0','--device','cuda']
        with (logs/('%s-%d.log' % (name,chunk))).open('x') as stream:
            result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        return {'comparison':name,'chunk':chunk,'returncode':result.returncode}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in as_completed([pool.submit(execute,job) for job in jobs]):
            outcome = result.result()
            print(json.dumps(outcome),flush=True)
    # Failed reservations stay consumed and visible. There is no selective
    # rerun. A subsequent invocation can dispatch only previously unreserved jobs.


if __name__=='__main__':
    main()
