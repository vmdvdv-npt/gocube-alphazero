"""One charged diagnostic pass: numerical, symmetry and tactical checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from session import BASE, load, reserve, replay, build_m137_five_channel_observation, write_json
from gocube_golden.rules import group_from_board, liberties_from_board, prepare_legal_actions, apply_action


def audit(model, checkpoint, starts):
    info = json.loads((checkpoint.parent/'writer.stdout.txt').read_text())
    reg = info['registers']
    states = [replay(m,1.5) for m in starts]
    obs = torch.stack([build_m137_five_channel_observation(s) for s in states])
    captured = []
    hook = model.blocks[-1].register_forward_hook(lambda m,i,o: captured.append(o.detach().clone()))
    with torch.inference_mode():
        policy,wdl = model(obs)
    hook.remove()
    nodes = captured[0]
    # Independent ideal arithmetic execution of the matrices, with fixed LN
    # cancellation replacing the state-dependent denominator. No model edits.
    with torch.inference_mode():
        ideal = F.linear(obs.double().transpose(1,2),model.input_projection.weight.double(),model.input_projection.bias.double())
        for block in model.blocks:
            normalized = ideal.clone()
            normalized[:,:,118:] = 0
            adjacent = normalized[:,block.message.neighbors,:].mean(2)
            hidden = F.relu(F.linear(normalized,block.message.self_linear.weight.double(),block.message.self_linear.bias.double())+
                            F.linear(adjacent,block.message.neighbor_linear.weight.double(),block.message.neighbor_linear.bias.double()))
            ideal += F.linear(hidden,block.output.weight.double(),block.output.bias.double())
    arithmetic_error = float((nodes[:,:,:116]-ideal[:,:,:116]).abs().max())
    # All 648 torus translations/dihedral transforms, not just rotations about
    # the central point. Test one nonsymmetric position in one charged batch.
    grid = np.arange(81).reshape(9,9)
    permutations = np.array([np.roll(np.roll(v,r,0),c,1).reshape(-1)
                            for v in [np.rot90(grid,k) for k in range(4)]+[np.rot90(grid.T,k) for k in range(4)]
                            for r in range(9) for c in range(9)])
    with torch.inference_mode():
        transformed = obs[-1,:,torch.tensor(permutations)].permute(1,0,2)
        p,w = model(transformed)
    expected = policy[-1,:81][torch.tensor(permutations)]
    symmetry_error = float((p[:,:81]-expected).abs().max())
    global_error = max(float((w-wdl[-1]).abs().max()),float((p[:,81]-policy[-1,81]).abs().max()))
    # Referee computations below are diagnostic labels only, never model input.
    errors, witnesses, captures = [],0,[]
    for si,state in enumerate(states):
        for point,stone in enumerate(state.stones):
            if not int(stone):
                continue
            color = 'own' if stone==state.side_to_move else 'opp'
            liberties = len(liberties_from_board(state,group_from_board(state,point)))
            for count in (2,3):
                if float(nodes[si,point,reg['cert%d_%s' % (count,color)]])>.9:
                    witnesses+=1
                    if liberties<count:
                        errors.append([si,point,count,liberties])
        for action in prepare_legal_actions(state).actions:
            if not isinstance(action,int):
                continue
            signal = float(nodes[si,action,reg['capture']])
            count = len(apply_action(state,action).captured)
            if signal>.9 and count==0:
                errors.append(['false capture',si,action,signal])
            if signal>.9 or count:
                captures.append({'position':si,'action':action,'signal':signal,'stones':count})
    report = {'status':'PASS' if arithmetic_error<.002 and symmetry_error<.002 and global_error<.002 and not errors else 'FAIL',
              'positions':len(states),'symmetries':648,'arithmetic_max_abs_error':arithmetic_error,
              'policy_symmetry_max_abs_error':symmetry_error,'global_symmetry_max_abs_error':global_error,
              'witnesses_checked':witnesses,'errors':errors,'capture_checks':captures,
              'registers':len(reg),'relu_units_per_block':info['relu_units_per_block'],
              'nonzero_parameters':info['nonzero_parameters']}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    model,digest = load(args.candidate)
    crafted = json.loads((BASE/'diagnostics-v000-20261103/continuation/positions.json').read_text())
    starts = [r['moves'] for r in crafted]+json.loads((args.session/'development.json').read_text())
    with reserve(args.session,'diagnostic',{'candidate_sha256':digest,'mode':'arithmetic-and-tactics'}) as folder:
        write_json(folder/'positions.json',starts)
        report = audit(model,args.candidate,starts)
        write_json(folder/'report.json',report)
    print(json.dumps({'round':folder.name,**report}),flush=True)
    if report['status']!='PASS':
        raise SystemExit(1)


if __name__=='__main__':
    main()
