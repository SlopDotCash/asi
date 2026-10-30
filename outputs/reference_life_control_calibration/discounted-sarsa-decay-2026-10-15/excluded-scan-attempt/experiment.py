"""Bounded nonpromoting diagnostic; no frozen scorecard or source is modified."""
from __future__ import annotations
import argparse, dataclasses, hashlib, importlib.metadata, json, platform, subprocess, time
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from alberta_framework.core.sarsa import SARSAAgent
from alberta_framework.reference_life import ExactDispatchConfig, ReferenceLifeMetricsConfig, ReferenceLifeRunner, RiverSwimReferenceEnvironment
from alberta_framework.reference_life_controls import DiscountedSARSAReferenceAdapter, DiscountedSARSAReferenceConfig
from alberta_framework.streams.closed_loop import RiverSwimConfig, RiverSwimMDP

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seed',type=int,choices=(100101,100102,100103),required=True)
parser.add_argument('--decay',type=int,choices=(2500,10000),required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
if args.out.exists(): raise SystemExit('Output already exists')
root=Path.cwd()
plan_path=Path(__file__).with_name('calibration-plan.json')
plan=json.loads(plan_path.read_text(encoding='utf-8'))
assert plan['horizon']==20000 and plan['scientific_promotion_allowed'] is False
cfg=DiscountedSARSAReferenceConfig.for_riverswim(RiverSwimConfig(),epsilon_decay_steps=args.decay)
adapter=DiscountedSARSAReferenceAdapter(cfg)
dispatch=ExactDispatchConfig(executor_id='asi.riverswim.executor',executor_epoch='asi.riverswim.executor_epoch.1')
env_adapter=RiverSwimReferenceEnvironment(RiverSwimConfig(),observation_spec=adapter.manifest.observation_spec,action_spec=adapter.manifest.action_spec,executor_id=dispatch.executor_id,executor_epoch=dispatch.executor_epoch)
runner=ReferenceLifeRunner.create(agent_adapter=adapter,environment_adapter=env_adapter,lifecycle_id=f'calibration.{args.seed}.{args.decay}',seed=args.seed,max_accepted_events=20000,dispatch_config=dispatch,metrics_config=ReferenceLifeMetricsConfig(mode='stationary'))
initial=runner.init()
agent=SARSAAgent(sarsa_config=cfg.core_config(),hidden_sizes=cfg.hidden_sizes,step_size=cfg.step_size,sparsity=cfg.sparsity,leaky_relu_slope=cfg.leaky_relu_slope,use_layer_norm=cfg.use_layer_norm,lamda=cfg.lamda)
env=RiverSwimMDP(RiverSwimConfig())
env_root=jr.fold_in(jr.key(args.seed,impl='threefry2x32'),0x415349)
def body(carry,index):
    state,environment=carry
    action=state.last_action
    observation,reward,next_env=env.step(environment,action,jr.fold_in(env_root,index))
    next_action,key=agent.select_action(state,observation)
    selected=state.replace(rng_key=key)
    result=agent.update(selected,reward,observation,jnp.asarray(False),next_action)
    return (result.state,next_env),(reward,action,next_env.state_index)
def scan(length):
    return jax.jit(lambda state,environment: jax.lax.scan(body,(state,environment),jnp.arange(length,dtype=jnp.uint32)))(initial.agent_state.agent_state,initial.environment_state)
started=time.monotonic()
prefix_final,prefix=scan(16)
jax.block_until_ready(prefix)
current=initial
host=[]
for _ in range(16):
    action=int(current.agent_state.agent_state.last_action)
    step=runner.step(current)
    if not step.accepted: raise RuntimeError(step.rejection_reason)
    current=step.state
    host.append((step.event.transaction.reward,action,int(current.environment_state.state_index)))
for observed,expected in zip(prefix,np.asarray(host).T,strict=True):
    if not np.array_equal(np.asarray(observed),expected): raise RuntimeError('16-step trajectory parity failed')
for a,b in zip(jax.tree.leaves(prefix_final[0]),jax.tree.leaves(current.agent_state.agent_state),strict=True):
    if jax.dtypes.issubdtype(a.dtype,jax.dtypes.prng_key): a,b=jr.key_data(a),jr.key_data(b)
    if not np.array_equal(np.asarray(a),np.asarray(b)): raise RuntimeError('16-step learner-state parity failed')
print('16-step exact reference-life parity passed',flush=True)
final,trace=scan(20000)
jax.block_until_ready(trace)
rewards,actions,positions=map(np.asarray,trace)
if len(rewards)!=20000 or not np.isfinite(rewards).all(): raise RuntimeError('Invalid reward trace')
if int(final[0].step_count)!=20000 or int(final[1].step_count)!=20000: raise RuntimeError('Counter mismatch')
def resource(state):
    arrays=[]
    for leaf in jax.tree.leaves(state):
        if jax.dtypes.issubdtype(leaf.dtype,jax.dtypes.prng_key): leaf=jr.key_data(leaf)
        arrays.append(np.asarray(leaf))
    return {'array_elements':sum(a.size for a in arrays),'persistent_numeric_bytes':sum(a.nbytes for a in arrays)}
sources={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'alberta_framework').rglob('*.py'))}
for p in (root/'pyproject.toml',root/'uv.lock'): sources[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
result={
    'schema':'asi.riverswim_sarsa_decay_diagnostic.shard.v1','development_only':True,'scientific_promotion_allowed':False,
    'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_sha256':sources,
    'seed':args.seed,'epsilon_decay_steps':args.decay,'config':cfg.to_config(),'environment':dataclasses.asdict(RiverSwimConfig()),
    'horizon':20000,'exact_reference_prefix_steps':16,'mean_reward':float(rewards.astype(np.float64).mean()),
    'late_mean_reward':float(rewards[-2000:].astype(np.float64).mean()),'rightmost_state_visits':int((positions==5).sum()),
    'action_counts':np.bincount(actions,minlength=2).tolist(),'resources':{'initial':resource((initial.agent_state.agent_state,initial.environment_state)),'final':resource(final)},
    'runtime':{'python':platform.python_version(),'system':platform.system(),'machine':platform.machine(),'backend':jax.default_backend(),'versions':{n:importlib.metadata.version(n) for n in ('jax','jaxlib','numpy','chex')}},
    'wall_seconds_telemetry':time.monotonic()-started,'rewards':rewards.astype(np.float64).tolist(),'actions':actions.tolist(),'positions':positions.tolist(),
    'limit':'Numerical diagnostic only; not full ReferenceLifeRunner execution or frozen scorecard qualification.'}
args.out.parent.mkdir(parents=True,exist_ok=True)
with args.out.open('x',encoding='utf-8') as f: json.dump(result,f,sort_keys=True,allow_nan=False); f.write('\n')
print(json.dumps({k:result[k] for k in ('seed','epsilon_decay_steps','mean_reward','late_mean_reward','rightmost_state_visits','wall_seconds_telemetry')}),flush=True)