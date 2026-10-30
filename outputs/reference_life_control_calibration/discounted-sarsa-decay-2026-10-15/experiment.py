"""Full reference-life SARSA decay diagnostic, permanently nonpromoting."""
from __future__ import annotations
import argparse,dataclasses,hashlib,importlib.metadata,json,platform,subprocess,time
from pathlib import Path
import numpy as np
import jax
from alberta_framework.reference_life import ExactDispatchConfig,LifePhase,ReferenceLifeMetricsConfig,ReferenceLifeRunner,RiverSwimReferenceEnvironment
from alberta_framework.reference_life_controls import DiscountedSARSAReferenceAdapter,DiscountedSARSAReferenceConfig,control_state_resource_usage
from alberta_framework.streams.closed_loop import RiverSwimConfig
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--seed',type=int,choices=(100201,100202,100203),required=True)
p.add_argument('--decay',type=int,choices=(2500,10000),required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists(): raise SystemExit('Output exists')
root=Path.cwd(); plan_path=Path(__file__).with_name('canonical-calibration-plan.json'); plan=json.loads(plan_path.read_text(encoding='utf-8'))
assert plan['horizon']==20000 and plan['scientific_promotion_allowed'] is False
cfg=DiscountedSARSAReferenceConfig.for_riverswim(RiverSwimConfig(),epsilon_decay_steps=a.decay)
adapter=DiscountedSARSAReferenceAdapter(cfg)
dispatch=ExactDispatchConfig(executor_id='asi.riverswim.executor',executor_epoch='asi.riverswim.executor_epoch.1')
environment=RiverSwimReferenceEnvironment(RiverSwimConfig(),observation_spec=adapter.manifest.observation_spec,action_spec=adapter.manifest.action_spec,executor_id=dispatch.executor_id,executor_epoch=dispatch.executor_epoch)
runner=ReferenceLifeRunner.create(agent_adapter=adapter,environment_adapter=environment,lifecycle_id=f'calibration2.{a.seed}.{a.decay}',seed=a.seed,max_accepted_events=20000,dispatch_config=dispatch,metrics_config=ReferenceLifeMetricsConfig(mode='stationary'))
started=time.monotonic(); state=runner.init(); initial_resource=dataclasses.asdict(control_state_resource_usage(state.agent_state)); rows=[]
while state.phase is LifePhase.QUIESCENT:
    action=int(state.agent_state.agent_state.last_action)
    result=runner.step(state)
    if not result.accepted: raise RuntimeError(result.rejection_reason)
    state=result.state
    rows.append((result.event.transaction.reward,action,int(state.environment_state.state_index)))
    if state.accepted_events%1000==0: print(f'seed={a.seed} decay={a.decay} accepted={state.accepted_events}/20000 seconds={time.monotonic()-started:.1f}',flush=True)
if state.phase is not LifePhase.COMPLETED or state.accepted_events!=20000: raise RuntimeError('Life did not complete')
trace=np.asarray(rows,dtype=np.float64); rewards=trace[:,0]
if not np.isfinite(trace).all(): raise RuntimeError('Nonfinite trace')
sources={f.relative_to(root).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((root/'alberta_framework').rglob('*.py'))}
for f in (root/'pyproject.toml',root/'uv.lock'): sources[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
r={'schema':'asi.riverswim_sarsa_decay_diagnostic.shard.v2','development_only':True,'scientific_promotion_allowed':False,'reference_dev_population_allowed':False,'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_sha256':sources,'seed':a.seed,'epsilon_decay_steps':a.decay,'config':cfg.to_config(),'life_config':runner.config.config,'environment':dataclasses.asdict(RiverSwimConfig()),'accepted_events':state.accepted_events,'executed_events':state.executed_events,'dispatch_attempts':state.dispatch_attempts,'environment_rng_cursor':state.environment_rng_cursor,'transcript_sha256':state.transcript_sha256,'parameter_change_events':state.metrics.parameter_change_events,'mean_reward':float(rewards.mean()),'late_mean_reward':float(rewards[-2000:].mean()),'rightmost_state_visits':int((trace[:,2]==5).sum()),'action_counts':np.bincount(trace[:,1].astype(np.int64),minlength=2).tolist(),'resources':{'initial':initial_resource,'final':dataclasses.asdict(control_state_resource_usage(state.agent_state))},'runtime':{'python':platform.python_version(),'system':platform.system(),'machine':platform.machine(),'backend':jax.default_backend(),'versions':{n:importlib.metadata.version(n) for n in ('jax','jaxlib','numpy','chex')}},'wall_seconds_telemetry':time.monotonic()-started,'rewards':rewards.tolist(),'actions':trace[:,1].astype(np.int64).tolist(),'positions':trace[:,2].astype(np.int64).tolist(),'limit':'New development diagnostic with fresh development seeds. Not qualification of frozen scorecard v1, scientific evidence, or reference-dev promotion.'}
a.out.parent.mkdir(parents=True,exist_ok=True)
with a.out.open('x',encoding='utf-8') as f: json.dump(r,f,sort_keys=True,allow_nan=False); f.write('\n')
print(json.dumps({k:r[k] for k in ('seed','epsilon_decay_steps','mean_reward','late_mean_reward','rightmost_state_visits','wall_seconds_telemetry')}),flush=True)