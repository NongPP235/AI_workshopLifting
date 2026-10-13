import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
import pandas as pd
import numpy as np
from lift_counter import LiftCounter

def run(values):
    counter = LiftCounter(30)
    return [counter.update(float(v),i) for i,v in enumerate(values)]

cycle = np.concatenate([np.full(30,80),np.linspace(80,30,30),np.linspace(30,140,45),np.full(30,140),np.linspace(140,80,45),np.full(30,80)])
assert run(cycle)[-1] == 1
assert run(np.concatenate([cycle,cycle]))[-1] == 2
assert run(np.linspace(140,30,90))[-1] == 0
assert run(80+np.sin(np.arange(300)) * 6)[-1] == 0
assert run(np.concatenate([np.full(30,80),np.full(30,np.nan),np.full(30,140)]))[-1] == 0
assert run(np.concatenate([cycle[:115],np.tile([140,110,140,125],20)]))[-1] == 1
root = Path(__file__).parent
rows=[]
for path in sorted((root/'data').glob('test-*/results.csv')):
    df=pd.read_csv(path)
    fps=1/(df.time_s.iloc[1]-df.time_s.iloc[0])
    counter=LiftCounter(fps)
    counts=[counter.update(float(row.V_cm),i) for i,row in df.iterrows()]
    rows.append({'clip':path.parent.name,'old':int(df.lift_count_realtime.iloc[-1]),'new':counts[-1]})
    if path.parent.name=='test-G16_S01_BK-WS_RF_T1_PF_01':
        assert counts[-1]==1
        trigger=next(i for i,n in enumerate(counts) if n)
        assert df.V_cm.iloc[trigger]>df.V_cm.iloc[max(0,trigger-5)]
pd.DataFrame(rows).to_csv(root/'count-regression.csv',index=False)
raw_path=root/'data'/'7b73b7f082c44a6cb5fc6e9428dd126a'/'results.csv'
if raw_path.exists():
    df=pd.read_csv(raw_path)
    assert run(df.V_cm.values)[-1]==1
print(pd.DataFrame(rows).to_string(index=False))
print('Upward, repeat cycle, downward-only, jitter, lock and target clip passed')
