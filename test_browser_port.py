import app, json, math
import numpy as np
from pathlib import Path
from lift_counter import LiftCounter
import pandas as pd
cases=[]
for path in (app.ROOT/'data').glob('test-*/results.csv'):
    d=pd.read_csv(path);fps=1/(d.time_s.iloc[1]-d.time_s.iloc[0]);counter=LiftCounter(fps)
    values=[float(v) for v in d.V_cm];expected=[counter.update(v,i) for i,v in enumerate(values)]
    cases.append({'name':path.parent.name,'fps':fps,'values':[v if math.isfinite(v) else None for v in values],'expected':expected})
# Real calibrated body coordinates from the original geometry, not dummy metrics.
video=next((app.ROOT/'dataset_for_test').rglob('G16_S01_BK-WS_RF_T1_PF_01.mp4'))
cap,info=app.model.open_video(video);app.model.FPS=info['fps'];smooth=app.model.PoseSmoother(info['width'],info['height']);samples=[]
with app.model.create_pose_detector() as detector:
    for i in range(150):
        ok,frame=cap.read()
        if not ok:break
        xy,world,conf=app.model.extract_keypoints(detector,frame,i)
        xys,ws,observed=smooth.update(xy,world,conf)
        if i<round(1.5*info['fps']): samples.append({'xy':xys,'world':ws,'observed':observed})
        if i==round(1.5*info['fps'])-1:calib=app.model.calibrate(samples,165,info['width'],info['height'])
    P=ws*100*calib['world_scale'];pose=app.model.solve_camera_pose(P,xys,observed,calib['K']);C=app.model.to_camera_3d(xys,P,pose,calib['K']);m=app.model.measure_hands(C,xys,calib)
cap.release()
geom={'C':C.tolist(),'xy':xys.tolist(),'world':ws.tolist(),'calib':{'up':calib['up'].tolist(),'floor':calib['floor'].tolist(),'knee':float(calib['knee_cm']),'waist':float(calib['waist_cm']),'shoulder':float(calib['shoulder_cm'])},'expected':{'H':m['H'],'V':m['V'],'limit':m['limit_kg'],'twist':app.model.twist_angle(ws)}}
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
(app.ROOT/'browser-port-fixture.json').write_text(json.dumps(clean({'counter':cases,'geometry':geom}),allow_nan=False),encoding='utf-8')
print('Python reference exported:',len(cases),'real clip signals')
