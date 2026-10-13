"""Integration tests using real MediaPipe/video processing, with no model mock."""
import app
import csv, io, json, time, subprocess, sys
from pathlib import Path

report=[]
client=app.app.test_client()
def check(name, condition, detail=''):
    report.append({'case':name,'passed':bool(condition),'detail':detail})
    (app.ROOT/'critical-flow-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    assert condition,(name,detail)
    print('PASS',name,flush=True)

for url in ['/','/app.js','/style.css','/fonts/fonts.css','/fonts/noto-sans-thai-400.ttf','/fonts/noto-sans-thai-700.ttf']:
    r=client.get(url)
    check('asset '+url,r.status_code==200 and len(r.data)>0)
clips=client.get('/api/dataset').json
check('dataset heights',all(any(c['height']==h and '_S'+s+'_' in c['name'] for c in clips) for s,h in app.HEIGHTS.items()))
for height in ['','nan','inf','49','251','not-a-number']:
    check('invalid height '+repr(height),client.post('/api/analyze',data={'height':height}).status_code==400)
check('missing file',client.post('/api/analyze',data={'height':'165'}).status_code==400)
check('wrong extension',client.post('/api/analyze',data={'height':'165','video':(io.BytesIO(b'x'),'bad.exe')}).status_code==400)
for body in [[1],{'id':-1,'height':165},{'id':99999,'height':165},{'id':0,'height':None},{'id':'oops','height':165}]:
    check('invalid dataset JSON '+repr(body),client.post('/api/analyze-dataset',json=body).status_code==400)
check('malformed JSON',client.post('/api/analyze-dataset',data='{',content_type='application/json').status_code==400)
saved_limit=app.app.config['MAX_CONTENT_LENGTH']
app.app.config['MAX_CONTENT_LENGTH']=1024
r=client.post('/api/analyze',data={'height':'165','video':(io.BytesIO(b'x'*2048),'large.mp4')})
app.app.config['MAX_CONTENT_LENGTH']=saved_limit
check('oversize response',r.status_code==413 and 'error' in r.json,'Test limit set to 1 KiB rather than allocating 500 MB')
check('unknown job',client.get('/api/jobs/missing').status_code==404)
check('unknown file',client.get('/files/missing/result.mp4').status_code==404)
model_path=app.model.POSE_MODEL_PATH
app.model.POSE_MODEL_PATH=app.ROOT/'missing-test-model.task'
check('missing model',client.post('/api/analyze',data={'height':'165','video':(io.BytesIO(b'x'),'test.mp4')}).status_code==503)
app.model.POSE_MODEL_PATH=model_path

video=next((app.ROOT/'dataset_for_test').rglob('G16_S01_BK-WS_RF_T1_PF_01.mp4'))
with video.open('rb') as f:
    r=client.post('/api/analyze',data={'height':'165','video':(f,video.name)})
check('upload accepted',r.status_code==202)
first=r.json['id']
check('upload metadata retained',client.get('/api/jobs/'+first).json['height']==165 and app.jobs[first]['filename']==video.name)
deadline=time.monotonic()+180
while app.jobs[first]['status']=='queued' and time.monotonic()<deadline: time.sleep(.05)
sample=next(c for c in clips if c['name']=='G16_S02_BK-WS_RM_T0_PN_01.mp4')
r=client.post('/api/analyze-dataset',json={'id':sample['id'],'height':183})
check('dataset accepted',r.status_code==202)
second=r.json['id']
check('dataset metadata retained',client.get('/api/jobs/'+second).json['height']==183 and app.jobs[second]['filename']==sample['name'])
check('second job queued',app.jobs[second]['status']=='queued')
check('unfinished download blocked',client.get('/files/'+first+'/results.csv').status_code==409)
while any(app.jobs[j]['status'] in {'queued','processing'} for j in (first,second)) and time.monotonic()<deadline:
    time.sleep(.25)
for j in (first,second):
    status=client.get('/api/jobs/'+j).json
    check('analysis completes '+j,status['status']=='done',status.get('error',''))
    result=status['result']
    check('one upward lift '+j,result['lifts']==1 and result['counter_version']=='upward-cycle-v2')
    csv_response=client.get(result['csv'])
    rows=list(csv.DictReader(io.StringIO(csv_response.data.decode('utf-8-sig'))))
    check('CSV matches summary '+j,csv_response.status_code==200 and len(rows)==result['frames'] and int(rows[-1]['lift_count_realtime'])==result['lifts'])
    r=client.get(result['video'],headers={'Range':'bytes=0-1023'})
    check('video seek range '+j,r.status_code==206 and len(r.data)==1024 and 'Content-Range' in r.headers)
    check('restricted download name '+j,client.get('/files/'+j+'/input.mp4').status_code==404)
    code="import app,json; print(json.dumps(app.jobs['"+j+"']))"
    restarted=subprocess.run([sys.executable,'-c',code],cwd=app.ROOT,capture_output=True,text=True,encoding='utf-8',check=True)
    saved=json.loads(restarted.stdout)
    check('restart restores results '+j,saved['status']=='done' and saved['result']['lifts']==1)
bad=client.post('/api/analyze',data={'height':'165','video':(io.BytesIO(b'not-a-video'),'broken.mp4')}).json['id']
while app.jobs[bad]['status'] in {'queued','processing'} and time.monotonic()<deadline: time.sleep(.1)
check('corrupt clip handled',app.jobs[bad]['status']=='error' and client.get('/api/jobs/'+bad).status_code==200)
check('failed download blocked',client.get('/files/'+bad+'/results.csv').status_code==409)
print(f'FINISHED {len(report)} checks',flush=True)
