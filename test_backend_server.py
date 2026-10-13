import os
os.environ['API_ACCESS_KEY']='test-only-backend-key-1234567890'
os.environ['ALLOWED_ORIGINS']='https://lucky-scone-171aa5.netlify.app'
import app
import io
import time
from unittest.mock import patch
client=app.app.test_client()
headers={'Authorization':'Bearer '+os.environ['API_ACCESS_KEY']}
assert client.get('/health').status_code==200
assert client.get('/api/dataset').status_code==401
assert client.get('/api/dataset',headers=headers).status_code==200
assert client.options('/api/analyze',headers={'Origin':os.environ['ALLOWED_ORIGINS']}).headers['Access-Control-Allow-Origin']==os.environ['ALLOWED_ORIGINS']
assert 'Access-Control-Allow-Origin' not in client.get('/health',headers={'Origin':'https://untrusted.example'}).headers
job=next(k for k,v in app.jobs.items() if v['status']=='done')
r=client.get('/api/jobs/'+job,headers=headers).json
assert client.get('/files/'+job+'/results.csv').status_code==401
assert client.get(r['result']['csv']).status_code==200
assert client.get(r['result']['video'],headers={'Range':'bytes=0-1023'}).status_code==206
assert client.get(r['result']['csv']+'broken').status_code==401
with patch('itsdangerous.timed.time.time',return_value=time.time()+3601):
    assert client.get(r['result']['csv']).status_code==401
assert 'attachment' in client.get(r['result']['video']+'&download=1').headers['Content-Disposition']
slots=[]
while app.admission.acquire(blocking=False): slots.append(True)
assert client.post('/api/analyze',headers=headers,data={'height':'165','video':(io.BytesIO(b'bad'),'test.mp4')}).status_code==429
for _ in slots: app.admission.release()
print('Health, authorization, CORS, signed files, download, queue-full checks passed')
