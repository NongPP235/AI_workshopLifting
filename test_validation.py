import app
import io
import numpy as np
import uuid

client = app.app.test_client()
assert client.get('/').status_code == 200
assert client.get('/app.js').status_code == 200
assert client.post('/api/analyze', data={'height':'nan'}).status_code == 400
assert client.post('/api/analyze', data={'height':'175', 'video':(io.BytesIO(b'bad'), 'bad.txt')}).status_code == 400
assert client.get('/api/jobs/does-not-exist').status_code == 404
assert client.get('/files/does-not-exist/results.csv').status_code == 404
for seconds in (1, 2):
    job_id = 'validation-' + uuid.uuid4().hex
    folder = app.DATA / job_id
    folder.mkdir()
    path = folder / 'input.mp4'
    writer = app.model.cv2.VideoWriter(str(path),app.model.cv2.VideoWriter_fourcc(*'mp4v'),30,(320,240))
    for _ in range(seconds*30): writer.write(np.zeros((240,320,3),dtype=np.uint8))
    writer.release()
    app.jobs[job_id] = {'status':'queued'}
    app.process(job_id,path,175)
    assert app.jobs[job_id]['status'] == 'error'
    assert (folder / 'job.json').exists()
    assert client.get('/api/jobs/'+job_id).status_code == 200
print('Validation passed: assets, invalid height/type, missing jobs, short clip, no-person clip, error persistence')
