import sys
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
import json, threading, uuid, subprocess, traceback, shutil, re
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import model
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
model.plt = plt
model.find_peaks = find_peaks

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
model.POSE_MODEL_PATH = ROOT / 'models' / 'pose_landmarker_lite.task'
jobs = {}
lock = threading.Lock()
HEIGHTS = {'01':165, '02':183, '03':180, '04':178, '05':173}
dataset = sorted((ROOT / 'dataset_for_test').rglob('*.mp4'))
def normalize_job(job):
    result = job.get('result',{})
    if 'realtime_lifts' in result:
        result.setdefault('robust_lifts', result['lifts'])
        result['lifts'] = result['realtime_lifts']
    return job
def save_job(job_id):
    folder = DATA / job_id
    if folder.exists():
        temp = folder / 'job.tmp'
        temp.write_text(json.dumps(jobs[job_id], ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temp.replace(folder / 'job.json')

for metadata in DATA.glob('*/job.json'):
    try:
        saved = normalize_job(json.loads(metadata.read_text('utf-8')))
        if saved['status'] in {'queued','processing'}:
            saved.update(status='error', error='เซิร์ฟเวอร์เริ่มใหม่ กรุณาส่งคลิปอีกครั้ง')
        jobs[metadata.parent.name] = saved
    except (ValueError, KeyError):
        pass

@app.get('/')
def index():
    return app.send_static_file('index.html')

@app.after_request
def cache_control(response):
    if request.path.startswith('/api/') or request.path in {'/','/app.js','/style.css'}:
        response.headers['Cache-Control'] = 'no-store'
    return response

@app.errorhandler(413)
def too_large(error):
    return jsonify(error='ไฟล์ต้องมีขนาดไม่เกิน 500 MB'), 413

def process(job_id, path, height, output_dir=None):
    job = jobs[job_id]
    job['height'] = height
    try:
        with lock:
            job['status'] = 'processing'
            save_job(job_id)
            cap, info = model.open_video(path)
            cap.release()
            if info['frames'] / info['fps'] <= model.CALIB_SECONDS:
                raise ValueError('วิดีโอต้องยาวกว่า 1.5 วินาที และยืนนิ่งให้เห็นทั้งตัวช่วงเริ่มต้น')
            model.FPS = info['fps']
            output_dir = output_dir or path.parent
            raw = output_dir / 'raw.mp4'
            def progress(frame,total):
                job.update(progress=round(frame/total*95), frame=frame, total_frames=total, phase='กำลังวิเคราะห์ท่าทาง')
            results, calib = model.run_pipeline(path, raw, height, progress=progress)
            valid = results.dropna(subset=['limit_kg','H_cm','V_cm']) if 'limit_kg' in results else results.iloc[:0]
            if valid.empty or calib is None or not np.isfinite(calib['cm_per_px']):
                raise ValueError('ไม่พบข้อมูลท่าทางที่เพียงพอ กรุณาใช้คลิปที่เห็นทั้งตัวและยืนนิ่ง 1.5 วินาทีแรก')
            count, _ = model.count_lifts_robust(results, calib)
            plt.close('all')
            job.update(progress=96, phase='กำลังเตรียมวิดีโอสำหรับเล่นบนเว็บ')
            results.to_csv(output_dir / 'results.csv', index=False, encoding='utf-8-sig')
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(raw), '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-movflags', '+faststart', str(output_dir / 'result.mp4')], check=True, capture_output=True)
            reach = valid['reach_class'].mode().iloc[0]
            job.update(status='done', result={'twist': round(float(valid['twist_deg'].mean()),1), 'h':round(float(valid['H_cm'].mean()),1), 'lifts':count, 'limit':float(valid['limit_kg'].min()), 'orientation':round(float(valid['orientation_deg'].mean()),1), 'height':height, 'hmin':round(float(valid['H_cm'].min()),1), 'hmax':round(float(valid['H_cm'].max()),1), 'reach':reach, 'video':f'/files/{job_id}/result.mp4', 'csv':f'/files/{job_id}/results.csv'})
            job['result'].update(duration=round(info['frames']/info['fps'],1), coverage=round(len(valid)/max(1,len(results)-round(model.CALIB_SECONDS*info['fps']))*100,1), realtime_lifts=int(results['lift_count_realtime'].iloc[-1]), frames=len(results), filename=job.get('filename', path.name))
            job['progress'] = 100
            job['result']['counter_version'] = 'upward-cycle-v2'
            normalize_job(job)
            save_job(job_id)
    except Exception as exc:
        traceback.print_exc()
        if isinstance(exc, FileNotFoundError):
            message = 'อ่านวิดีโอไม่ได้ ไฟล์อาจเสียหรือรูปแบบไม่รองรับ กรุณาเลือกคลิปใหม่'
        elif isinstance(exc, subprocess.CalledProcessError):
            message = 'เตรียมวิดีโอผลลัพธ์ไม่สำเร็จ กรุณาลองวิเคราะห์ใหม่'
        else:
            message = str(exc)
        job.update(status='error', error=message)
        save_job(job_id)

@app.post('/api/analyze')
def analyze():
    upload = request.files.get('video')
    try:
        height = float(request.form.get('height', ''))
        if not 50 <= height <= 250 or not np.isfinite(height): raise ValueError()
    except ValueError:
        return jsonify(error='กรอกส่วนสูงระหว่าง 50–250 ซม.'), 400
    if not upload or Path(upload.filename).suffix.lower() not in {'.mp4','.mov','.avi','.webm'}:
        return jsonify(error='เลือกวิดีโอ MP4, MOV, AVI หรือ WebM'), 400
    if not model.POSE_MODEL_PATH.exists():
        return jsonify(error='ยังไม่มีโมเดล MediaPipe กรุณารัน setup.ps1'), 503
    job_id = uuid.uuid4().hex
    folder = DATA / job_id
    folder.mkdir()
    path = folder / ('input' + Path(upload.filename).suffix.lower())
    upload.save(path)
    jobs[job_id] = {'status':'queued', 'filename':upload.filename, 'height':height}
    save_job(job_id)
    threading.Thread(target=process, args=(job_id,path,height), daemon=True).start()
    return jsonify(id=job_id), 202

@app.get('/api/dataset')
def list_dataset():
    clips = []
    for i,path in enumerate(dataset):
        subject = re.search(r'_S(\d\d)_',path.name)
        clips.append({'id':str(i),'name':path.name,'height':HEIGHTS.get(subject.group(1)) if subject else None})
    return jsonify(clips)

@app.post('/api/analyze-dataset')
def analyze_dataset():
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify(error='ข้อมูลคำขอต้องเป็น JSON object'),400
    try:
        index = int(body.get('id', -1))
        height = float(body.get('height',0))
        if not 0 <= index < len(dataset) or not 50 <= height <= 250: raise ValueError()
    except (TypeError,ValueError):
        return jsonify(error='เลือกคลิปทดสอบและส่วนสูงที่ถูกต้อง'),400
    if not model.POSE_MODEL_PATH.exists():
        return jsonify(error='ไม่พบโมเดล MediaPipe'),503
    job_id = uuid.uuid4().hex
    folder = DATA / job_id
    folder.mkdir()
    source = dataset[index]
    path = folder / 'input.mp4'
    shutil.copyfile(source,path)
    jobs[job_id] = {'status':'queued','filename':source.name, 'height':height}
    save_job(job_id)
    threading.Thread(target=process,args=(job_id,path,height),daemon=True).start()
    return jsonify(id=job_id),202

@app.get('/api/jobs/<job_id>')
def status(job_id):
    return jsonify(normalize_job(jobs[job_id])) if job_id in jobs else (jsonify(error='ไม่พบงาน'),404)

@app.get('/files/<job_id>/<name>')
def files(job_id,name):
    if job_id not in jobs or name not in {'result.mp4','results.csv'}: return '',404
    if jobs[job_id]['status'] != 'done':
        return jsonify(error='ผลลัพธ์ยังไม่พร้อมดาวน์โหลด'),409
    return send_from_directory(DATA / job_id, name, as_attachment=name.endswith('.csv'))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, threaded=True)
