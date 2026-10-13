"""Run every supplied video through the actual backend; save a resumable report."""
import app
import json, shutil, time, re, argparse
from pathlib import Path

report_path = app.ROOT / 'test-report-calibrated.json'
report = json.loads(report_path.read_text('utf-8')) if report_path.exists() else []
done = {r['file'] for r in report}
parser = argparse.ArgumentParser()
parser.add_argument('--shard',type=int,default=0)
parser.add_argument('--shards',type=int,default=1)
args = parser.parse_args()
if args.shards > 1:
    report_path = app.ROOT / f'test-report-shard-{args.shard}.json'
    report = json.loads(report_path.read_text('utf-8')) if report_path.exists() else []
    done.update(r['file'] for r in report)
for video_index,video in enumerate(sorted((app.ROOT / 'dataset_for_test').rglob('*.mp4'))):
    if video_index % args.shards != args.shard: continue
    relative = str(video.relative_to(app.ROOT))
    if relative in done: continue
    job_id = 'test-' + video.stem
    folder = app.DATA / job_id
    folder.mkdir(exist_ok=True)
    start = time.monotonic()
    app.jobs[job_id] = {'status':'queued'}
    match = re.search(r'_S(\d\d)_', video.name)
    height = {'01':165, '02':183, '03':180, '04':178, '05':173}.get(match.group(1) if match else '',175)
    app.process(job_id, video, float(height), output_dir=folder)
    job = app.jobs[job_id]
    entry = {'file':relative, 'height_cm':height, 'height_source':'user' if match else 'default; unknown subject', 'seconds':round(time.monotonic()-start,2), **job}
    if job['status'] == 'done':
        cap, info = app.model.open_video(folder / 'result.mp4')
        ok, frame = cap.read()
        cap.release()
        entry['output_decodable'] = bool(ok)
        entry['output_frames'] = info['frames']
        entry['csv_exists'] = (folder / 'results.csv').exists()
        client = app.app.test_client()
        entry['video_http'] = client.get(job['result']['video'], headers={'Range':'bytes=0-1023'}).status_code
        entry['csv_http'] = client.get(job['result']['csv']).status_code
    report.append(entry)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TEST', video.name, job['status'], flush=True)
