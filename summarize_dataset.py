import json
import csv
from pathlib import Path

root = Path(__file__).parent
entries = {}
for path in [root/'test-report-calibrated.json', *sorted(root.glob('test-report-shard-*.json'))]:
    if not path.exists(): continue
    for row in json.loads(path.read_text('utf-8')):
        result = row.get('result',{})
        if 'realtime_lifts' in result:
            result.setdefault('robust_lifts',result['lifts'])
            result['lifts'] = result['realtime_lifts']
        entries[row['file']] = row
rows = sorted(entries.values(),key=lambda r:r['file'])
for row in rows:
    if row['status'] != 'done': continue
    result = row['result']
    job_id = result['csv'].split('/')[2]
    with (root/'data'/job_id/'results.csv').open(encoding='utf-8-sig',newline='') as f:
        frames = list(csv.DictReader(f))
    assert len(frames) == result['frames'], row['file']
    assert int(frames[-1]['lift_count_realtime']) == result['lifts'], row['file']
(root/'test-report-calibrated.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
columns = ['clip','height_cm','status','frames','coverage_percent','realtime_lifts','robust_lifts','limit_kg','video_http','csv_http']
flat = []
for row in rows:
    r = row.get('result',{})
    flat.append(dict(zip(columns,[Path(row['file']).name,row['height_cm'],row['status'],r.get('frames'),r.get('coverage'),r.get('lifts'),r.get('robust_lifts'),r.get('limit'),row.get('video_http'),row.get('csv_http')])))
with (root/'dataset-results.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=columns)
    writer.writeheader()
    writer.writerows(flat)
ok = [r for r in rows if r['status']=='done' and r.get('output_decodable') and r.get('csv_exists') and r.get('video_http')==206 and r.get('csv_http')==200 and r.get('output_frames')==r.get('result',{}).get('frames')]
lines = ['# รายงานทดสอบเว็บวิเคราะห์การยกของ เวอร์ชัน 1.1','',f'ทดสอบเสร็จ {len(rows)}/26 คลิป; ผ่านการประมวลผล วิดีโออ่านกลับได้ จำนวนเฟรมตรง CSV และ HTTP ดาวน์โหลด/เล่นวิดีโอ {len(ok)}/{len(rows)} คลิป','', 'ใช้ส่วนสูงจริงจากผู้ใช้: S01 165, S02 183, S03 180, S04 178, S05 173 ซม. คลิป UUID ไม่มีรหัสผู้ทดสอบจึงใช้ 175 ซม. เพื่อทดสอบการทำงานเท่านั้น','', 'สูตรพิกัด การปรับเทียบ และตารางน้ำหนักใช้ notebook เดิม การ์ดหลักใช้จำนวน realtime เดียวกับวิดีโอ/CSV; แสดง Robust เป็นข้อมูลเปรียบเทียบ','', 'ไม่มี ground truth ของ H/V/มุมหรือจำนวนยก จึงยังไม่ใช่การยืนยันความแม่นยำเชิงเรขาคณิต จากคลิป NEG-BEND พบว่าวิธี realtime อาจนับการก้มที่ไม่ได้ยกวัตถุ ระบบตรวจท่าทางและการเคลื่อนไหวมือ ไม่ได้ตรวจวัตถุหรือการจับวัตถุ','', '| คลิป | สูง ซม. | สถานะ | เฟรม | วัดมือได้ % | realtime | Robust | ต่ำสุด กก. |','|---|---:|---|---:|---:|---:|---:|---:|']
for r in flat:
    lines.append('| '+' | '.join(str(r[c]) for c in columns[:8])+' |')
lines += ['', 'ตรวจหน้าเว็บจริงแล้ว: อัปโหลดคลิป เลือก Dataset เติมส่วนสูง S02=183 ซม. ประมวลผล เล่นวิดีโอ H.264 ได้ รีเฟรชและเริ่มเซิร์ฟเวอร์ใหม่แล้วยังเรียกผลเดิมได้ ตรวจรูปแบบจอคอมพิวเตอร์/มือถือ และตรวจกรณีข้อมูลส่วนสูงผิด ไฟล์ชนิดผิด คลิปสั้น และคลิปไม่พบคน', '']
(root/'TEST_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(f'{len(rows)}/26 finished; {len(ok)} passed')
assert len(ok)==len(rows), 'Some completed clips failed artifact checks'
