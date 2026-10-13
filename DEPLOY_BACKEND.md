# Deploy backend สำหรับหน้า Netlify

โค้ด backend มีอยู่แล้วและจัดเตรียมให้รันด้วย Waitress, Docker, health check และคิวสูงสุด 3 งาน (วิเคราะห์ทีละงาน) ไม่ใช้ Flask development server สำหรับ Deploy

## Render

### Python native (ไม่ใช้ Docker)

1. New > Web Service เชื่อม repository NongPP235/AI_workshopLifting และ branch main
2. เปลี่ยน Language/Runtime เป็น **Python 3** แม้ repository จะมี Dockerfile อย่าเลือก Docker
3. Root Directory เว้นว่าง (ใช้ root ของ repository ไม่ใช่ static)
4. Build Command: `pip install -r requirements-server.txt && python prepare_backend.py`
5. Start Command: `python serve.py`
6. Health Check Path: `/health`
7. ตั้ง ALLOWED_ORIGINS และ API_ACCESS_KEY ตามรายการด้านล่าง แล้ว Deploy

`.python-version` กำหนด Python 3.12 ให้ตรงกับที่ทดสอบ หากมี PYTHON_VERSION ใน environment ให้ลบค่าที่ขัดกัน หรือกำหนดเป็น Python 3.12.x รุ่นเต็มที่ Render รองรับ เพราะ environment มีลำดับสูงกว่าไฟล์นี้

prepare_backend.py ดาวน์โหลดโมเดลและลองโหลด MediaPipe/OpenCV/FFmpeg ตอน Build เพื่อให้พบปัญหา native libraries ก่อนเริ่มให้บริการ เครื่องนี้ทดสอบบน Windows ยังไม่ได้รันบน Render Linux จริง หาก Build ล้มเหลว ให้ตรวจ log ของขั้นนี้และแก้ข้อผิดพลาดก่อน Start

หลัง Deploy ส่ง HTTPS URL ของ backend เพื่อใส่ใน static/config.js แล้ว Deploy Netlify ใหม่ อย่านำ URL ของหน้า Netlify มาใช้เป็น backend URL

### Docker (ทางเลือก)

1. ใน Render เลือก New > Web Service แล้วเชื่อม GitHub repository NongPP235/AI_workshopLifting
2. เลือก branch main และ Language/Runtime เป็น Docker ใช้ Dockerfile ที่ root ไม่ต้องใส่ build/start command เอง
3. ตั้ง environment variables:
   - ALLOWED_ORIGINS = https://lucky-scone-171aa5.netlify.app
   - API_ACCESS_KEY = รหัสส่วนตัวอย่างน้อย 24 ตัวอักษร สร้างและเก็บใน Render เท่านั้น ไม่ใส่ GitHub/config.js
   - MAX_PENDING_JOBS = 3 (ค่าเริ่มต้น)
   - DATA_DIR = จุด mount ของ persistent disk ถ้ามี เช่น /var/data/lifting
4. ตั้ง Health Check Path เป็น /health แล้ว Deploy
5. เมื่อสำเร็จ คัดลอก HTTPS URL เช่น https://ชื่อบริการ.onrender.com
6. แก้ static/config.js ให้ LIFTING_API_BASE_URL เป็น URL จริงนั้น แล้ว push เพื่อ Deploy หน้า Netlify ใหม่
7. กรอกรหัส API_ACCESS_KEY ในช่องรหัสเข้าใช้ backend บนหน้าเว็บก่อนอัปโหลด รหัสเก็บเฉพาะ sessionStorage ของแท็บ และส่งผ่าน Authorization header

ยังต้องใช้บัญชี Render ของคุณเพื่อสร้างบริการ ไม่มีการซื้อแพ็กเกจหรือ Deploy ออนไลน์จากสคริปต์นี้ เลือกทรัพยากรและ disk ให้เพียงพอกับวิดีโอที่จะใช้ เพราะการวิเคราะห์ใช้ CPU/RAM และเก็บไฟล์ไว้บน disk

ไฟล์ใน data/ จะหายเมื่อ filesystem ชั่วคราวของผู้ให้บริการถูกสร้างใหม่ ถ้าไม่ได้ใช้ persistent disk หากบริการหยุดหรือ Deploy ใหม่ระหว่างทำงาน ต้องส่งคลิปใหม่ งานที่เสร็จแล้วเรียกคืนได้เมื่อ data/ ยังอยู่

## รันเอง

ติดตั้ง `pip install -r requirements-server.txt`, รัน `python download_model.py`, ตั้ง API_ACCESS_KEY และ ALLOWED_ORIGINS ใน environment แล้วรัน `python serve.py` ค่าเริ่มต้น PORT=8000 และ HOST=0.0.0.0

Docker: `docker build -t lifting-backend .` แล้วส่ง environment variables ข้างต้นเมื่อรัน container และ mount พื้นที่เก็บข้อมูล Dockerfile ดาวน์โหลดโมเดลตอน build จึงไม่ต้องนำไฟล์โมเดลเข้า Git

## การเข้าถึงและข้อจำกัด

API มีรหัสเข้าใช้ร่วมกันสำหรับทีมที่เชื่อถือได้ ไม่ใช่ระบบบัญชีผู้ใช้หรือการแยกงานรายบุคคล ลิงก์วิดีโอ/CSV ที่ได้หลังยืนยันรหัสใช้ได้ 1 ชั่วโมง ผู้ที่มีลิงก์เปิดได้จนหมดอายุ กลับเข้าเว็บและดึงสถานะใหม่เพื่อรับลิงก์ใหม่

ใช้เพียง 1 process/instance เพราะคิวและสถานะปัจจุบันอยู่ใน memory ถ้าต้องรองรับหลายเครื่องต้องเปลี่ยนเป็น shared queue/database ก่อน

Docker ไม่ส่ง Dataset ทดสอบหรือไฟล์วิดีโอขึ้นไปด้วย หน้าออนไลน์ใช้การอัปโหลดคลิปเอง

ตรวจ deployment /health, อัปโหลดคลิป, ติดตามงาน, เล่นวิดีโอ, ดาวน์โหลด CSV, รหัสผิด, คิวเต็ม และ CORS บน URL จริงก่อนเปิดใช้

ตรวจ Waitress บน Windows ได้ แต่เครื่องพัฒนานี้ไม่มี Docker จึงยังไม่ได้ยืนยัน Linux image build หรือ Deploy บน Render จริง
