# เชื่อมหน้า Netlify กับ backend Python

หน้า Netlify เผยแพร่เฉพาะ static/ ไม่ได้รัน app.py หาก /api/dataset ตอบ 404 แสดงว่ายังไม่มี API ที่โดเมนนั้น

1. นำ backend Python ขึ้นบริการที่รัน Python หรือ Docker ได้ โดยติดตั้ง requirements.txt และโมเดล MediaPipe เดียวกับ setup.ps1 ใช้ production server แทน app.run และเตรียมสิทธิ์เข้าถึง/คิวตาม PRE_DEPLOY_TEST_REPORT.md ก่อนเปิดสาธารณะ
2. ตั้ง environment variable ของ backend เป็น `ALLOWED_ORIGINS=https://lucky-scone-171aa5.netlify.app` ไฟล์ .env.example เป็นตัวอย่างเท่านั้น app.py ไม่โหลดไฟล์ .env อัตโนมัติ
3. เมื่อมี HTTPS URL จริงของ backend ให้แก้ static/config.js จากค่าว่างเป็น URL นั้น เช่น `window.LIFTING_API_BASE_URL = 'https://YOUR-BACKEND-DOMAIN';` ตัวอย่างนี้ต้องแทนด้วย URL ที่ Deploy สำเร็จแล้ว และห้ามใส่รหัสผ่านหรือ token ใน frontend
4. push โค้ดแล้ว Deploy Netlify ใหม่ ทดสอบเลือกคลิป อัปโหลด ติดตามงาน เล่นวิดีโอ และดาวน์โหลดบน URL จริง

ค่าว่างใน config.js ใช้โดเมนเดียวกับหน้าเว็บ เหมาะสำหรับ localhost ที่ Flask ให้บริการทั้งหน้าเว็บและ API หาก Deploy Netlify อย่างเดียวจะยังวิเคราะห์ไม่ได้

การอัปโหลดจะส่งตรงไป backend จึงไม่ผ่าน Netlify proxy ฝั่ง backend ต้องรองรับขนาดไฟล์และเวลาในการอัปโหลด ส่วนการประมวลผลยังใช้ job/polling เหมือนเดิม

CORS เป็นการกำหนดโดเมนที่ browser เรียกได้ ไม่ใช่ระบบ authentication หรือการแยกสิทธิ์เจ้าของงาน
