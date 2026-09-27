# คู่มือการติดตั้งและใช้งาน Deep-Live-Cam 2.1 (ภาษาไทย)

คู่มือนี้สรุปขั้นตอนการติดตั้งและรันโปรแกรม **Deep-Live-Cam** ด้วยการ์ดจอ (GPU) บนระบบปฏิบัติการ Windows สำหรับผู้ใช้งานภาษาไทย

---

## 🚀 วิธีเริ่มต้นใช้งานอย่างรวดเร็ว (Quick Start)

### วิธีที่ 1: ดับเบิลคลิกไฟล์ Batch (แนะนำ)
ดับเบิลคลิกไฟล์สคริปต์ในโฟลเดอร์โครงการ:
- **`run-th.bat`** (รันโปรแกรมพร้อมเมนูภาษาไทยและการ์ดจอ DirectML)
- **`run-directml.bat`** หรือ **`run-cuda.bat`**

---

### วิธีที่ 2: รันผ่าน Command Line / PowerShell
1. เปิด PowerShell หรือ Command Prompt ในโฟลเดอร์นี้
2. เปิดใช้งาน Virtual Environment:
   ```powershell
   venv\Scripts\activate
   ```
3. พิมพ์คำสั่งรันโปรแกรมด้วย GPU (DirectML):
   ```powershell
   python run.py --execution-provider dml
   ```

---

## 📁 โครงสร้างไฟล์ Model ที่จำเป็น (`models/`)
ไฟล์โมเดลการประมวลผลถูกดาวน์โหลดไว้ในโฟลเดอร์ `models/` เรียบร้อยแล้ว:
1. `inswapper_128_fp16.onnx` (สลับใบหน้า)
2. `GFPGANv1.4.onnx` (ปรับความคมชัดใบหน้า)
3. `buffalo_l/` (ตรวจจับตำแหน่งใบหน้าจาก InsightFace)

---

## 🎬 ขั้นตอนการใช้งานหน้าต่างโปรแกรม (3 ขั้นตอน)

1. **เลือกใบหน้าต้นฉบับ (Source Face)**:
   - คลิกปุ่ม **Select a Face** เพื่อเลือกรูปภาพใบหน้าของคนที่ต้องการนำมาสลับ
2. **เลือกโหมดเป้าหมาย (Target / Live)**:
   - **กรณีใช้กล้องเว็บแคม (Live Webcam)**: ในช่อง **Select Camera** เลือกกล้องของคุณ แล้วกดปุ่ม **Live**
   - **กรณีใช้รูปภาพหรือวิดีโอ**: เลือกไฟล์ในช่อง **Select a Target** แล้วกดปุ่ม **Start**
3. **การปรับแต่งเพิ่มเติม (Options)**:
   - **Mouth Mask**: เลื่อนแถบ Mouth Mask เพื่อเปิดใช้งานโหมดรักษาความสมจริงของปากและคางเดิม
   - **Many Faces**: ติ๊กเลือกหากต้องการสลับหน้าทุกคนที่อยู่ในเฟรม

---

## 🛠️ รายละเอียดการแก้ไขระบบ (System Fixes in Repository)
- **DirectML GPU Acceleration**: สลับมาใช้ `onnxruntime-directml==1.21.0` ช่วยให้การ์ดจอ NVIDIA / AMD / Intel ประมวลผลได้เต็มประสิทธิภาพโดยไม่เกิดปัญหา `LoadLibrary Error 126` จาก CUDA DLL ที่สูญหาย
- **Model Provider Compatibility**: แก้ไขไฟล์ `modules/processors/frame/face_swapper.py` ให้รองรับโมเดล `inswapper_128_fp16.onnx` กับ DirectML และ Execution Providers ทั้งหมด
- **Cloudflare / Image Error Guard**: ปรับปรุงฟังก์ชัน `fetch_random_face` ใน `modules/ui.py` ให้ตรวจสอบไฟล์รูปภาพและแจ้งเตือนอย่างถูกต้องเมื่อเว็บไซต์ภายนอกถูกบล็อก
