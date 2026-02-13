## 📌 [งานกลุ่ม] การออกแบบระบบ AutoML ด้วย Metaheuristic Optimization

ให้ น.ศ. จับกลุ่ม ๆ ละไม่เกิน 5 คน

## 🎯 วัตถุประสงค์ของงาน

ให้นักศึกษาศึกษาและประยุกต์ใช้ Metaheuristic Algorithm เพื่อออกแบบระบบ Automated Machine Learning (AutoML) สำหรับแก้ปัญหา Machine Learning จากข้อมูลจริง โดยระบบที่พัฒนาต้องสามารถ

- ค้นหาและเลือก โมเดล (Model), ค่า Hyperparameters และ กระบวนการเตรียมข้อมูล (Feature Preprocessing) ที่เหมาะสมโดยอัตโนมัติ
- ออกแบบระบบ AutoML ที่สามารถรองรับทั้งปัญหา Regression และ Classification

## 🔍 สิ่งที่ต้อง Optimize (AutoML Search Space)

แต่ละกลุ่มต้องออกแบบ Search Space อย่างชัดเจน โดยครอบคลุมองค์ประกอบดังต่อไปนี้

1. Preprocessing:
   - Feature Scaling: None / Standard Scaling / Min-Max Scaling / อื่น ๆ
   - Feature Selection / Extraction: None / SelectKBest / PCA / อื่น ๆ
2. Model Selection:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest
   - XGBoost
   - Neural Network
   - อื่น ๆ
3. Hyperparameters: กำหนดและอธิบาย Hyperparameters ที่ใช้ในการปรับจูนของแต่ละโมเดลอย่างเหมาะสม
4. Objective Function: ต้องกำหนดและอธิบาย Fitness Function (Objective Function) อย่างชัดเจน เพื่อใช้ในการประเมินประสิทธิภาพของโมเดลที่ได้จากกระบวนการค้นหา
5. Constraint (ถ้ามี): อธิบายเงื่อนไขหรือข้อจำกัดที่ใช้ในกระบวนการ Optimization (ถ้ามี)

## 📦 สิ่งที่ต้องส่ง

1. Source Code: GitHub หรือ Google Colab
2. รายงาน (ประกอบด้วยหัวข้ออย่างน้อยดังต่อไปนี้)
   - Problem Definition
   - Metaheuristic Design
   - AutoML Architecture
   - Experimental Results
   - Discussion and Limitations

## 📝 เกณฑ์การให้คะแนน

การวิเคราะห์และการออกแบบระบบ: 30 คะแนน
ความสามารถในการบูรณาการองค์ความรู้จากรายวิชา CP413202: 30 คะแนน
ความคิดสร้างสรรค์ในการออกแบบวิธีการแก้ปัญหา: 20 คะแนน
ผลลัพธ์และการวิเคราะห์เชิงทดลอง: 20 คะแนน
ส่งก่อน วันอาทิตย์ที่ 22 มี.ค. 2569
