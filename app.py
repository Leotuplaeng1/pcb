from flask import Flask, request, render_template, jsonify
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import os

app = Flask(__name__)

# ตั้งค่าการเชื่อมต่อกับ Roboflow ผ่าน InferenceHTTPClient
CLIENT = request(
    api_url="https://detect.roboflow.com",
    api_key="DinIFQuB3og5F3IQzf6e"
)

MODEL_ID = "circuit-board-defect-detection/1"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # รับไฟล์รูปจาก request
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # แปลงไฟล์เป็นรูปภาพและเซฟไฟล์ที่อัปโหลด
    image = Image.open(io.BytesIO(file.read()))

    # แปลงภาพเป็น RGB ถ้าเป็น RGBA
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image.save("uploaded_image.jpg")

    try:
        # ส่งไปยัง Roboflow API ผ่าน requests
        with open("uploaded_image.jpg", "rb") as img_file:
            response = requests.post(
                ROBOFLOW_API_URL,
                headers={
                    "Authorization": f"Bearer {ROBOFLOW_API_KEY}"
                },
                files={"file": img_file}
            )

        # ตรวจสอบว่าได้ผลลัพธ์หรือไม่
        if response.status_code == 200:
            result = response.json()

            if 'predictions' in result:
                details = result['predictions']  # ดึงข้อมูลเกี่ยวกับสิวที่ตรวจจับได้
                img_with_boxes, cropped_acnes = draw_boxes_and_crop("uploaded_image.jpg", details)
                img_with_boxes_path = 'static/detected_image.jpg'
                img_with_boxes.save(img_with_boxes_path)

                # บันทึกรูปที่ crop แยกออกมา
                cropped_paths = []
                for i, crop in enumerate(cropped_acnes):
                    crop_path = f'static/cropped_acne_{i}.jpg'
                    crop.save(crop_path)
                    cropped_paths.append(crop_path)

                return render_template('index.html', img_path=img_with_boxes_path, details=details, cropped_images=cropped_paths)
            else:
                return jsonify({'error': 'No predictions found in the result'}), 500
        else:
            return jsonify({'error': f"Failed to get predictions, status code: {response.status_code}"}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def draw_boxes_and_crop(image_path, predictions):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 15)  # ปรับขนาด font ให้ใหญ่ขึ้น

    cropped_acnes = []
    
    for detection in predictions:
        confidence = detection['confidence']
        if confidence >= 0.3:  # ตรวจสอบความเชื่อมั่น (confidence) ที่ 30% ขึ้นไป
            x = detection['x']
            y = detection['y']
            w = detection['width']
            h = detection['height']
            label = detection['class']

            # คำนวณพิกัดของกรอบ
            left = x - w / 2
            top = y - h / 2
            right = x + w / 2
            bottom = y + h / 2

            # วาดกรอบและแสดง label และ confidence บนภาพ
            draw.rectangle([left, top, right, bottom], outline="cyan", width=3)
            draw.text((left, top - 30), f'{label} {confidence:.2%}', font=font, fill="cyan")

            # Crop บริเวณที่ตรวจจับสิวและปรับขนาดให้ใหญ่ขึ้น
            cropped_acne = image.crop((left, top, right, bottom))
            cropped_acne = cropped_acne.resize((100, 100))  # ปรับขนาดเป็น 100x100
            cropped_acnes.append(cropped_acne)

    return image, cropped_acnes

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # ใช้ค่า PORT จาก Heroku ถ้ามี
    app.run(debug=True, host='0.0.0.0', port=port)

