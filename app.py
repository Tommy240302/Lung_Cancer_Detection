import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pickle
import numpy as np
import pandas as pd
from numpy.array_api import result_type
from tensorflow.keras.models import load_model

with open(r"F:\TTCS\Lung-Cancer-Detection\lung-cancer-detection-ml\model.pkl", "rb") as file:
    model = pickle.load(file)
print("Model loaded successfully!")
print("Model type:", type(model))
print("Model details:", model)

# Load model CNN
model_cnn = load_model(r"F:\TTCS\Lung-Cancer-Detection\lung-cancer-detection-cnn\model.h5")
print("Model CNN loaded successfully!")

# Tạo cửa sổ chính
window = tk.Tk()
window.title("Vehicle Detection System")
window.geometry("900x600")
window.configure(bg="#F5F5F5")  # Background

# Fonts
title_font = ("Helvetica", 16, "bold")
label_font = ("Helvetica", 12)
button_font = ("Helvetica", 12, "bold")

# Chức năng chọn ảnh
def open_image_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        try:
            # Hiển thị ảnh lên giao diện
            image = Image.open(file_path)
            image = image.resize((300, 300), Image.LANCZOS)
            img = ImageTk.PhotoImage(image)
            la3.config(image=img)
            la3.image = img  # Giữ tham chiếu để không bị xóa

            # Dự đoán ảnh
            predicted_class = predict_image('F:\TTCS\Lung-Cancer-Detection\lung-cancer-detection-cnn\model.h5', file_path)  # 'model.h5' là tên model của bạn
            if predicted_class is not None:
                messagebox.showinfo("Kết quả phân loại", f"Ảnh được phân loại vào lớp: {predicted_class}")
            else:
                messagebox.showerror("Lỗi", "Không thể phân loại ảnh.")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi khi mở hoặc phân tích ảnh:\n{str(e)}")


def predict_image(model_path, image_path):
    try:
        # Load model CNN bằng hàm load_model của Keras
        model = load_model(model_path)
        print("✅ Model CNN đã được load thành công!")

        # Kiểm tra input_shape của model
        print("🔹 Model input shape:", model.input_shape)

        # Load và xử lý ảnh đầu vào
        img = Image.open(image_path).convert('RGB')  # Đảm bảo ảnh RGB
        img = img.resize((224, 224))  # Resize theo kích thước model yêu cầu
        img_array = np.array(img) / 255.0  # Chuẩn hóa ảnh về [0, 1]

        # Kiểm tra shape sau xử lý
        print("🔹 Shape ảnh sau xử lý:", img_array.shape)

        # Thêm chiều batch_size nếu cần thiết
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)

        print("🔹 Shape ảnh đưa vào model:", img_array.shape)

        # Dự đoán và lấy kết quả
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        print("🔹 Kết quả dự đoán:", prediction)
        print("🔹 Lớp dự đoán:", predicted_class)

        return predicted_class

    except Exception as e:
        print(f"❌ Lỗi khi dự đoán ảnh: {e}")
        return None


def reset_image():
    la3.config(image='')

def reset_symptoms():
    for var in check_vars:
        var.set(0)


def predict_lung_cancer():
    try:
        print("🔍 Kiểm tra model trước khi dự đoán...")
        print("Model type:", type(model))
        print("Model details:", model)

        # Lấy dữ liệu đầu vào
        age = int(spin_age.get())
        gender = 1 if gender_var.get() == "Nam" else 0
        symptoms = [2 if var.get() == 1 else 1 for var in check_vars]  # 0 nếu không tích, 1 nếu tích

        # Sử dụng đúng tên cột từ model
        feature_names = list(model.feature_names_in_)

        # Tạo DataFrame với đúng tên cột
        input_data = pd.DataFrame([symptoms], columns=[
            "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
            "CHRONIC DISEASE", "FATIGUE ", "ALLERGY ", "WHEEZING",
            "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH",
            "SWALLOWING DIFFICULTY", "CHEST PAIN"
        ])
        input_data.insert(0, "GENDER", gender)
        input_data.insert(1, "AGE", age)

        # Đồng bộ tên cột với model
        input_data.columns = feature_names

        # **Debug: In dữ liệu đầu vào**
        print("\n🔹 Dữ liệu đầu vào cho model:")
        print(input_data)

        # **Dự đoán**
        try:
            prediction = model.predict(input_data)[0]
            print("✅ Dự đoán thành công! Kết quả:", prediction)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)
                print("📊 Xác suất dự đoán:", probabilities)

            result_text = "Có nguy cơ mắc ung thư phổi!" if prediction == 1 else "Không có nguy cơ mắc ung thư phổi."
        except Exception as pred_error:
            print("❌ Lỗi khi dự đoán:", str(pred_error))
            result_text = f"Lỗi khi dự đoán: {str(pred_error)}"

    except ValueError as ve:
        print("❌ Lỗi nhập liệu:", str(ve))
        result_text = "Lỗi: Tuổi phải là số nguyên hợp lệ!"
    except Exception as e:
        print("❌ Lỗi toàn cục:", str(e))
        result_text = f"Lỗi khi xử lý dữ liệu: {str(e)}"

    # Hiển thị kết quả
    print("📢 Kết quả hiển thị:", result_text)
    result_entry.delete(0, tk.END)
    result_entry.insert(0, result_text)


# Khai báo toàn cục trước khi sử dụng
check_vars = []


def nhap_trieu_chung():
    global check_vars  # Đảm bảo có thể truy cập biến toàn cục
    check_vars = []  # Reset danh sách để tránh lưu trạng thái cũ
    check_vars.clear()

    new_window = tk.Toplevel()
    new_window.title("Nhập thông tin")
    new_window.geometry("900x600")
    new_window.configure(bg="#E8F0F2")  # Nền nhẹ nhàng

    # Tiêu đề
    label_title = tk.Label(new_window, text="NHẬP THÔNG TIN", font=("Helvetica", 18, "bold"), bg="#4A90E2", fg="white")
    label_title.pack(fill="x", pady=10)

    # ==== Khung thông tin cá nhân ====
    frame_info = tk.Frame(new_window, bg="#E8F0F2")
    frame_info.pack(pady=10, padx=20, fill="x")

    tk.Label(frame_info, text="Tuổi:", font=("Helvetica", 12), bg="#E8F0F2").grid(row=0, column=0, padx=5, pady=5,
                                                                                  sticky="w")
    global spin_age
    spin_age = tk.Spinbox(frame_info, from_=0, to=100, width=5, font=("Helvetica", 12))
    spin_age.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(frame_info, text="Giới tính:", font=("Helvetica", 12), bg="#E8F0F2").grid(row=0, column=2, padx=5, pady=5,
                                                                                       sticky="w")
    global gender_var
    gender_var = tk.StringVar(value="Nam")
    gender_dropdown = ttk.Combobox(frame_info, textvariable=gender_var, values=["Nam", "Nữ"], state="readonly", width=7)
    gender_dropdown.grid(row=0, column=3, padx=10, pady=5)

    # ==== Khung triệu chứng ====
    frame_table = tk.Frame(new_window, bg="#E8F0F2")
    frame_table.pack(pady=10, padx=20, fill="x")

    tk.Label(frame_table, text="Triệu chứng", font=("Helvetica", 12, "bold"), bg="#4A90E2", fg="white").grid(
        row=0, column=0, columnspan=2, pady=5, sticky="ew"
    )

    symptoms = [
        "Hút thuốc", "Vàng ngón tay", "Lo lắng", "Áp lực từ bạn bè",
        "Bệnh mãn tính", "Mệt mỏi", "Dị ứng", "Thở khò khè",
        "Uống rượu", "Ho", "Khó thở", "Khó nuốt", "Đau ngực"
    ]

    for i, symptom in enumerate(symptoms):
        tk.Label(frame_table, text=symptom, font=("Helvetica", 11), bg="#E8F0F2").grid(
            row=i + 1, column=0, sticky="w", padx=10, pady=3
        )
        var = tk.IntVar()
        check_vars.append(var)
        tk.Checkbutton(frame_table, variable=var, bg="#E8F0F2").grid(row=i + 1, column=1, padx=10)

    # ==== Khung dự đoán và kết quả ====
    frame_buttons = tk.Frame(new_window, bg="#E8F0F2")
    frame_buttons.pack(pady=10, padx=20, fill="x")

    btn_predict = tk.Button(frame_buttons, text="Dự đoán", font=("Helvetica", 12, "bold"), bg="#4A90E2", fg="white",
                            padx=10, command=predict_lung_cancer)
    btn_predict.pack(side="left", padx=20)
    btn_reset = tk.Button(frame_buttons, text="Reset", font=("Helvetica", 12, "bold"), bg="#4A90E2", fg="white",
                            padx=10, command=reset_symptoms)
    btn_reset.pack(side="left", padx=10)

    global result_entry
    result_entry = tk.Entry(frame_buttons, width=40, font=("Helvetica", 12))
    result_entry.pack(side="right", padx=20, pady=5)


# Tạo khung menu bên trái
frame1 = tk.Frame(window, width=250, height=600, bg="#C4E1F6", padx=10, pady=10)
frame1.grid(row=0, column=0, rowspan=2, sticky="nsw")
frame1.grid_propagate(False)

la1 = tk.Label(frame1, text='Mục Tùy Chọn', font=title_font, bg="#C4E1F6", fg="#003161")
btnNhapAnh = tk.Button(frame1, width=20, height=2, text='Ảnh chụp X-Quang',
                    font=button_font, bg="#4A628A", fg="#ffffff", command=open_image_file)
btnPhanTichAnh = tk.Button(frame1, width=20, height=2, text='Phân tích ảnh',
                    font=button_font, bg="#4A628A", fg="#ffffff", command=open_image_file)
btnLamMoi = tk.Button(frame1, width=20, height=2, text='Làm mới',
                    font=button_font, bg="#4A628A", fg="#ffffff", command=reset_image)
btnNhapTrieuChung = tk.Button(frame1, width=20, height=2, text='Nhập triệu chứng',
                    font=button_font, bg="#4A628A", fg="#ffffff", command=nhap_trieu_chung)

btnThoat = tk.Button(frame1, width=20, height=2, text='Thoát', font=button_font, bg="#D9534F", fg="#ffffff", command=window.quit)

la1.pack(pady=10)
btnNhapAnh.pack(pady=10)
btnPhanTichAnh.pack(pady=10)
btnLamMoi.pack(pady=10)
btnNhapTrieuChung.pack(pady=10)
btnThoat.pack(pady=10)

# Khung hiển thị ảnh
frame2 = tk.Frame(window, width=600, height=400, bg="#4A628A", padx=10, pady=10)
frame2.grid(row=0, column=1, sticky="nsew")
frame2.grid_propagate(False)

la2 = tk.Label(frame2, text='Ảnh đầu vào', font=title_font, bg="#4A628A", fg="#ffffff")
la3 = tk.Label(frame2, width=300, height=300, bg="#ffffff")
la2.pack()
la3.pack()

# Khung kết quả
frame3 = tk.Frame(window, width=600, height=200, bg="#4A628A", padx=10, pady=10)
frame3.grid(row=1, column=1, sticky="nsew")
frame3.grid_propagate(False)

la4 = tk.Label(frame3, text='Kết Quả Phát Hiện', font=title_font, bg="#4A628A", fg="#ffffff")
kq_text = tk.Text(frame3, padx=5, pady=5, width=40, height=8, font=label_font)

la4.pack(pady=5)
kq_text.pack()

window.rowconfigure(0, weight=1)
window.rowconfigure(1, weight=1)
window.columnconfigure(1, weight=1)
window.mainloop()