# ==================== MEDICINAL LEAF IDENTIFIER GUI ====================

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

from descriptions import DESCRIPTIONS

tf.get_logger().setLevel('ERROR')

def preprocess_image(img_array):
    """
    Normalize image pixels to range [-1, 1]
    """
    img_array = img_array.astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    return img_array


MODEL_PATH = "leaf_classifier_fast.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    messagebox.showerror("Model Error", str(e))
    raise SystemExit

class_names = list(DESCRIPTIONS.keys())

root = tk.Tk()
root.title("Medicinal Leaf Identifier")
root.geometry("900x780")
root.resizable(False, False)


tk.Label(
    root,
    text="Medicinal Leaf Identifier",
    font=("Helvetica", 22, "bold"),
    fg="green"
).pack(pady=10)


container = tk.Frame(
    root,
    width=600,
    height=320,
    bg="#f5f5f5",
    bd=3,
    relief="ridge"
)
container.pack(pady=8)
container.pack_propagate(False)

img_label = tk.Label(
    container,
    text="Upload a leaf image",
    font=("Arial", 16),
    fg="gray",
    bg="#f5f5f5"
)
img_label.place(relx=0.5, rely=0.5, anchor="center")


result_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
result_label.pack(pady=6)

desc_label = tk.Label(
    root,
    text="",
    font=("Arial", 13),
    wraplength=860,
    justify="left"
)
desc_label.pack(pady=6)

def upload_and_predict():
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )
    if not path:
        return

    try:
        img = Image.open(path).convert("RGB")

       
        disp = img.resize((540, 300), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(disp)
        img_label.config(image=tk_img, text="")
        img_label.image = tk_img

       
        inp = img.resize((160, 160))
        arr = np.array(inp)
        arr = preprocess_image(arr)
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr, verbose=0)[0]
        confidence = float(np.max(preds) * 100)
        idx = int(np.argmax(preds))
        leaf_name = class_names[idx]

        if confidence < 60:
            result_label.config(text=" Not a Leaf Image", fg="red")
            desc_label.config(text="Description not available.")
        elif confidence < 80:
            result_label.config(text="Non-Medicinal Leaf", fg="orange")
            desc_label.config(text="This leaf is not part of the medicinal dataset.")
        else:
            result_label.config(
                text=f" {leaf_name} ({confidence:.2f}%)",
                fg="green"
            )
            desc_label.config(
                text=DESCRIPTIONS.get(leaf_name)
            )

    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Button(
    root,
    text="Upload Leaf Image",
    command=upload_and_predict,
    font=("Arial", 14, "bold"),
    bg="#4CAF50",
    fg="white",
    padx=22,
    pady=8
).pack(pady=10)

tk.Button(
    root,
    text="Exit",
    command=root.destroy,
    font=("Arial", 13, "bold"),
    bg="#f44336",
    fg="white",
    padx=22,
    pady=6
).pack(pady=6)

root.mainloop()
