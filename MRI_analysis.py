import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "brain_tumor_cnn.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  # same order as train_dataset.class_names (alphabetical)


def main():
    st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")

    st.title("🧠 Brain Tumor MRI Classifier")
    st.write(
        "Upload a brain MRI scan and the model will classify it into one of 4 categories: "
        "**glioma, meningioma, notumor, pituitary**."
    )
    st.warning(
        "This is a student/demo project, NOT a medical diagnostic tool. "
        "Do not use this for real medical decisions.",
        icon="⚠️",
    )

    @st.cache_resource
    def load_model():
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model, None
        except Exception as e:
            return None, str(e)

    model, error = load_model()

    if error is not None:
        st.error(
            f"Could not load model from '{MODEL_PATH}'. "
            f"Make sure the file is in the same folder as this script.\n\nError: {error}"
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload an MRI image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded MRI", use_column_width=True)

        # Preprocess: resize + normalize (same as training pipeline: Rescaling(1./255))
        img_resized = image.resize(IMG_SIZE)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)

        with st.spinner("Running inference..."):
            predictions = model.predict(img_array)

        probs = predictions[0]
        predicted_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probs[predicted_idx]) * 100

        with col2:
            st.subheader("Prediction")
            st.markdown(f"### **{predicted_class.upper()}**")
            st.write(f"Confidence: **{confidence:.2f}%**")

        st.subheader("Class probabilities")
        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(prob_dict)

        with st.expander("Raw probability values"):
            for cls, p in sorted(prob_dict.items(), key=lambda x: -x[1]):
                st.write(f"{cls}: {p:.4f}")

    else:
        st.info("Upload an image to get a prediction.")


if __name__ == "__main__":
    main()