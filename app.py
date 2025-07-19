import streamlit as st
import joblib
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from transformers import AutoTokenizer, AutoModel

st.set_page_config(page_title="Analisis Sentimen PPN 12%", layout="centered")

# Load IndoBERT tokenizer dan model
@st.cache_resource
def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
    return tokenizer, model

# Load model klasifikasi SVM
@st.cache_resource
def load_svm_model():
    return joblib.load("svm_model.pkl")

# Fungsi encoding teks menjadi vektor dengan mean pooling
def bert_encode(text, tokenizer, model, max_len=128):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_len,
            add_special_tokens=True
        )
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings.reshape(1, -1)  # Reshape agar sesuai input model

# Load dataset
@st.cache_data
def load_dataset():
    df = pd.read_csv("filtered_tweets.csv", sep=';')
    df = df.drop_duplicates()  # Menghapus duplikat
    return df

# Load metrik evaluasi model
@st.cache_data
def load_model_metrics():
    with open("svm_metrics.json", "r") as f:
        return json.load(f)

# Load semua model
tokenizer, bert_model = load_bert_model()
svm_model = load_svm_model()
df = load_dataset()
metrics = load_model_metrics()

# Label mapping
label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# UI Streamlit
st.title("💬 Analisis Sentimen Kenaikan PPN 12%")
st.write("Masukkan opini Anda terkait kenaikan PPN menjadi 12% di Indonesia.")

user_input = st.text_area("📝 Opini Anda:", height=150)

if st.button("🔍 Analisis Sentimen"):
    if user_input.strip() == "":
        st.warning("⚠️ Silakan masukkan opini terlebih dahulu.")
    else:
        vector = bert_encode(user_input, tokenizer, bert_model)
        prediction = svm_model.predict(vector)[0]
        prediction_label = label_map.get(prediction, "unknown")

        st.subheader("📊 Hasil Prediksi:")
        if prediction_label == "positive":
            st.success("✅ Sentimen Positif 😊")
        elif prediction_label == "negative":
            st.error("❌ Sentimen Negatif 😠")
        elif prediction_label == "neutral":
            st.info("➖ Sentimen Netral 😐")
        else:
            st.warning("⚠️ Hasil tidak dikenali.")

# Gambaran dataset
st.markdown("---")
st.subheader("📂 Gambaran Dataset Pelatihan")

st.write("Berikut adalah cuplikan data yang digunakan untuk melatih model:")
st.dataframe(df.head(100))

st.write(f"**Distribusi label sentimen:**")
label_counts = df['sentiment'].value_counts().sort_index()
label_names = [label_map.get(i, str(i)) for i in label_counts.index]

fig, ax = plt.subplots()
ax.pie(label_counts, labels=label_names, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Buat pie chart jadi lingkaran

st.pyplot(fig)

# Evaluasi model
st.markdown("---")
st.subheader("🧠 Evaluasi Model SVM")

st.write(f"**Akurasi:** {metrics['accuracy']:.2f}")

labels = list(metrics['precision'].keys())
precision = [metrics['precision'][label] for label in labels]
recall = [metrics['recall'][label] for label in labels]
f1 = [metrics['f1_score'][label] for label in labels]

metrics_df = pd.DataFrame({
    "Label": labels,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
})

st.dataframe(metrics_df.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}"}))

st.write(f"**Confusion Matrix:**")
cm = np.array(metrics["confusion_matrix"])

fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")

st.pyplot(fig_cm)