import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# Load model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model_en = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)
model_en.load_state_dict(torch.load('./weigths/en_best_distilbert.pth'))
model_en.eval()

# Define emotion classes
en_emotion_classes = {
    0: "Sadness 😞",
    1: "Joy 😃",
    2: "Love ❤️",
    3: "Anger 😠",
    4: "Fear 😱",
    5: "Surprise 😮"
}

# Create a Streamlit app
st.title("Emotion analysis [DistilBERT]")

# English input
user_input = st.text_input("Enter your text here")
if st.button("Predict"):
    # Tokenize user text
    inputs = tokenizer(user_input, return_tensors='pt', truncation=True, padding=True)

    # Make a prediction
    with torch.no_grad():
        outputs = model_en(**inputs)

    # Get predicted class
    _, predicted_class = torch.max(outputs.logits, dim=1)
    predicted_emotion_en = en_emotion_classes[predicted_class.item()]
    st.write(f"The predicted emotion is : **{predicted_emotion_en}**")

# Separator
st.markdown("---")

# Create a Streamlit app
st.title("Analyse des émotions [DistilBERT]")

# Define emotion classes
fr_emotion_classes = {
    0: "Tristesse 😞",
    1: "Joie 😃",
    2: "Amour ❤️",
    3: "Colère 😠",
    4: "Peur 😱",
    5: "Surprise 😮"
}

model_fr = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)
model_fr.load_state_dict(torch.load('./weigths/fr_best_distilbert.pth'))
model_fr.eval()

# French input
user_input_fr = st.text_input("Entrez votre texte ici")
if st.button("Prédire"):
    # Tokenize user text
    inputs_fr = tokenizer(user_input_fr, return_tensors='pt', truncation=True, padding=True)

    # Make a prediction
    with torch.no_grad():
        outputs_fr = model_fr(**inputs_fr)

    # Get predicted class
    _, predicted_class_fr = torch.max(outputs_fr.logits, dim=1)
    predicted_emotion_fr = fr_emotion_classes[predicted_class_fr.item()]
    st.write(f"L'émotion prédite est : **{predicted_emotion_fr}**")