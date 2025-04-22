# 🎭 Multimodal Emotion Analysis with LLaMA Inference

This Flask-based web application performs **emotion analysis** from **text**, **audio**, and **video** inputs. It leverages multiple models to detect emotional states, identify underlying causes, and generate natural, human-like inferences using the **LLaMA language model** via Ollama.

## 🚀 Features

- **Text Emotion Detection**: Classifies emotions from user-inputted text and extracts the likely cause.  
- **Audio Emotion Analysis**: Converts audio to a standard format, detects vocal emotion, transcribes speech, and analyzes text emotion.  
- **Video Emotion Detection**: Extracts visual emotions from facial cues, analyzes speech audio, and integrates transcription-based emotion classification.  
- **LLM-Based Inference**: Summarizes emotional findings in a natural, friendly tone using LLaMA via Ollama.

## 🛠️ Tech Stack

- **Backend**: Python, Flask  
- **Audio/Video**: `moviepy`, `pydub`, custom `audio_utils.py` & `video_utils.py`  
- **Emotion Models**: Pre-trained models for audio, text, and visual emotion detection  
- **LLM Integration**: Local Ollama installation running `llama3.2`  

## 🧪 Requirements

- Python 3.7+
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```
- Local installation of **[Ollama](https://ollama.com/)** with the `llama3.2` model
- Ensure `uploads/` directory exists in your project folder

## 📦 Running the App

```
python app.py
```

Then open your browser and go to: `http://127.0.0.1:5000`

## 📁 Project Structure

```
├── app.py
├── templates/
│   └── index.html
├── static/
├── uploads/
├── audio_utils.py
├── video_utils.py
├── text_utils.py
└── README.md
```

## 📌 Notes

- Update the path to your Ollama installation (`ollama.exe`) in `generate_llama_inference()`  
- Make sure media files are properly handled and cleaned up  
- Easily extendable to support additional models or LLMs

## 🧠 Credits

Made by **Abisek**  
Inspired by research in AI-driven emotion understanding and contextual reasoning.

