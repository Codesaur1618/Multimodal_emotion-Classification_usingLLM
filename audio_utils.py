import os
import json
import wave
import librosa
import soundfile as sf
import torch
import numpy as np
from vosk import Model, KaldiRecognizer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

def convert_audio(input_path, output_path="converted_audio.wav"):
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    sf.write(output_path, y, 16000)
    return output_path

def transcribe_audio(audio_path, vosk_model_path="vosk-model-en-us-0.22"):
    wf = wave.open(audio_path, "rb")
    model = Model(vosk_model_path)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    results = []

    while True:
        data = wf.readframes(4000)
        if not data:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            if res.get("text"):
                results.append(res["text"])
    final_res = json.loads(rec.FinalResult())
    if final_res.get("text"):
        results.append(final_res["text"])
    wf.close()
    return " ".join(results)

def detect_audio_emotion(audio_path):
    audio, rate = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
    id2label = emotion_model.config.id2label
    results = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))