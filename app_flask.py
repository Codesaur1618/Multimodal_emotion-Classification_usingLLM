from flask import Flask, render_template, request
import os
import numpy as np
import json
from text_utils import classify_text_emotion, extract_emotion_cause
from audio_utils import convert_audio, transcribe_audio, detect_audio_emotion
from video_utils import detect_visual_emotion
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# LLaMA Inference function
def generate_llama_inference(summary_dict):
    summary_text = json.dumps(summary_dict, indent=2)

    prompt = f"""
You are an intelligent assistant that interprets emotional analysis summaries.

Given the following emotion summary:
{summary_text}

Your task:
1. Provide a natural, human-friendly explanation of the person's emotional state.
2. Clearly identify and restate the exact cause of the emotion, based on the cause phrase.
3. Don't use the starting phrase "Based on the provided emotional analysis summary, here's an interpretation"

Output:
Inference:
"""

    # Update this path to your local Ollama installation
    ollama_path = r"C:\Users\abise\AppData\Local\Programs\Ollama\ollama.exe"

    try:
        result = subprocess.run(
            [ollama_path, 'run', 'llama3.2'],
            input=prompt.encode(),
            capture_output=True,
            timeout=60  # add a timeout just in case
        )
        return result.stdout.decode().strip()
    except Exception as e:
        return f"⚠️ Failed to generate inference: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    emotion_summary = {}

    if request.method == "POST":
        input_type = request.form.get("input_type")

        if input_type == "text":
            text = request.form.get("text_input")
            if text:
                text_result = classify_text_emotion([text])[0]
                cause = extract_emotion_cause(text)
                emotion_summary = {
                    "type": "Text",
                    "text_input": text,
                    "text_emotion": text_result,
                    "cause": cause
                }

        elif input_type == "audio":
            audio_file = request.files.get("audio_file")
            if audio_file:
                path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
                audio_file.save(path)

                converted = convert_audio(path)
                audio_result = detect_audio_emotion(converted)
                transcription = transcribe_audio(converted)
                cause = extract_emotion_cause(transcription)

                sorted_audio = sorted(audio_result.items(), key=lambda x: x[1], reverse=True)
                top_audio_emotion = sorted_audio[0][0]
                if len(sorted_audio) > 1 and np.isclose(sorted_audio[0][1], sorted_audio[1][1], atol=1e-4):
                    top_audio_emotion = "neutral"

                text_result = classify_text_emotion([transcription])[0]
                emotion_summary = {
                    "type": "Audio",
                    "transcription": transcription,
                    "audio_emotion": top_audio_emotion,
                    "text_emotion": text_result,
                    "cause": cause
                }

                os.remove(converted)

        elif input_type == "video":
            video_file = request.files.get("video_file")
            if video_file:
                path = os.path.join(UPLOAD_FOLDER, video_file.filename)
                video_file.save(path)

                vis_emo, vis_scores = detect_visual_emotion(path)

                from moviepy.editor import VideoFileClip
                audio_path = os.path.join(UPLOAD_FOLDER, "temp_audio.wav")

                try:
                    video = VideoFileClip(path)
                    video.audio.write_audiofile(audio_path, logger=None)

                    converted = convert_audio(audio_path)
                    audio_result = detect_audio_emotion(converted)
                    transcription = transcribe_audio(converted)

                    sorted_audio = sorted(audio_result.items(), key=lambda x: x[1], reverse=True)
                    top_audio_emotion = sorted_audio[0][0]
                    if len(sorted_audio) > 1 and np.isclose(sorted_audio[0][1], sorted_audio[1][1], atol=1e-4):
                        top_audio_emotion = "neutral"

                    text_result = classify_text_emotion([transcription])[0]
                    cause = extract_emotion_cause(transcription)

                    emotion_summary = {
                        "type": "Video",
                        "vis_emotion": vis_emo,
                        "vis_scores": vis_scores,
                        "audio_emotion": top_audio_emotion,
                        "transcription": transcription,
                        "text_emotion": text_result,
                        "cause": cause
                    }

                    os.remove(converted)
                    os.remove(audio_path)
                except Exception as e:
                    emotion_summary = {
                        "error": f"Video processing failed: {str(e)}"
                    }

        # === Generate LLM Inference with LLaMA ===
        if emotion_summary and not emotion_summary.get("error"):
            inference = generate_llama_inference(emotion_summary)
            emotion_summary["inference"] = inference

    return render_template("index.html", emotion_summary=emotion_summary)

if __name__ == "__main__":
    app.run(debug=True)
