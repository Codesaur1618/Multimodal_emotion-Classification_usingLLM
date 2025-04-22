import os
from moviepy import VideoFileClip
from text_utils import classify_text_emotion, extract_emotion_cause
from audio_utils import convert_audio, transcribe_audio, detect_audio_emotion
from video_utils import detect_visual_emotion
import numpy as np

def handle_text_input():
    text = input("Enter your text: ")
    result = classify_text_emotion([text])
    cause = extract_emotion_cause(text)
    print("\n=== Text Emotion ===")
    print(f"Text: {text}")
    print(f"Emotion: {result[0]['emotion']} ({result[0]['confidence']:.4f})")
    print(f"Cause of Emotion: {cause}")

def handle_audio_input():
    audio_path = input("Enter path to audio file (.mp3 or .wav): ")
    if not os.path.exists(audio_path):
        print("File not found.")
        return
    converted = convert_audio(audio_path)
    transcription = transcribe_audio(converted)
    print("\n=== Transcribed Text ===")
    print(transcription)
    text_result = classify_text_emotion([transcription])[0]
    audio_result = detect_audio_emotion(converted)
    cause = extract_emotion_cause(transcription)

    print("\n=== Audio Emotion ===")
    top_two_audio = list(audio_result.items())[:2]
    final_audio_emo = "neutral" if len(top_two_audio) > 1 and np.isclose(top_two_audio[0][1], top_two_audio[1][1], atol=1e-4) else top_two_audio[0][0]
    print(f"Final Audio Emotion: {final_audio_emo}")

    print("\n=== Transcribed Text Emotion ===")
    print(f"{text_result['emotion']} ({text_result['confidence']:.4f})")
    print(f"Cause of Emotion: {cause}")
    os.remove(converted)

def handle_video_input():
    video_path = input("Enter path to video file: ")
    if not os.path.exists(video_path):
        print("Video file not found.")
        return
    temp_audio = "video_temp_audio.wav"

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio)
    converted = convert_audio(temp_audio)

    vis_emo, vis_scores = detect_visual_emotion(video_path)
    print("\n=== Visual Emotion ===")
    print(f"Detected Visual Emotion: {vis_emo}")

    audio_result = detect_audio_emotion(converted)
    print("\n=== Audio Emotion ===")
    top_two_audio = list(audio_result.items())[:2]
    final_audio_emo = "neutral" if len(top_two_audio) > 1 and np.isclose(top_two_audio[0][1], top_two_audio[1][1], atol=1e-4) else top_two_audio[0][0]
    print(f"Final Audio Emotion: {final_audio_emo}")

    transcription = transcribe_audio(converted)
    text_result = classify_text_emotion([transcription])[0]
    cause = extract_emotion_cause(transcription)

    print("\n=== Transcribed Text ===")
    print(transcription)
    print("\n=== Text Emotion ===")
    print(f"{text_result['emotion']} ({text_result['confidence']:.4f})")
    print(f"Cause of Emotion: {cause}")

    os.remove(temp_audio)
    os.remove(converted)

    print("\n=== Final Emotion Summary ===")
    print(f"Visually, the person appears to be experiencing **{vis_emo}**.")
    print(f"From the audio tone, they sound **{final_audio_emo}**.")
    print(f"Their spoken words convey **{text_result['emotion']}** (confidence: {text_result['confidence']:.4f}) — caused by the phrase: \"{cause}\".")

def main():
    print("Choose input type:")
    print("1. Text")
    print("2. Audio (.mp3/.wav)")
    print("3. Video")
    choice = input("Enter choice (1/2/3): ")

    if choice == "1":
        handle_text_input()
    elif choice == "2":
        handle_audio_input()
    elif choice == "3":
        handle_video_input()
    else:
        print("Invalid option.")

if __name__ == "__main__":
    main()