import sounddevice as sd
import numpy as np
import whisper
import tempfile
import scipy.io.wavfile as wav

# default
duration = 5  
sample_rate = 16000 # sampling rate  

def record_audio(duration, sample_rate):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete.")
    return audio

def save_temp_wav(audio, sample_rate):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(temp_file.name, sample_rate, audio)
    return temp_file.name

def transcribe_with_whisper(audio_path):
    model = whisper.load_model("base")  
    result = model.transcribe(audio_path)
    return result['text']

if __name__ == "__main__":
    print("Now Recording")
    audio = record_audio(duration, sample_rate)
    print("Recording finished.")
    audio_path = save_temp_wav(audio, sample_rate)
    transcription = transcribe_with_whisper(audio_path)
    print("Transcription complete.")
    print("\nTranscription:")
    print(transcription)
