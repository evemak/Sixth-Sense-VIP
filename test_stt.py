import threading
import sounddevice as sd
import numpy as np
import whisper
import tempfile
import scipy.io.wavfile as wav

class WhisperTranscriber(threading.Thread):
    def __init__(self, duration=10, sample_rate=16000, model_name="base", **kwargs):
        """
        Parameters:
          - duration: Recording duration in seconds.
          - sample_rate: Sampling rate for the recording.
          - model_name: The Whisper model variant to load ("base", "small", etc.).
        """
        super().__init__(**kwargs)
        self.duration = duration
        self.sample_rate = sample_rate
        self.model_name = model_name
        self.transcription = None

    def record_audio(self):
        print(f"Recording for {self.duration} seconds...")
        audio = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='int16')
        sd.wait()
        print("Recording complete.")
        return audio

    def save_temp_wav(self, audio):
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.write(temp_file.name, self.sample_rate, audio)
        print(f"Audio saved temporarily at: {temp_file.name}")
        return temp_file.name

    def transcribe_audio(self, audio_path):
        print(f"Loading Whisper model '{self.model_name}'...")
        model = whisper.load_model(self.model_name)
        print("Transcribing audio...")
        result = model.transcribe(audio_path)
        return result["text"]

    def run(self):
        audio = self.record_audio()
        audio_path = self.save_temp_wav(audio)
        self.transcription = self.transcribe_audio(audio_path)
        print("Transcription complete.")
        print("\nTranscription:")
        print(self.transcription)

if __name__ == "__main__":
    # Example of using the thread directly.
    transcriber = WhisperTranscriber(duration=5, sample_rate=16000, model_name="base")
    transcriber.start()
    transcriber.join()  # Wait for the thread to finish.
