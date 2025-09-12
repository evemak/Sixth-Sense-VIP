import subprocess
import os
from transformers import pipeline


def text_to_speech_offline(text, voice="en_US/ljspeech_low", output_file="output.wav"):
    # Define the mimic3_tts command
    mimic3_cmd = [
        "mimic3",    # Mimic 3 command
        "--voice", voice,  # specific voice
        "--stdout",      # output to stdout
        text             # convert text
    ]
    try:
        # Run the Mimic 3 command and capture the output audio
        with open(output_file, "wb") as audio_file:
            subprocess.run(mimic3_cmd, stdout=audio_file, check=True)
        print(f"Generated speech saved to {output_file}")

        # Play the generated audio file
        os.system(f"afplay {output_file}")  # For Linux systems with 'aplay'
    except FileNotFoundError:
        print("Mimic 3 is not installed or not found in PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error generating speech: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
