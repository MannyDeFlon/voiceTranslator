import gradio as gr
from dotenv import load_dotenv
import whisper
from translate import Translator
from elevenlabs.client import ElevenLabs
import os


load = load_dotenv()

elevens_api_key = os.getenv("ELEVENLABS_API_KEY")
save_file_path = "audio.mp3"


def voice_translator(audio_file : str, to_lang : str = "en") -> str:
    """
    Translate voice from different lenguages in real time.

    audio_file: path to the audio file
    to_lang: language to translate to

    returns: path to the saved file
    """

    # transcribir audio
    try:
        model = whisper.load_model("base")  # base es el modelo mas pequeño, small es otra opcion
        result = model.transcribe(audio_file, language="Spanish", fp16=False)
        transcription = result["text"]
    except Exception as e:
        gr.Error(f"Hubo un error al transcribir el audio: {str(e)}")

    # traducir texto
    try:
        en_translation = Translator(from_lang="es", to_lang="en").translate(transcription)
        fr_translation = Translator(from_lang="es", to_lang="fr").translate(transcription)


    except Exception as e:
        gr.Error(f"Hubo un error al traducir el texto: {str(e)}")

    # generar audio
    try: 
        text_to_speech(en_translation, language="en")

    except Exception as e:
        gr.Error(f"Hubo un error al generar el audio: {str(e)}")
        print(f"Hubo un error al generar el audio: {str(e)}")

    return save_file_path  

def text_to_speech(text: str, language: str = "en") -> str:
    """
    Convert text to speech and save it to a file. Returns path to the saved file.
    """

    client = ElevenLabs()

    response = client.text_to_speech.convert(text=text,
                                            voice_id="pNInz6obpgDQGcFmaJgB",  # Adam
                                            optimize_streaming_latency="0",
                                            output_format="mp3_22050_32",
                                            model_id="eleven_multilingual_v2")

    save_file_path = f"audio_{language}.mp3"

    with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

    return save_file_path


web = gr.Interface(
                   fn=voice_translator,
                   inputs=gr.Audio(sources=["microphone"], type="filepath"),
                   outputs=gr.Audio(),
                   title="Voice Translator",
                   description="Translate voice from different lenguages in real time."
                  )	

if __name__ == "__main__":
    web.launch()
