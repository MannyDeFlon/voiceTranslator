import gradio as gr
from dotenv import load_dotenv
import whisper
from translate import Translator
from elevenlabs.client import ElevenLabs
import os


load = load_dotenv()

ELEVENS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

languages_dict = {"English": "en", 
                  "Spanish": "es", 
                  "French": "fr", 
                  "German": "de", 
                  "Italian": "it", 
                  "Japanese": "ja", 
                  "Korean": "ko", 
                  "Portuguese": "pt", 
                  "Russian": "ru", 
                  "Chinese": "zh"}

whisper_model = "base" # base es el modelo mas pequeño, small es otra opcion


def voice_translator(audio_file : str, to_lang : str = "en") -> str:
    """
    Translate voice from different lenguages in real time.

    audio_file: path to the audio file
    to_lang: language to translate to

    returns: path to the saved file
    """

    # transcribir audio
    try:
        model = whisper.load_model(whisper_model)
        result = model.transcribe(audio_file, language="Spanish", fp16=False)
        transcription = result["text"]
    except Exception as e:
        gr.Error(f"Hubo un error al transcribir el audio: {str(e)}")
        print(f"Hubo un error al transcribir el audio: {str(e)}")
    # TODO: add transcription writing to file

    # traducir texto
    try:
        translation = Translator(from_lang="es", to_lang=languages_dict[to_lang]).translate(transcription)
    except Exception as e:
        gr.Error(f"Hubo un error al traducir el texto: {str(e)}")
        print(f"Hubo un error al traducir el texto: {str(e)}")
    # TODO: add translation writing to file

    # generar audio
    try: 
        audio = text_to_speech(translation, language=to_lang)

    except Exception as e:
        gr.Error(f"Hubo un error al generar el audio: {str(e)}")
        print(f"Hubo un error al generar el audio: {str(e)}")

    return audio  

def text_to_speech(text: str, language: str = "en") -> str:
    """
    Convert text to speech and save it to a file. Returns path to the saved file.
    """

    client = ElevenLabs(api_key=ELEVENS_API_KEY)

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
                   inputs=[gr.Audio(sources=["microphone"], type="filepath"), 
                           gr.Dropdown(label="Language to", choices=languages_dict.keys())],
                   outputs=gr.Audio(),
                   title="Voice Translator",
                   description="Translate voice from different lenguages in real time."
                  )	

if __name__ == "__main__":
    web.launch()
