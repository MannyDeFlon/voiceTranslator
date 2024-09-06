import gradio as gr
from dotenv import load_dotenv
import whisper
from translate import Translator
from elevenlabs.client import ElevenLabs
import os


load = load_dotenv()

elevens_api_key = os.getenv("ELEVENLABS_API_KEY")

def voice_translator(audio_file):

    # transcribir audio
    try:
        model = whisper.load_model("base")  # base es el modelo mas pequeño, small es otra opcion
        result = model.transcribe(audio_file, language="Spanish", fp16=False)
        transcription = result["text"]
    except Exception as e:
        gr.Error(f"Hubo un error al transcribir el audio: {str(e)}")
    
    print("The transcribed text is: ", transcription)

    # traducir texto
    try:
        en_translation = Translator(from_lang="es", to_lang="en").translate(transcription)

    except Exception as e:
        gr.Error(f"Hubo un error al traducir el texto: {str(e)}")

    print("The translation is: ", en_translation)

    # generar audio
    try: 
        client = ElevenLabs(elevens_api_key)
        audio = client.text_to_speech.convert(text=en_translation,
                                voice="Rachel",
                                model="eleven_monolingual_v2")
    except Exception as e:
        gr.Error(f"Hubo un error al generar el audio: {str(e)}")

    return audio



web = gr.Interface(
                   fn=voice_translator,
                   inputs=gr.Audio(sources=["microphone"], type="filepath"),
                   outputs=gr.Audio(),
                   title="Voice Translator",
                   description="Translate voice from different lenguages in real time."
                  )	

if __name__ == "__main__":
    web.launch()
