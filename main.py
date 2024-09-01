import gradio as gr
from dotenv import load_dotenv
import whisper
from translate import Translator

load = load_dotenv()


def voice_translator(audio_file):

	# transcribir audio
	try:
		model = whisper.load_model("base")  # base es el modelo mas pequeño, small es otra opcion
		result = model.transcribe(audio_file, language="spanish")
		translation = result["text"]
	except Exception as e:
		gr.Error(f"Hubo un error al transcribir el audio: {str(e)}")
	
	# traducir texto
	try:
		translator = Translator(to_lang="en")
		translation = translator.translate(translation)

	except Exception as e:
		gr.Error(f"Hubo un error al traducir el texto: {str(e)}")
	# generar audio
	
	pass


web = gr.Interface(
                   fn=voice_translator,
				   inputs=gr.Audio(sources=["microphone"], type="filepath"),
				   outputs=gr.Audio(),
				   title="Voice Translator",
				   description="Translate voice from different lenguages in real time."
				  )	

if __name__ == "__main__":
	web.launch()
