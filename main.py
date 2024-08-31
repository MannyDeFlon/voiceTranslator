import gradio as gr
from dotenv import load_dotenv


def voice_translator(audio):
    

	return translated_audio

gr.Interfgace(
	fn=voice_translator,       # función que lleva la lógica de la ágina
	inputs="audio",  # elementos de entrada (Audio en este caso) 
	outputs="audio"  # elementos de salida
)