import speech_recognition as sr
import time
import pyttsx3
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import re


# Initialize the recognizer
r = sr.Recognizer()

# Load the hugging face API key
load_dotenv()

## Initialize the model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Use the microphone as a source of audio
with sr.Microphone() as source:
    print("Adjusting for ambient noise, please wait...")
    # Adjusts for ambient noise for 1 second (default)
    r.adjust_for_ambient_noise(source, duration=1) 
    print("Listening now... Speak into your microphone.")
    # Listen for the first phrase and extract the audio data
    audio = r.listen(source)
     

try:
    # Using Google Speech Recognition to convert audio to text
    text = r.recognize_google(audio)
    print(f"You said: {text}")

    # pass this input to the model and save the response
    response = model.invoke(text)
    print(response.content)
    
    # remove unnecessary punctuations/symbols for the speech to be clear
    pattern = r"[^a-z A-z.,]"
    result = re.sub(pattern,"",response.content)
    print(result)

    #convert response to speech
    engine = pyttsx3.init()
    engine.say(result)
    engine.runAndWait()
    del engine
except sr.UnknownValueError:
    # Error handling for unintelligible audio
    print("Sorry, I could not understand the audio.")
except sr.RequestError as e:
    # Error handling for API connection issues
    print(f"Could not request results from Google Speech Recognition service; {e}.")
