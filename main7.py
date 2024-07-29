import streamlit as st
from gtts import gTTS
import sounddevice as sd
import scipy.io.wavfile as wav
import assemblyai as aai
import time
from pydub import AudioSegment
from pydub.playback import play
from gradio_client import Client
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from PyPDF2 import PdfReader
import io

# Set your AssemblyAI API key
aai.settings.api_key = "103ebf64fbd147368ed78b2064138f43"

# Initialize Gradio Client for LLM with retry mechanism
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3), retry=retry_if_exception_type(httpx.RequestError))
def initialize_gradio_client():
    return Client("osanseviero/mistral-super-fast")

client = initialize_gradio_client()

# Define roles with voices
roles = {
    "HR": ("HR", "en"),
    "Candidate": ("Candidate", None)  # No voice needed for "Candidate"
}

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0

if "recording" not in st.session_state:
    st.session_state.recording = False

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "interview_active" not in st.session_state:
    st.session_state.interview_active = False

if "follow_up_active" not in st.session_state:
    st.session_state.follow_up_active = False

if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = []

if "feedback" not in st.session_state:
    st.session_state.feedback = ""

# Function to get TTS audio using gTTS
def get_tts_audio(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save("temp_audio.mp3")
    return "temp_audio.mp3"

# Function to play audio from a file path
def play_audio_from_file(file_path):
    audio = AudioSegment.from_file(file_path)
    play(audio)
    time.sleep(audio.duration_seconds)

# Function to record audio
def start_recording(filename, sample_rate=44100):
    st.session_state.recording = True
    recording = sd.rec(int(60 * sample_rate), samplerate=sample_rate, channels=1)
    st.session_state['recording_obj'] = recording

def stop_recording(filename, sample_rate=44100):
    sd.stop()
    wav.write(filename, sample_rate, st.session_state['recording_obj'])
    st.session_state.recording = False

# Function to transcribe audio using AssemblyAI with retry logic
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3), retry=retry_if_exception_type(httpx.RequestError))
def transcribe_audio(filename):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(filename)
    return transcript.text

# Function to get response from LLM
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3), retry=retry_if_exception_type(httpx.RequestError))
def get_response(prompt):
    result = client.predict(
        prompt=prompt,
        temperature=0.7,
        max_new_tokens=300,
        top_p=0.9,
        repetition_penalty=1.2,
        api_name="/chat"
    )
    response = result.strip()
    return response

# Function to generate interview questions based on resume
def generate_interview_questions(resume_text):
    prompt = (
        "Based on the following resume, generate a set of interview questions. "
        "The questions should be tailored to the candidate's experience and skills listed in the resume, "
        "and should be relevant to a typical job interview for a position in the related field.\n\n"
        "Resume:\n{resume_text}\n\nQuestions:"
    ).format(resume_text=resume_text)
    questions = get_response(prompt)
    return [q.strip() for q in questions.split('\n') if q.strip()]

# Function to ask the next question
def ask_next_question():
    if st.session_state.current_question_index == 0:
        # Always start with "Introduce yourself."
        question = "Introduce yourself."
    elif st.session_state.current_question_index < len(st.session_state.interview_questions) + 1:
        question = st.session_state.interview_questions[st.session_state.current_question_index - 1]
    else:
        st.session_state.chat_history += "\n\nHR: That concludes our interview. Thank you."
        audio_file_path = get_tts_audio("That concludes our interview. Thank you.", roles["HR"][1])
        play_audio_from_file(audio_file_path)
        st.session_state.interview_active = False  # End the interview
        return
    
    st.session_state.chat_history += f"\n\nHR: {question}"
    audio_file_path = get_tts_audio(question, roles["HR"][1])
    play_audio_from_file(audio_file_path)
    start_recording("my_local_audio_file.wav")  # Start recording after asking question

# Function to generate follow-up questions using LLM
def generate_follow_up(question, response):
    prompt = f"As an HR, generate a follow-up question for the candidate based on their response.\n\nHR: {question}\nCandidate: {response}\nHR:"
    follow_up = get_response(prompt)
    return follow_up

# Function to generate feedback on the user's performance
def generate_feedback(interview_history):
    prompt = (
        "Based on the following interview history, provide feedback on the candidate's performance. "
        "Highlight their strengths and areas for improvement to help them perform better in real-life interviews.\n\n"
        "Interview History:\n{interview_history}\n\nFeedback:"
    ).format(interview_history=interview_history)
    feedback = get_response(prompt)
    return feedback

# Function to process the response
def process_response():
    stop_recording("my_local_audio_file.wav")  # Stop recording
    user_input = transcribe_audio("my_local_audio_file.wav")
    st.session_state.user_input = user_input
    st.session_state.chat_history += f"\n\nCandidate: {user_input}"

    if st.session_state.follow_up_active:
        # If follow-up was active, go to the next question
        st.session_state.current_question_index += 1
        st.session_state.follow_up_active = False
        ask_next_question()
    else:
        # Generate follow-up question based on user input
        follow_up = generate_follow_up(st.session_state.interview_questions[st.session_state.current_question_index - 1], user_input)
        st.session_state.chat_history += f"\n\nHR: {follow_up}"
        audio_file_path = get_tts_audio(follow_up, roles["HR"][1])
        play_audio_from_file(audio_file_path)
        start_recording("my_local_audio_file.wav")  # Start recording for follow-up
        st.session_state.follow_up_active = True

# Streamlit interface
st.title("HR Interview Simulator with TTS")
st.write("Simulate an HR interview where the HR bot asks questions based on your resume.")

# Prompt the user to upload their resume
resume_file = st.file_uploader("Upload your resume (PDF format)", type="pdf")

if resume_file is not None:
    # Extract text from the uploaded PDF
    pdf_reader = PdfReader(resume_file)
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text()

    # Generate interview questions based on the resume
    additional_questions = generate_interview_questions(resume_text)
    st.session_state.interview_questions = additional_questions
    st.success("Interview questions generated based on your resume. You can start the interview.")

# Display the chat history
st.text_area("Interview", st.session_state.chat_history, height=300)

# Start interview button
if st.button("Start Interview") and st.session_state.current_question_index == 0:
    st.session_state.interview_active = True
    st.session_state.follow_up_active = False
    ask_next_question()

# Submit response button
if st.button("Submit Response") and st.session_state.recording:
    process_response()
    st.experimental_rerun()  # Rerun to refresh the interface

# Conclude interview button
if st.button("Conclude Interview"):
    st.session_state.chat_history += "\n\nHR: That concludes our interview. Thank you."
    audio_file_path = get_tts_audio("That concludes our interview. Thank you.", roles["HR"][1])
    play_audio_from_file(audio_file_path)
    
    # Generate and display feedback
    feedback = generate_feedback(st.session_state.chat_history)
    st.session_state.feedback = feedback
    
    st.session_state.interview_active = False
    st.text_area("Interview", st.session_state.chat_history, height=300)
    st.text_area("Feedback", st.session_state.feedback, height=300)
