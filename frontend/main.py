import streamlit as st
from typing import Dict, List
import json
from collections import Counter
import re
import difflib
import boto3

from backend.chat import BedrockChat
from backend.get_transcript import YouTubeTranscriptDownloader
from backend.language_learning_assistant import LanguageLearningAssistant
from backend.rag import TranscriptVectorStore

# Page config
st.set_page_config(
    page_title="Spanish Learning Assistant",
    page_icon="🇪🇸",
    layout="wide"
)

# Initialize session state
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

def render_header():
    """Render the header section"""
    st.title("🇪🇸 Spanish Learning Assistant")
    st.markdown("""
    Transform YouTube transcripts into interactive Spanish learning experiences.
    
    This tool demonstrates:
    - Base LLM Capabilities
    - RAG (Retrieval Augmented Generation)
    - Amazon Bedrock Integration
    - Agent-based Learning Systems
    """)

def render_sidebar():
    """Render the sidebar with component selection"""
    with st.sidebar:
        st.header("Development Stages")
        
        # Main component selection
        selected_stage = st.radio(
            "Select Stage:",
            [
                "1. Chat with Nova",
                "2. Raw Transcript",
                "3. Structured Data",
                "4. RAG Implementation",
                "5. Interactive Learning"
            ]
        )
        
        # Stage descriptions
        stage_info = {
            "1. Chat with Nova": """
            **Current Focus:**
            - Basic Spanish learning
            - Understanding LLM capabilities
            - Identifying limitations
            """,
            
            "2. Raw Transcript": """
            **Current Focus:**
            - YouTube transcript download
            - Raw text visualization
            - Initial data examination
            """,
            
            "3. Structured Data": """
            **Current Focus:**
            - Text cleaning
            - Dialogue extraction
            - Data structuring
            """,
            
            "4. RAG Implementation": """
            **Current Focus:**
            - Bedrock embeddings
            - Vector storage
            - Context retrieval
            """,
            
            "5. Interactive Learning": """
            **Current Focus:**
            - Scenario generation
            - Audio synthesis
            - Interactive practice
            """
        }
        
        st.markdown("---")
        st.markdown(stage_info[selected_stage])
        
        return selected_stage

def render_chat_stage():
    """Render an improved chat interface"""
    st.header("Chat with Nova")

    # Initialize BedrockChat instance if not in session state
    if 'bedrock_chat' not in st.session_state:
        st.session_state.bedrock_chat = BedrockChat()

    # Introduction text
    st.markdown("""
    Start by exploring Nova's base Spanish language capabilities. Try asking questions about Spanish grammar, 
    vocabulary, or cultural aspects.
    """)

    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    # Chat input area
    if prompt := st.chat_input("Ask about Spanish language..."):
        # Process the user input
        process_message(prompt)

    # Example questions in sidebar
    with st.sidebar:
        st.markdown("### Try These Examples")
        example_questions = [
            "How do I say 'Where is the train station?' in Spanish?",
            "Explain the difference between ser and estar",
            "What's the difference between preterite and imperfect?",
            "How do I conjugate irregular verbs in Spanish?",
            "What's the difference between por and para?",
            "How do I ask for directions politely?"
        ]
        
        for q in example_questions:
            if st.button(q, use_container_width=True, type="secondary"):
                # Process the example question
                process_message(q)
                st.rerun()

    # Add a clear chat button
    if st.session_state.messages:
        if st.button("Clear Chat", type="primary"):
            st.session_state.messages = []
            st.rerun()

def process_message(message: str):
    """Process a message and generate a response"""
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": message})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(message)

    # Generate and display assistant's response
    with st.chat_message("assistant", avatar="🤖"):
        response = st.session_state.bedrock_chat.generate_response(message)
        if response:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})



def count_characters(text):
    """Count Spanish and total characters in text"""
    if not text:
        return 0, 0
        
    def is_spanish(char):
        return any([
            char in 'áéíóúñü¿¡',
            char.isalpha()
        ])
    
    sp_chars = sum(1 for char in text if is_spanish(char))
    return sp_chars, len(text)

def render_transcript_stage():
    """Render the raw transcript stage"""
    st.header("Raw Transcript Processing")
    
    # URL input
    url = st.text_input(
        "YouTube URL",
        placeholder="Enter a Spanish lesson YouTube URL"
    )
    
    # Download button and processing
    if url:
        if st.button("Download Transcript"):
            try:
                downloader = YouTubeTranscriptDownloader()
                transcript = downloader.get_transcript(url)
                if transcript:
                    # Store the raw transcript text in session state
                    transcript_text = "\n".join([entry['text'] for entry in transcript])
                    st.session_state.transcript = transcript_text
                    st.success("Transcript downloaded successfully!")
                else:
                    st.error("No transcript found for this video.")
            except Exception as e:
                st.error(f"Error downloading transcript: {str(e)}")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Transcript")
        if st.session_state.transcript:
            st.text_area(
                label="Raw text",
                value=st.session_state.transcript,
                height=400,
                disabled=True
            )
    
        else:
            st.info("No transcript loaded yet")
    
    with col2:
        st.subheader("Transcript Stats")
        if st.session_state.transcript:
            # Calculate stats
            sp_chars, total_chars = count_characters(st.session_state.transcript)
            total_lines = len(st.session_state.transcript.split('\n'))
            
            # Display stats
            st.metric("Total Characters", total_chars)
            st.metric("Spanish Characters", sp_chars)
            st.metric("Total Lines", total_lines)
        else:
            st.info("Load a transcript to see statistics")

def render_structured_stage():
    """Render the structured data stage"""
    st.header("Structured Data Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dialogue Extraction")
        # Placeholder for dialogue processing
        st.info("Dialogue extraction will be implemented here")
        
    with col2:
        st.subheader("Data Structure")
        # Placeholder for structured data view
        st.info("Structured data view will be implemented here")

def render_rag_stage():
    """Render the RAG implementation stage"""
    st.header("RAG System")
    
    # Query input
    query = st.text_input(
        "Test Query",
        placeholder="Enter a question about Spanish..."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Retrieved Context")
        # Placeholder for retrieved contexts
        st.info("Retrieved contexts will appear here")
        
    with col2:
        st.subheader("Generated Response")
        # Placeholder for LLM response
        st.info("Generated response will appear here")

def render_interactive_stage():
    """Render the interactive learning stage"""
    st.header("Interactive Learning")
    
    # Initialize the Language Learning Assistant if not already done
    if 'learning_assistant' not in st.session_state:
        try:
            st.session_state.learning_assistant = LanguageLearningAssistant()
        except Exception as e:
            st.error(f"Error initializing Learning Assistant: {e}")
            return
    
    # Practice type selection
    practice_type = st.selectbox(
        "Select Question Type",
        ["Comprehension", "Vocabulary", "Grammar", "Listening"],
        help="Select the type of question you want to generate"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Conversation & Question")
        
        # Generate button
        if st.button("Generate New Question", type="primary"):
            with st.spinner("Generating question..."):
                try:
                    # Generate exercise using RAG
                    exercise = st.session_state.learning_assistant.generate_learning_exercise(
                        practice_type.lower()
                    )
                    
                    if exercise:
                        # Store the exercise in session state
                        st.session_state.current_exercise = exercise
                    else:
                        st.error("Could not generate a question. Please try again.")
                except Exception as e:
                    st.error(f"Error generating question: {e}")
        
        # Display current exercise if available
        if 'current_exercise' in st.session_state and st.session_state.current_exercise:
            exercise = st.session_state.current_exercise
            
            # Display conversation context
            st.markdown("### Conversation")
            conversation_lines = exercise['context'].split('\n')
            for line in conversation_lines:
                if line.strip():
                    # Check if it's a speaker line
                    if ':' in line:
                        speaker, text = line.split(':', 1)
                        st.markdown(f"**{speaker}:** {text.strip()}")
                    else:
                        st.markdown(line)
            
            st.markdown("---")  # Divider between conversation and question
            
            # Display question
            st.markdown("### Question")
            st.markdown(f"🇪🇸 *{exercise['question']['question_spanish']}*")
            st.markdown(f"🇬🇧 {exercise['question']['question_english']}")
            
            # Answer input
            user_answer = st.text_input(
                "Your Answer (in Spanish)",
                key="user_answer",
                help="Type your answer in Spanish"
            )
            
            if st.button("Check Answer"):
                if user_answer.strip():
                    # Compare with correct answer
                    correct_answer = exercise['question']['answer_spanish']
                    similarity = difflib.SequenceMatcher(None, 
                        user_answer.lower().strip(), 
                        correct_answer.lower().strip()
                    ).ratio()
                    
                    # Show result
                    if similarity > 0.8:
                        st.success("¡Correcto! (Correct!) 🎉")
                    else:
                        st.error("Not quite right. Try again! 🤔")
                    
                    # Show correct answer
                    st.markdown("### Correct Answer")
                    st.markdown(f"🇪🇸 *{exercise['question']['answer_spanish']}*")
                    st.markdown(f"🇬🇧 {exercise['question']['answer_english']}")
                else:
                    st.warning("Please enter an answer first.")
    
    with col2:
        st.subheader("Exercise Information")
        
        # Display metadata about the current exercise
        if 'current_exercise' in st.session_state and st.session_state.current_exercise:
            st.metric("Question Type", practice_type)
            st.markdown("**Difficulty Level:** A1 (Beginner)")
            
            # Tips section
            with st.expander("Tips & Hints"):
                st.markdown("""
                - Read the conversation carefully
                - Pay attention to who is speaking
                - Look for key words and phrases
                - Consider the context
                - Think about the type of question being asked
                """)
            
            # Show conversation transcript button
            with st.expander("Show Full Conversation"):
                st.markdown(exercise['context'])
        else:
            st.info("Generate a question to get started!")

def main():
    render_header()
    selected_stage = render_sidebar()
    
    # Render appropriate stage
    if selected_stage == "1. Chat with Nova":
        render_chat_stage()
    elif selected_stage == "2. Raw Transcript":
        render_transcript_stage()
    elif selected_stage == "3. Structured Data":
        render_structured_stage()
    elif selected_stage == "4. RAG Implementation":
        render_rag_stage()
    elif selected_stage == "5. Interactive Learning":
        render_interactive_stage()
    
    # Debug section at the bottom
    with st.expander("Debug Information"):
        st.json({
            "selected_stage": selected_stage,
            "transcript_loaded": st.session_state.transcript is not None,
            "chat_messages": len(st.session_state.messages)
        })

if __name__ == "__main__":
    main()