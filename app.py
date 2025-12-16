import streamlit as st
from unsloth import FastVisionModel
import torch
from PIL import Image
from transformers import TextIteratorStreamer
import threading
import os

# Set page configuration
st.set_page_config(
    page_title="Medical VLM Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better layout
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for Image Upload and Controls
with st.sidebar:
    st.header("🖼️ Image Upload")
    uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "jpeg", "png"])
    
    st.divider()
    
    st.header("⚙️ Controls")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Model Info")
    st.caption("Model: Qwen2.5-VL-7B (LoRA Fine-tuned)")
    st.caption("Task: Medical Image Analysis")

# Main Area
st.title("🏥 Medical Assistant")
st.markdown("Upload a medical image and ask questions to the AI radiologist.")

# Load Model (Cached) with improved status display
@st.cache_resource
def load_model():
    model_path = "/root/autodl-tmp/lora_model"
    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_path,
            load_in_4bit=True,
            local_files_only=True,
        )
        FastVisionModel.for_inference(model)
        return model, tokenizer
    except Exception as e:
        return None, None

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# Handle Image Upload
if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state.uploaded_image = image
    
    # Display image in sidebar
    with st.sidebar:
        st.image(image, caption="Current Image", use_container_width=True)
    
    # Also display image in main area (collapsible) for better visibility
    with st.expander("👁️ View High-Resolution Image", expanded=True):
        st.image(image, use_container_width=True)

# Load the model
with st.status("🚀 Loading Medical VLM...", expanded=True) as status:
    st.write("Initializing model architecture...")
    model, tokenizer = load_model()
    
    if model:
        status.update(label="✅ Model Loaded Successfully!", state="complete", expanded=False)
    else:
        status.update(label="❌ Model Loading Failed", state="error")
        st.error("Failed to load model. Please check logs.")
        st.stop()


# Display Chat History
st.subheader("💬 Diagnosis & Discussion")
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        if st.session_state.uploaded_image:
            st.info("👋 Image uploaded! Ask a question below to start the diagnosis.")
        else:
            st.info("👈 Please upload a medical image in the sidebar to get started.")

    for msg in st.session_state.messages:
        avatar = "🧑‍⚕️" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask about the image (e.g., 'Describe the pathology')..."):
    # Check if image is present
    if st.session_state.uploaded_image is None:
        st.warning("⚠️ Please upload an image first to start the analysis.")
    else:
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(prompt)
        
        # Prepare inputs for the model
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Construct messages for the model
                model_messages = []
                
                # Add a system message to encourage detailed responses
                system_prompt = "你是一名专业的放射科医生。请详细描述你在图片中看到的内容，包括病变的位置、形态特征以及可能的诊断依据。请避免简短的回答，尽可能提供详尽的分析。"
                # model_messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
                
                image_included = False
                
                # We reconstruct the full conversation history for the model
                for i, msg in enumerate(st.session_state.messages):
                    content = []
                    if msg["role"] == "user":
                        # Enhance the first user message with the system prompt instructions if it's the very first interaction
                        if i == 0:
                            text_content = f"{system_prompt}\n\n用户问题: {msg['content']}"
                        else:
                            text_content = msg["content"]
                            
                        content.append({"type": "text", "text": text_content})
                        # Attach image to the first user message
                        if not image_included:
                            content.append({"type": "image"})
                            image_included = True
                        model_messages.append({"role": "user", "content": content})
                    else:
                        model_messages.append({"role": "assistant", "content": [{"type": "text", "text": msg["content"]}]})
                
                # Apply chat template
                input_text = tokenizer.apply_chat_template(model_messages, add_generation_prompt=True)
                
                # Debug: Show the prompt being sent to the model
                with st.expander("🛠️ Debug: Model Input Prompt"):
                    st.code(input_text, language="text")

                # Prepare model inputs
                inputs = tokenizer(
                    st.session_state.uploaded_image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to("cuda")
                
                # Streamer setup
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                # Clear CUDA cache before generation to prevent OOM
                torch.cuda.empty_cache()

                generation_kwargs = dict(
                    inputs,
                    streamer=streamer,
                    max_new_tokens=1024, # Increased token limit for longer responses
                    use_cache=True,
                    temperature=1.0, # Increased slightly for more diversity
                    repetition_penalty=1.1, # Prevent repetition
                    min_p=0.1
                )
                
                # Run generation in a separate thread
                thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Consume stream
                for new_text in streamer:
                    full_response += new_text
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error during generation: {e}")
