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



# Main Area
st.title("🏥 Medical Assistant")
st.markdown("Upload a medical image and ask questions to the AI radiologist.")

# Load Model (Cached) with improved status display
@st.cache_resource
def load_model(model_selection="SFT (LoRA)"):
    # Determine model path based on selection
    if model_selection == "SFT (LoRA)":
        model_path = "/root/autodl-tmp/lora_model"
        base_model_path = "/root/autodl-tmp/models/unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
    elif model_selection == "GRPO (RL)":
        model_path = "/root/autodl-tmp/grpo_model"
        base_model_path = "/root/autodl-tmp/models/unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
    else:
        return None, None

    try:
        # Load model with specific adapter
        # Unsloth's from_pretrained can handle both base+adapter loading if path points to adapter
        # But to be safe and support switching, we might want to load base then adapter
        # However, for efficiency in Streamlit (caching), we'll let FastVisionModel handle it
        
        # Check if the specific model path exists
        if not os.path.exists(model_path):
            st.error(f"❌ Model path not found: {model_path}")
            return None, None

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_path, # This loads base + adapter if it's a PEFT model
            load_in_4bit=True,
            local_files_only=True,
        )
        FastVisionModel.for_inference(model)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# Sidebar for Image Upload and Controls
with st.sidebar:
    st.header("🖼️ Image Upload")
    
    # Model Selection
    st.subheader("🤖 Model Selection")
    model_option = st.selectbox(
        "Choose Model Version:",
        ("SFT (LoRA)", "GRPO (RL)"),
        help="SFT: Base fine-tuned model. GRPO: Reinforcement learning optimized for reasoning."
    )
    
    # Reload model if selection changes
    if "current_model" not in st.session_state or st.session_state.current_model != model_option:
        st.session_state.current_model = model_option
        # Clear cached model to force reload (optional, but streamlit cache handles args)
        # st.cache_resource.clear() 
    
    uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "jpeg", "png"], key="sidebar_uploader")
    
    st.divider()
    
    st.header("⚙️ Controls")
    if st.button("🗑️ Clear Conversation", use_container_width=True, key="clear_conv_btn_sidebar"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Model Info")
    st.caption(f"Model: Qwen3-VL-8B ({model_option})")
    st.caption("Task: Medical Image Analysis")

    # Display current image in sidebar if uploaded
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image
        st.image(image, caption="Current Image", use_container_width=True)

# Main Area
st.title("🏥 Medical Assistant")
st.markdown("Upload a medical image and ask questions to the AI radiologist.")

# Display image in main area (collapsible) for better visibility
if st.session_state.uploaded_image:
    with st.expander("👁️ View High-Resolution Image", expanded=True):
        st.image(st.session_state.uploaded_image, use_container_width=True)

# Load the model
with st.status(f"🚀 Loading {model_option} Model...", expanded=True) as status:
    st.write("Initializing model architecture...")
    model, tokenizer = load_model(model_option)
    
    if model:
        status.update(label=f"✅ {model_option} Model Loaded Successfully!", state="complete", expanded=False)
    else:
        status.update(label="❌ Model Loading Failed", state="error")
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
                    return_tensors="pt",
                    # Fix for "Mismatch in image token count": Disable truncation or increase max_length
                    truncation=False, 
                    # max_length=4096 # Optional: Set a safe upper limit if needed
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
