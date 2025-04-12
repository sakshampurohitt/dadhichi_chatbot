import streamlit as st
from langchain_ollama import ChatOllama

st.title("🧠 Dadhichi")

st.write("I'm your personal AI fitness and nutrition coach powered by Llama 3. "
         "Upload fitness-related PDFs to enhance my knowledge!👋")

with st.form("llm-form"):
    text = st.text_area("Enter your question or statement:")
    submit = st.form_submit_button("Submit")

def generate_response(input_text):
    model = ChatOllama(
        model="llama3:instruct",  # Ensure this variant supports instructions
        base_url="http://localhost:11434/",
        system=(
            "You are Dadhichi, a highly knowledgeable and supportive fitness and nutrition coach. "
            "You specialize in crafting personalized workout routines, balanced Indian diet plans, "
            "yoga practices for wellness, and providing motivational health tips. "
            "Always give clear, culturally relevant suggestions focused on exercise, fitness, "
            "yoga, hydration, sleep, and nutrition. Avoid medical advice or diagnosing conditions. "
            "Act like a real coach who understands the daily life and diet habits of people in India."
        )
    )

    response = model.invoke(input_text)
    return response.content

# Initialize chat history in session
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

# On form submission
if submit and text:
    with st.spinner("Generating response..."):
        response = generate_response(text)
        st.session_state['chat_history'].append({"user": text, "ollama": response})
        st.write(response)

# Display chat history
st.write("## Chat History")
for chat in reversed(st.session_state['chat_history']):
    st.write(f"**🧑 User**: {chat['user']}")
    st.write(f"**🧠 Assistant**: {chat['ollama']}")
    st.write("---")

st.write("📝 *Note: Dadhichi is powered by Ollama's Llama 3 and LangChain. For fitness guidance only, not medical advice.*")