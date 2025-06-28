import gradio as gr
from qa_chain import get_qa_chain
import traceback

#  Try to load the QA chain (FAISS-based)
try:
    qa_chain = get_qa_chain()
except Exception as e:
    print("❌ Error loading QA chain:")
    traceback.print_exc()
    qa_chain = None

#  Define the chatbot function
def chat_with_doc(question):
    if qa_chain is None:
        return "❌ QA chain failed to load. Please check the logs for details."
    try:
        result = qa_chain.invoke(question)
        return result
    except Exception:
        return f"❌ Error:\n{traceback.format_exc()}"

#  Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 RAG PDF Chatbot (FAISS-powered)")
    
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
            submit_btn = gr.Button("Ask")
        with gr.Column():
            answer = gr.Textbox(label="Answer", interactive=False)

    submit_btn.click(chat_with_doc, inputs=[question], outputs=[answer])

#  Launch app
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
