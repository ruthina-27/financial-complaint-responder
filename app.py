import gradio as gr
from src.rag_pipeline import rag_answer

def chat_interface(question):
    """
    Handles user input, runs the RAG pipeline, and formats the answer and sources for display.
    Args:
        question (str): The user's question.
    Returns:
        Tuple[str, str]: The AI's answer and the formatted sources.
    """
    result = rag_answer(question)
    answer = result['answer']
    sources = result['sources']
    sources_display = '\n\n'.join([
        f"Source {i+1}: {meta['text']}\n(Product: {meta['product']}, ID: {meta['complaint_id']})"
        for i, (chunk, meta) in enumerate(sources[:2])
    ])
    return answer, sources_display

# To implement streaming, you would use a generator function and Gradio's streaming interface.
# Example (not active):
# def chat_interface_stream(question):
#     for token in rag_answer_stream(question):
#         yield token

with gr.Blocks() as demo:
    gr.Markdown("# CrediTrust Complaint-Answering Chatbot")
    with gr.Row():
        question = gr.Textbox(label="Ask a question about financial complaints:")
    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")
    with gr.Row():
        answer = gr.Textbox(label="AI Answer:", interactive=False)
    with gr.Row():
        sources = gr.Textbox(label="Retrieved Sources:", interactive=False)

    submit_btn.click(chat_interface, inputs=question, outputs=[answer, sources])
    clear_btn.click(lambda: ("", ""), None, [answer, sources])

if __name__ == "__main__":
    demo.launch() 