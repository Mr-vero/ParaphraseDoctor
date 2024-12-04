import gradio as gr
from document_paraphraser import DocumentParaphraser
import tempfile
import os

class ParaphraserGradioApp:
    def __init__(self):
        self.paraphraser = DocumentParaphraser()
        
    def process_file(self, file, preserve_citations=True, preserve_equations=True, progress=gr.Progress()):
        """Process the uploaded file with additional options"""
        if file is None:
            return "Please upload a file first."
            
        file_extension = os.path.splitext(file.name)[1].lower()
        output_suffix = '.txt' if file_extension not in ['.txt', '.docx'] else file_extension
        temp_output = tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False)
        
        try:
            # Process document with structure preservation
            self.paraphraser.process_document(
                file.name,
                temp_output.name,
                progress_callback=lambda msg, val: progress(val, desc=msg)
            )
            
            return temp_output.name
            
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def create_interface(self):
        """Create enhanced Gradio interface"""
        interface = gr.Interface(
            fn=self.process_file,
            inputs=[
                gr.File(label="Upload Document", file_types=[".txt", ".docx", ".pdf"]),
                gr.Checkbox(label="Preserve Citations", value=True),
                gr.Checkbox(label="Preserve Equations", value=True)
            ],
            outputs=[
                gr.File(label="Download Paraphrased Document")
            ],
            title="Research Paper Paraphraser",
            description="Upload a research paper to paraphrase its content while preserving academic structure.",
            article="""
            This tool is designed specifically for research papers and academic documents.
            It preserves:
            - Document structure
            - Citations and references
            - Mathematical equations
            - Technical terminology
            - Section formatting
            """,
            cache_examples=False,
            api_name="paraphrase"
        )
        return interface

def main():
    app = ParaphraserGradioApp()
    interface = app.create_interface()
    interface.queue().launch(share=True)

if __name__ == "__main__":
    main() 