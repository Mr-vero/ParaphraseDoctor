import gradio as gr
from document_paraphraser import DocumentParaphraser
import tempfile
import os

class ParaphraserGradioApp:
    def __init__(self):
        self.paraphraser = DocumentParaphraser()
        
    def get_usage_info(self):
        """Get current API usage information"""
        stats = self.paraphraser.get_usage_stats()
        return f"""
        Requests Today: {stats['requests_today']}/1500
        Requests Remaining: {stats['requests_remaining']}
        Tokens Used: {stats['tokens_used']}
        """
    
    def get_logs(self):
        """Get current log messages"""
        return "\n\n".join(self.paraphraser.log_messages)
    
    def process_file(self, file, preserve_citations=True, preserve_equations=True, progress=gr.Progress()):
        """Process the uploaded file with additional options"""
        if file is None:
            return "Please upload a file first."
            
        file_extension = os.path.splitext(file.name)[1].lower()
        output_suffix = '.txt' if file_extension not in ['.txt', '.docx'] else file_extension
        temp_output = tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False)
        
        try:
            # Process document with structure preservation
            result = self.paraphraser.process_document(
                file.name,
                temp_output.name,
                progress_callback=lambda msg, val: progress(val, desc=msg)
            )
            # Update usage info and logs after processing
            usage_info = self.get_usage_info()
            logs = self.get_logs()
            return [result, usage_info, logs]
            
        except Exception as e:
            return [f"An error occurred: {str(e)}", self.get_usage_info(), self.get_logs()]

    def create_interface(self):
        """Create enhanced Gradio interface"""
        with gr.Blocks() as interface:
            gr.Markdown("# Research Paper Paraphraser")
            
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload Document",
                        file_types=[".txt", ".docx", ".pdf"]
                    )
                    preserve_citations = gr.Checkbox(
                        label="Preserve Citations",
                        value=True
                    )
                    preserve_equations = gr.Checkbox(
                        label="Preserve Equations",
                        value=True
                    )
                    process_btn = gr.Button("Process Document")
                
                with gr.Column():
                    output_file = gr.File(label="Download Paraphrased Document")
                    usage_info = gr.Textbox(
                        label="API Usage Statistics",
                        value=self.get_usage_info(),
                        interactive=False
                    )
                    log_output = gr.Textbox(
                        label="Processing Logs",
                        value="Logs will appear here during processing...",
                        interactive=False,
                        lines=10
                    )
            
            gr.Markdown("""
            This tool is designed specifically for research papers and academic documents.
            It preserves:
            - Document structure
            - Citations and references
            - Mathematical equations
            - Technical terminology
            - Section formatting
            """)
            
            process_btn.click(
                fn=self.process_file,
                inputs=[file_input, preserve_citations, preserve_equations],
                outputs=[output_file, usage_info, log_output]
            )
            
        return interface

def main():
    app = ParaphraserGradioApp()
    interface = app.create_interface()
    interface.queue().launch(share=True)

if __name__ == "__main__":
    main() 