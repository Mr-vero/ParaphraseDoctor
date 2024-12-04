# Research Paper Paraphraser

A tool for paraphrasing research papers and academic documents while preserving their structure, citations, equations, and technical terminology.

## Features

- Supports multiple document formats (DOCX, PDF, TXT)
- Preserves document structure and formatting
- Handles special content:
  - Citations and references
  - Mathematical equations
  - Technical terminology
  - Section-specific formatting
- Parallel processing for better performance
- User-friendly web interface
- Progress tracking
- Multiple language model support

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd research-paper-paraphraser
```

2. Install the required dependencies:

```bash
pip install numpy tqdm python-docx pypdf transformers torch gradio
```

## Usage

1. Run the application:

```bash
python main.py
```

2. Access the web interface:
   - The application will start a local server
   - Open your web browser and go to http://localhost:7860
   - A public URL will also be provided for temporary access

3. Using the interface:
   - Upload your document (supported formats: .docx, .pdf, .txt)
   - Configure preservation options:
     - Preserve Citations: Keep citation formatting and references
     - Preserve Equations: Maintain mathematical formulas
   - Click "Submit" to start processing
   - Monitor progress through the progress bar
   - Download the paraphrased document when complete

## How It Works

### Document Processing Pipeline

1. **Document Reading**
   - Parses different document formats
   - Extracts text while maintaining structure
   - Identifies special content (citations, equations, etc.)

2. **Content Analysis**
   - Identifies different sections (abstract, methodology, etc.)
   - Detects technical terms and special notation
   - Preserves document hierarchy

3. **Paraphrasing**
   - Uses specialized models for different content types:
     - Main content: IndoT5-base-paraphrase
     - Technical sections: MT5-small
   - Processes content in parallel for better performance
   - Maintains context and coherence

4. **Structure Preservation**
   - Keeps original formatting
   - Maintains citations and references
   - Preserves equations and technical terms
   - Retains document layout

## Configuration

The tool can be configured through the following parameters in `document_paraphraser.py`:

```python
self.models = {
    'main': {
        'name': "Wikidepia/IndoT5-base-paraphrase",
        'max_length': 128,
        'batch_size': 4
    },
    'technical': {
        'name': "google/mt5-small",
        'max_length': 64,
        'batch_size': 8
    }
}
```

## System Requirements

- Python 3.7 or higher
- Minimum 4GB RAM (8GB recommended)
- CUDA-capable GPU (optional, for better performance)
- Internet connection for initial model download
- Supported operating systems:
  - Windows 10/11
  - macOS 10.15 or later
  - Linux (Ubuntu 18.04 or later)

## Limitations

1. Processing Speed
   - Large documents may take significant time
   - Performance depends on hardware capabilities
   - GPU recommended for faster processing

2. Content Handling
   - Complex mathematical equations might need manual review
   - Some technical terminology may require verification
   - Document formatting might need minor adjustments

3. Model Limitations
   - Initial model loading takes time
   - Internet required for first run
   - Memory usage increases with document size

## Troubleshooting

Common issues and solutions:

1. **Memory Error**
   - Reduce batch size in configuration
   - Close other applications
   - Use smaller document chunks

2. **Slow Processing**
   - Enable GPU if available
   - Reduce document size
   - Adjust batch processing parameters

3. **Format Issues**
   - Ensure document is properly formatted
   - Check for supported file types
   - Verify document encoding

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- HuggingFace Transformers for NLP models
- Gradio for the web interface
- Python-docx for document processing
- PyPDF for PDF handling

## Support

For issues and feature requests:
1. Check existing issues on GitHub
2. Create a new issue with:
   - Clear description
   - Steps to reproduce
   - System information
   - Error messages if any

## Future Improvements

Planned features:
- Additional language model support
- Enhanced formatting preservation
- Batch processing capabilities
- Custom model training options
- API integration support
```

This comprehensive README now includes:
1. Detailed installation and usage instructions
2. Technical explanation of how the tool works
3. Configuration options
4. System requirements and limitations
5. Troubleshooting guide
6. Contributing guidelines
7. Support information
8. Future improvements

Would you like me to expand on any section or add additional information?