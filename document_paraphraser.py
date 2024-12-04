import docx
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from docx import Document
import os
from typing import Callable
import gradio as gr
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from docx.shared import Pt, RGBColor
import math

class DocumentParaphraser:
    def __init__(self):
        # Initialize models - using multiple models for different sections
        self.models = {
            'main': {
                'name': "Wikidepia/IndoT5-base-paraphrase",
                'max_length': 128,
                'batch_size': 4
            },
            'technical': {
                'name': "google/mt5-small",  # Better for technical content
                'max_length': 64,
                'batch_size': 8
            }
        }
        
        # Load models
        self.tokenizers = {}
        self.model_instances = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for key, model_info in self.models.items():
            self.tokenizers[key] = AutoTokenizer.from_pretrained(model_info['name'])
            self.model_instances[key] = AutoModelForSeq2SeqLM.from_pretrained(model_info['name']).to(self.device)

        # Special content patterns
        self.patterns = {
            'equation': r'\$.*?\$|\\\[.*?\\\]',
            'citation': r'\[(.*?)\]',
            'reference': r'(?<=[^A-Za-z])\d{4}(?=[^A-Za-z])',
            'technical_term': r'(?<![A-Za-z])[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?![A-Za-z])'
        }

    def _identify_section_type(self, text, style):
        """Identify the type of section for appropriate processing"""
        if style.startswith('Heading'):
            return 'heading'
        elif any(term in text.lower() for term in ['abstract', 'introduction', 'conclusion']):
            return 'main'
        elif any(term in text.lower() for term in ['methodology', 'results', 'discussion', 'experiment']):
            return 'technical'
        return 'main'

    def _extract_special_content(self, text):
        """Extract and preserve special content"""
        special_contents = {
            'equations': re.findall(self.patterns['equation'], text),
            'citations': re.findall(self.patterns['citation'], text),
            'references': re.findall(self.patterns['reference'], text),
            'technical_terms': re.findall(self.patterns['technical_term'], text)
        }
        
        # Replace special content with placeholders
        processed_text = text
        placeholders = {}
        
        for content_type, contents in special_contents.items():
            for idx, content in enumerate(contents):
                placeholder = f"__{content_type}{idx}__"
                placeholders[placeholder] = content
                processed_text = processed_text.replace(content, placeholder)
        
        return processed_text, placeholders

    def _restore_special_content(self, text, placeholders):
        """Restore special content from placeholders"""
        restored_text = text
        for placeholder, content in placeholders.items():
            restored_text = restored_text.replace(placeholder, content)
        return restored_text

    def _parallel_process_chunks(self, chunks, model_key):
        """Process chunks in parallel"""
        model = self.model_instances[model_key]
        tokenizer = self.tokenizers[model_key]
        max_length = self.models[model_key]['max_length']
        batch_size = self.models[model_key]['batch_size']
        
        def process_batch(batch):
            inputs = tokenizer(batch, truncation=True, padding=True,
                             max_length=max_length, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    temperature=0.6,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                    do_sample=False
                )
            
            return tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Process in parallel batches
        results = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                futures.append(executor.submit(process_batch, batch))
            
            for future in futures:
                results.extend(future.result())
        
        return results

    def paraphrase_text(self, text_structure, progress_callback: Callable = None):
        """Enhanced paraphrasing with parallel processing and section-specific handling"""
        paraphrased_structure = []
        total_items = len(text_structure)
        
        for idx, item in enumerate(text_structure):
            if progress_callback:
                progress_callback(f"Processing section {idx + 1}/{total_items}", idx/total_items)
            
            # Skip certain content types
            if item['type'] == 'heading':
                paraphrased_structure.append(item.copy())
                continue
            
            # Identify section type and extract special content
            section_type = self._identify_section_type(item['text'], item['style'])
            text, placeholders = self._extract_special_content(item['text'])
            
            # Split and process
            chunks = self._split_text_carefully(text, self.models[section_type]['max_length'])
            paraphrased_chunks = self._parallel_process_chunks(chunks, section_type)
            
            # Post-process and restore special content
            processed_chunks = [self._post_process_chunk(chunk) for chunk in paraphrased_chunks]
            combined_text = ' '.join(processed_chunks)
            restored_text = self._restore_special_content(combined_text, placeholders)
            
            # Create paraphrased item
            paraphrased_item = item.copy()
            paraphrased_item['text'] = restored_text
            paraphrased_structure.append(paraphrased_item)
        
        if progress_callback:
            progress_callback("Processing complete!", 1.0)
        
        return paraphrased_structure

    def _save_docx(self, structured_text, output_path):
        """Enhanced DOCX saving with better formatting"""
        doc = Document()
        
        # Define styles
        styles = {
            'Heading1': {'size': 16, 'bold': True},
            'Heading2': {'size': 14, 'bold': True},
            'Heading3': {'size': 12, 'bold': True},
            'Normal': {'size': 11, 'bold': False}
        }
        
        for item in structured_text:
            if item['type'] == 'heading':
                heading = doc.add_heading(item['text'], level=int(item['level']))
                heading.alignment = item['alignment']
                
                # Apply heading style
                style_name = f"Heading{item['level']}"
                if style_name in styles:
                    for run in heading.runs:
                        run.font.size = Pt(styles[style_name]['size'])
                        run.font.bold = styles[style_name]['bold']
            else:
                paragraph = doc.add_paragraph()
                run = paragraph.add_run(item['text'])
                paragraph.style = item['style']
                paragraph.alignment = item['alignment']
                
                # Apply normal style
                run.font.size = Pt(styles['Normal']['size'])
                
                # Handle special content formatting
                if '[' in item['text'] and ']' in item['text']:
                    # Format citations differently
                    citations = re.findall(self.patterns['citation'], item['text'])
                    for citation in citations:
                        text_parts = item['text'].split(f'[{citation}]')
                        for i, part in enumerate(text_parts):
                            run = paragraph.add_run(part)
                            if i < len(text_parts) - 1:
                                citation_run = paragraph.add_run(f'[{citation}]')
                                citation_run.font.color.rgb = RGBColor(0, 0, 255)
        
        doc.save(output_path)

    def _split_text_carefully(self, text, max_length):
        """Split text into smaller chunks while preserving sentence and context"""
        import re
        
        # Split by sentences while preserving punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= max_length:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _post_process_chunk(self, text):
        """Clean up and verify paraphrased text"""
        # Remove duplicate spaces
        text = ' '.join(text.split())
        # Ensure proper capitalization
        text = text[0].upper() + text[1:] if text else text
        # Ensure proper punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        return text

    def _combine_chunks_with_citations(self, chunks, citations):
        """Combine paraphrased chunks while restoring citations"""
        text = ' '.join(chunks)
        
        # Restore citations if they exist
        if citations:
            for citation in citations:
                # Add citation at appropriate positions
                text = text.rstrip('.') + f' [{citation}].'
        
        return text

    def save_document(self, text, output_path):
        """Save the paraphrased text to a document"""
        file_extension = os.path.splitext(output_path)[1].lower()
        
        if file_extension == '.docx':
            self._save_docx(text, output_path)
        elif file_extension == '.txt':
            self._save_txt(text, output_path)
        else:
            raise ValueError(f"Unsupported output format: {file_extension}")

    def _save_txt(self, text, output_path):
        """Save text as TXT"""
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text) 

    def process_document(self, input_path, output_path, progress_callback=None):
        """Process entire document with structure preservation"""
        # Read document with structure
        if progress_callback:
            progress_callback("Reading document...", 0.1)
        
        document_structure = self._read_docx(input_path)
        
        # Paraphrase while maintaining structure
        paraphrased_structure = self.paraphrase_text(
            document_structure,
            progress_callback=progress_callback
        )
        
        # Save with preserved structure
        if progress_callback:
            progress_callback("Saving document...", 0.9)
        
        self._save_docx(paraphrased_structure, output_path)
        
        return output_path 