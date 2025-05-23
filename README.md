# Assamese Conversational AI — TinyLlama Fine-Tuning
This repository contains all the code, data processing scripts, and training configuration files used to build an Assamese Conversational AI model via two-stage fine-tuning of the TinyLlama-1.1B-Chat model.

**Entire fine-tuning pipeline was executed in Kaggle Notebooks. Code is included in train.ipynb for full reproducibility.**

The final fine-tuned model adapters are available on Hugging Face:  
[Assamese-TinyLlama-Chat](https://huggingface.co/themid6t/assamese-tinyllama-chat)  
[Assamese-TinyLlama-Base](https://huggingface.co/themid6t/assamese-tinyllama-base)

## Project Summary
Over 15 million people speak Assamese, yet it's vastly underrepresented in modern NLP. This project brings Assamese into the LLM space by fine-tuning a small language model (TinyLlama) using a two-stage pipeline:

### Stage 1: Language Adaptation
Trained on 12,000 Assamese sentences from CC-100 and Assamese Wikipedia  
Goal: Make the model understand Assamese grammar, vocabulary, and structure

### Stage 2: Conversational Specialization

Fine-tuned on 2,000 user-assistant dialogues (translated from OpenAssistant/oasst1)  
Goal: Enable fluent, context-aware chat in Assamese

**Both stages were trained using:**

LoRA (Low-Rank Adaptation) to update only ~1.13% of the model parameters

4-bit quantization for memory efficiency (training done on a single NVIDIA T4 GPU)

### Model Highlights
**Model:** TinyLlama-1.1B-Chat-v1.0  
**Fine-tuning type:** LoRA + 4-bit quantization
**Total trainable parameters:** ~12.6M
**Training hardware:** Single NVIDIA Tesla T4 (15GB VRAM)