# LLAMA2 Demo for Elastic's Semantic Search and GenAI Capabilities

This repository contains a demo program designed for Solutions Architects (SAs) to showcase Elastic's Semantic Search and Retrieval-Augmented Generation (RAG)-based GenAI capabilities. By following the instructions below, you can set up and run the demo using either the LLAMA2 model from Hugging Face or OpenAI's models.

## Prerequisites

- A Hugging Face account
- Permission to access the gated LLAMA2 model on Hugging Face
- A cloud VM in GCP or AWS with NVDA CUDA support (minimum requirements: 1TB disk, 2 NVDA cores, 8-core CPU)
- SSH access to the VM
- Elastic Cloud credentials (for connecting the demo to Elastic)

## Steps to Set Up and Run the Demo

### Step 1: Set Up a Hugging Face Account

1. Sign up for a Hugging Face account at [Hugging Face Signup](https://huggingface.co/join).
2. Once signed up, go to your Hugging Face account settings to obtain your access token at [Hugging Face Access Token](https://huggingface.co/settings/tokens).

### Step 2: Request Access to LLAMA2

To use the LLAMA2 model from Meta, follow these steps:

1. Fill out Meta’s request form to get access to the LLAMA2 model on Hugging Face.
2. Ensure that you agree to the Meta license agreement.
3. After approval, you will receive an email with a URL to download the model. This email should be sent to the same address you used for your Hugging Face account.

If you do not have access to LLAMA2, you can alternatively use OpenAI models if you have access to them.

### Step 3: Set Up Your Cloud Environment

1. Set up a VM in either Google Cloud Platform (GCP) or Amazon Web Services (AWS) with the following specifications:
   - 1TB disk
   - 2 NVDA cores
   - 8-core CPU
2. Ensure that the VM has NVDA CUDA support enabled.

### Step 4: Clone the Repository and Set Up the Environment

1. SSH into your VM.
2. Clone this repository by running the following command:
   ```bash
   git clone https://github.com/sbomma1973/LLAMA2
   cd LLAMA2
   ./env.sh
   streamlit run main.py


