
Readme:

This a demo program for SAs to be able to download this repo and demo Elastic's Semantic search as well as RAG-based GenAI capabilities
Some prereqs to run this 
  This uses a gated Hugging face model (LLAMA2)
  Follow the instructions to get permissions to use LLAMA2 at huggingface
    
    
    Step 1 To download models from Hugging Face, you must first have a Huggingface account. Sign up at this URL, and then obtain your token at this location.
    
    
    Step 2. Request Llama 2 --> Otherwise use OpenAI if you have access to that model 
       To download and use the Llama 2 model, simply fill out Meta’s form to request access. Please note that utilizing Llama 2 is contingent upon accepting the Meta license agreement.
    After filling out the form, you will receive an email containing a URL that can be used to download the model.

    Step 3. Access to Llama-2 model on Huggingface, submit access form
    Please note that the email you enter in step 2 must match the one you used to create your Hugging Face account in step 1. If they do not match, step 3 will not be successful.

    Then 
     - Set up a VM in GCP or AWS with NVDA cuda support ( make sure you have at least 1TB Disk and 2 NVDA cores and 8 Core CPU)

     ssh into the shell and execute the follwoing command
     git clone https://github.com/sbomma1973/LLAMA2
     Change Directory to LLAMA2
     and execute the command
     %>./env.sh 

     once the libraries are setup edit config.yml file to set up connections to Elastic cloud, 
     %> streamlit run main.py 

     Use the app to test 
    
