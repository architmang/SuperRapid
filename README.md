# How to Run Code
1. Initialize the virtual env
   > conda create --name redhen python=3.9<br>
   > conda activate redhen<br>
   > conda install --file requirements.txt 
2. Run the annotator code
   > python3 annotator.py
3. Upload image [person.png](person.png) or use Webcam
4. Copy the JSON from [QA.txt](QA.txt) and submit
   
# Gradio App
Gradio is leveraged to construct an intuitive web-based interface where users can seamlessly upload images and input question-target pairs. 
Through Gradio's simple API, the interface components such as image upload and text input are effortlessly defined and connected to the underlying functionality.



# Current Approach
BLIP (Blind Language Image Pre-training) models, such as the one used in the code (Salesforce/blip-vqa-base), are designed to understand both images and text simultaneously. This makes them well-suited for tasks involving multimodal inputs, such as answering questions about images. <br><br>
Semantic similarity scores computed using models like Sentence Transformers provide a quantitative measure of how similar two pieces of text are in meaning. By comparing the generated answer from the BLIP model with the expected value, we can determine if the answer aligns semantically with the target value, enhancing the accuracy of the response.
