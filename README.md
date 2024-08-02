
# TechDaddy: RAG ChatBot for college assistance based on Mixtral-8x7b an Open-source LLM

## Project Overview
TechDaddy is an AI chatbot built using a Retrieval-Augmented Generation (RAG) pipeline to assist college students by providing accurate and contextually relevant answers to their queries. Leveraging LangChain, HuggingFace, and open-source LLMs like Meta-Llama 3.1 and Mistral-7B, TechDaddy is trained on all available data of the college to address various student needs.

## Project Goals and Objectives
- Answer common questions that freshmen might have.
- Provide information about courses and academic structures.
- Offer office timings of faculty members.
- Guide students on where to obtain forms for facilities like the swimming pool.
- Explain the organizational hierarchy within the college.
- Provide roadmaps for various tech fields.

## Target Audience
The primary users of TechDaddy are college students, especially freshmen. Faculty and administrative staff can also benefit from the chatbot by quickly responding to student queries.

## Features and Functionality
- **Q&A for Freshmen**: Clears doubts and answers frequently asked questions by new students.
- **Course Information**: Provides detailed information about courses, including syllabus, structure, and prerequisites.
- **Faculty Office Timings**: Displays the office hours of faculty members.
- **Form Locations**: Guides users on where to obtain various forms, such as those for the swimming pool.
- **College Hierarchy**: Explains the organizational structure of the college.
- **Tech Roadmaps**: Provides guidance and roadmaps for different technology fields.

## Technologies Used
- **Streamlit**: For creating the web interface.
- **LangChain**: To build the RAG pipeline.
- **Groq**: For using the ChatGroq LLM.
- **HuggingFace**: For integrating open-source LLMs.
- **Python**: For overall development.
- **FAISS**: For vector similarity search.
- **dotenv**: For managing environment variables.
- **PyPDFLoader**: For loading PDF documents.

## Expected Outcomes
- **Enhanced Student Experience**: Freshmen will have a smoother transition into college life with easy access to information.
- **Efficient Information Dissemination**: Reduces the workload on administrative staff by automating responses to common queries.
- **Resource Accessibility**: Students will easily find resources and forms they need, improving their overall college experience.

## Setup and Installation
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/aayushkanjani/TechDaddy.git
    cd TechDaddy
    ```

2. **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables:**
   - Create a `.env` file in the project root and add your Groq API key:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```

5. **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## Contact Information
- **Aayush Kanjani**: aayushkanjani6@gmail.com
