# VALORANT Esports Team Manager

This project is a **Streamlit** application designed to manage and display VALORANT esports team compositions, player details, and strategies. It integrates **FAISS**, **Google Generative AI**, **LangChain**, and **Amazon Bedrock** to provide real-time insights based on user input.

## Features

- **Team Overview**: Shows an overview of the team with potential synergies and strategies.
- **Player Details**: Provides detailed information about each player such as preferred agents and justification for selection.
- **Question-Based Responses**: Users can input a question, and the app generates a response using embeddings and conversational AI.

## Tech Stack

- **Streamlit**: Interactive web application.
- **FAISS**: Vector store for efficient similarity search.
- **Google Generative AI (ChatGoogleGenerativeAI)**: Conversational responses based on team composition.
- **LangChain**: Chain management for AI-powered question-answering.
- **Amazon Bedrock**: Model invocation and query classification.
- **JSON**: Data format for storing player data.

## Installation

### Prerequisites

- Python 3.8+
- Streamlit
- FAISS
- LangChain
- Google Generative AI
- Amazon Bedrock
- OpenAI API
- Boto3

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/valorant-team-manager.git
   cd vct_team_manager
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
2. **Add .env File:
   ```bash
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
3. **Run project:
   ```bash
   cd rag_application
   streamlit run streamlitapp.py
4. If getting bedrock InvokeModel operation error use commented "get_query_type" function.

