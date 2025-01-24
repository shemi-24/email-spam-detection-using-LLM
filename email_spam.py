import os
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain     # LLMChain is a chain that combines a language model (in your case, the ChatGroq model) with a prompt template to produce a response. It takes in input data (such as email_subject and email_body),

app = Flask(__name__)

# THIS PROGRAM IS FOR EMAIL CLASSIFICATION

# Load Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Initialize the Groq model via LangChain
def get_llm():
    if not GROQ_API_KEY:
        raise ValueError("Groq API key is not set")
    llm = ChatGroq(api_key=GROQ_API_KEY)             
    return llm


# Define the prompt for categorization
prompt = PromptTemplate(
    input_variables=['email_subject', 'email_body'],
    template=(
        "You are an email categorization assistant. Categorize the email into one of the following categories:\n"
        "- Primary: Important personal or work emails.\n"
        "- Spam: Unwanted or unsolicited messages.\n"
        "- Social: Emails related to social networks, media, or communities.\n"
        "- Promotions: Marketing or promotional emails.\n"
        "- Scheduled: Emails related to events or calendar schedules.\n\n"
        "Email Subject: {email_subject}\n"
        "Email Body: {email_body}\n\n"
        "Provide the category as a single word."
    )
)

# Categorize the email
def categorize_email(email_subject, email_body):
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"email_subject": email_subject, "email_body": email_body})
    return response.strip()


# Define the API route
@app.route('/categorize', methods=['POST'])
def categorize():
    try:
        data = request.get_json()  # Correct method to get JSON from request
        email_subject = data.get("subject", "")
        email_body = data.get("body", "")

        # Validate input
        if not email_subject or not email_body:
            return jsonify({"error": "Subject and body are required"}), 400

        # Get category
        category = categorize_email(email_subject, email_body)
        return jsonify({"category": category})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == '__main__':
    app.run(debug=True)

