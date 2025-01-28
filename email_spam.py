import os
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain     # LLMChain is a chain that combines a language model (in your case, the ChatGroq model) with a prompt template to produce a response. It takes in input data (such as email_subject and email_body),
from dotenv import load_dotenv

app = Flask(__name__)

# THIS PROGRAM IS FOR EMAIL CLASSIFICATION


# Load environment variables from the .env file
load_dotenv()

# Load Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("GROQ");
print(GROQ_API_KEY);

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
      "- Primary: Important personal or work-related emails. These include account recovery, security alerts, direct communication from known individuals, or any email with high priority that requires immediate attention,if there is any promotion or scheduled messages will be categorizre to primary strictly\n"
      "- Spam: Unwanted or unsolicited messages. Any fake or suspicious offers must be strictly categorized as spam. and fake offers are strictly to store to spam and also there is any fake offer fake surprise or fake scratch and win then strictly that is spam\n"
      "there is only two categorizies primary and spam when important messages and schedules like important messages then categorize:to primary when offers or unrelated messages  categorize to :spam"
      "strictly not show the categorizies:promotions,scheduled,social"
      "if the scheduled and important mails are categorize to primary"
            "Email Subject: {email_subject}\n"
      "Email Body: {email_body}\n\n"
      "Provide the category as a single word (Primary, Spam, Social, Promotions, or Scheduled"
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

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200


# Run the app
if __name__ == '__main__':
    app.run()

