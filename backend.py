from langchain.chains.question_answering import load_qa_chain
import http.server
import json
import os
from urllib.parse import parse_qs
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Old import (deprecated)
# from langchain.vectorstores import FAISS

# New import
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Preloaded text (use your actual four pages content here)
preloaded_text = """
Vidyavardhaka College of Engineering (VVCE) – A Leading Institution in Engineering and Management Education
Vidyavardhaka College of Engineering (VVCE) – A Leading Institution in Engineering and Management Education

Introduction
Vidyavardhaka Sangha (VVS) has been synonymous with quality education in Mysuru since its establishment in 1949 by visionaries such as Late Sahukar Channaiah and Late K. Puttaswamy (former Minister of Karnataka). VVS manages ten educational institutions, from nursery schools to engineering colleges, under the leadership of Sri Gundappa Gowda, Sri Shivalingappa B (Vice President), Sri P. Vishwanath (Secretary), and Sri Shrishaila Ramannavar (Treasurer).
Founded in 1997, VVCE is a prestigious engineering college affiliated with Visvesvaraya Technological University (VTU), Belagavi. VVCE is AICTE and UGC approved and recognized by the Government of Karnataka. With NAAC “A” grade accreditation and NBA accreditation for six undergraduate programs, VVCE offers seven UG programs, three PG programs, and nine research centers for PhD studies, enrolling over 3,200 students. Dr. B. Sadashive Gowda, the Principal, leads VVCE with a rich background in academia and industry.

Vision, Mission & Core Values
Vision:VVCE aims to be a leading institution in engineering and management education, preparing individuals to make significant contributions to society.
Mission:
To offer an exceptional teaching-learning environment with competent staff and excellent infrastructure.
To instill professional ethics, leadership, communication, and entrepreneurial skills to meet societal needs.
To foster innovation through research and development.
To strengthen industry-institute interactions for knowledge sharing.
Core Values:
Integrity: Ensuring truthfulness, fairness, and transparency in all academic, administrative, and professional activities.
Excellence: Striving for academic excellence by setting high standards for faculty and student performance.
Teamwork: Encouraging a collaborative environment for continuous institutional improvement.
Professionalism: Nurturing quality education and upholding a commitment to ethical and professional standards.
Research & Collaboration: Providing opportunities for continuous improvement in teaching, research, and collaboration.

Campus Facilities and Sustainability
Hostel for Non-Day-Scholars
VVCE provides hostels for non-local students, contributing to a diverse and vibrant campus life. These hostels are managed sustainably, with waste segregation, water conservation practices, and solar-powered heating.
Transportation: Buses for Industrial Visits
VVCE offers buses for industrial visits, reducing the carbon footprint associated with student and staff travel by encouraging group transportation.
Sports Facilities
Sports Complex: Houses facilities for various indoor games, fostering a culture of health and physical activity.
Outdoor Grounds: Includes a basketball court and a large ground for sports like cricket and football. This ground also serves as a venue for concerts and college events, promoting a green, open space for community gatherings.
Infrastructure and Safety
Backup Electric Supply: Ensures uninterrupted learning experiences, even during power outages, with sustainable energy sources integrated into backup systems.
24/7 Security Surveillance: Security personnel and CCTV surveillance maintain campus safety, with eco-friendly LED lighting around the campus to reduce energy consumption.
Library
VVCE’s library, spanning 890 sq. m., is located on the ground floor of E Block. The library includes:
Reference Section, Lending Section, and Digital Library.
A unique laptop borrowing service for students, promoting digital accessibility and reducing paper usage.
Eco-friendly lighting and ventilation to reduce energy consumption.
Banking and Stationary
ATM Machine & Bank: Located on campus for student convenience, these facilities reduce travel needs.
Stationary & Reprography Shop: Offers necessary academic supplies and reprography services, reducing the need for external travel.
Health and Wellness
Counseling Center: Provides mental health support for students, creating a healthier campus environment.
Health Center: Offers basic medical services on campus, reducing the environmental impact of frequent hospital trips.
Gym: Equipped with modern fitness equipment, encouraging physical wellness among students and staff.
On-Campus Dining
Cafe Coffee Day: Located on campus, this cafe reduces the need for students to leave campus for refreshments.
Canteen: Offers a range of healthy meal options, reducing the carbon footprint associated with frequent off-campus dining.
Water and Waste Management
RO Purified Drinking Water: Provided at various points across campus, reducing the need for plastic bottled water.
Waste Segregation Practices: Ensures proper disposal of waste and supports recycling efforts.
Accessible Infrastructure
Wheelchair Accessible Ramps & Toilets for the Physically Challenged: VVCE is an inclusive campus with ramps and accessible toilets.
On-Campus Wheelchairs: Available to students and visitors in need of assistance.
Lecture Capture Solutions
VVCE has equipped classrooms with UpGrad Campus Lecture Capture Solutions so students can revisit lectures as needed, promoting flexibility in learning and reducing physical attendance when possible.

Sustainability Efforts
VVCE’s campus design and management focus on sustainability:
Green Campus: Lush greenery across campus enhances air quality and provides natural cooling.
Energy Conservation: Use of LED lights, energy-efficient equipment, and solar power where possible.
Waste Management and Recycling: Campus-wide initiatives to reduce waste, promote recycling, and minimize environmental impact.

Nearby Landmarks and Pathways
VVCE is strategically located along High Tension Double Road and PG Road. Key landmarks nearby include:
Karnataka Sahitya Parishath Road: A popular route leading to various cultural centers.
Industrial Areas for Visits: Accessible by VVCE buses, fostering practical learning experiences for students.
Recent Events
Two-day Workshop on “Fundamentals of Python Programming”: Aimed at improving technical skills, the workshop included discussions on sustainable programming practices to reduce computational costs.
Technical Talk on “Recent Trends in Tools and Programming Technologies in Computer Science”: This talk focused on emerging technologies, highlighting tools that promote efficiency, which indirectly contributes to sustainability by reducing excessive use of computing power.
Fun Fact
The recent workshops at VVCE incorporate discussions on reducing CO₂ emissions through efficient coding practices and sustainable technology choices. By implementing greener programming practices, students contribute to reducing the institution’s environmental footprint.



...
"""

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate vector store from the chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to generate the conversational chain
def get_conversational_chain():
    prompt_template = """
    Based on the provided document content, please answer the following question. Please provide answers related to sustainability, prioritizing environmentally friendly solutions, eco-conscious practices, and sustainable development based on the content in the provided text. Reference principles of green living, renewable resources, energy efficiency, waste reduction, and conservation wherever applicable. Frame responses in a way that aligns with eco-friendly values and promote sustainable choices.
    
    End the answer with eco-centric phrases such as:
    'Have a greener day!'
    'Let's make the world more sustainable, one step at a time!'
    'Together, we can build a greener future.'

    Document content: {context}
    
    Question: {question}
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt, document_variable_name="context")
    return chain

# Function to handle user input and fetch chatbot response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # Retrieve the conversational chain
    chain = get_conversational_chain()
    
    # Generate response, setting 'context' to the document content
    response = chain(
        {"input_documents": docs, "context": " ".join([doc.page_content for doc in docs]), "question": user_question},
        return_only_outputs=True
    )
    
    return response["output_text"]

# Create the handler class to handle POST requests
class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        # Handle CORS pre-flight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')  # Allow any domain
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        # Set the response type to JSON
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')  # Allow any domain
        self.end_headers()

        # Get the length of the incoming data
        content_length = int(self.headers['Content-Length'])
        
        # Read the data
        post_data = self.rfile.read(content_length)

        # Parse the JSON data
        data = json.loads(post_data)

        # Get the user's message from the POST data
        user_message = data.get('message')

        # Get the response from the chatbot
        answer = user_input(user_message)

        # Send the chatbot response as JSON
        response = json.dumps({"response": answer})
        self.wfile.write(response.encode())


def run(server_class=http.server.HTTPServer, handler_class=SimpleHTTPRequestHandler, port=5001):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    # Split the content into chunks and generate vector store (do this once)
    text_chunks = get_text_chunks(preloaded_text)
    get_vector_store(text_chunks)

    # Start the server
    run()

#Run Backend in separate terminal with command python backend.py
#Open frontend in browswer
