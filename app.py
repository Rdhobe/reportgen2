from flask import Flask, render_template, request, redirect
import os
from openai import OpenAI
from groq import Groq
import logging
import queue
from datetime import datetime
import anthropic
import matplotlib.pyplot as plt
import matplotlib
import re , json
from markdown_pdf import MarkdownPdf, Section
import matplotlib
from dotenv import load_dotenv
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
# Initialize Flask app
app = Flask(__name__)
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup a queue for storing log messages
log_queue = queue.Queue()
# Initialize the MarkdownPdf object
pdf = MarkdownPdf()
system_prompt ="""You are an advanced data processing and visualization AI. Given structured data on media trends and caste-wise support, format the output as a JSON object containing different types of charts or graphs. The output should adhere to the following structure:

- The JSON should be an array containing objects with:  
  - A 'type' field (e.g., 'chart' or 'graph').  
  - A 'credentials' object with:  
    - 'title': The title of the visualization.  
    - 'x-axis': Label for the x-axis.  
    - 'y-axis': Label for the y-axis.  
    - 'data': An array of key-value pairs representing the data points.  

Example Output:
[
  {
    'type': 'chart',
    'credentials': {
      'title': 'Media Mentions Trend',
      'x-axis': 'Quarter',
      'y-axis': 'Total Mentions',
      'data': [
        {'quarter': 'Q1 2023', 'mentions': 42460},
        {'quarter': 'Q2 2023', 'mentions': 39750},
        {'quarter': 'Q3 2023', 'mentions': 45680},
        {'quarter': 'Q4 2023', 'mentions': 57920},
        {'quarter': 'Q1 2024', 'mentions': 89570},
        {'quarter': 'Q2 2024', 'mentions': 156840},
        {'quarter': 'Q3 2024', 'mentions': 72350},
        {'quarter': 'Q4 2024', 'mentions': 45780},
        {'quarter': 'Q1 2025', 'mentions': 32450}
      ]
    }
  },
  {
    'type': 'graph',
    'credentials': {
      'title': 'Caste-wise Support',
      'x-axis': 'Caste Group',
      'y-axis': 'Support Percentage',
      'data': [
        {'caste': 'Reddy', 'support': 73},
        {'caste': 'Kapu', 'support': 29},
        {'caste': 'BC', 'support': 40},
        {'caste': 'SC', 'support': 50},
        {'caste': 'ST', 'support': 54},
        {'caste': 'Muslims', 'support': 61},
        {'caste': 'Other OBCs', 'support': 37}
      ]
    }
  }
]

- Ensure the output remains structured, maintaining proper nesting and labeling.
- Use meaningful chart titles and axis labels to reflect the dataset.
- The 'data' field should accurately map values to respective categories.
- If additional data categories exist, generate new objects within the JSON array, ensuring consistency in formatting."""
# pdf.add_section(Section(markdown_content))
custom_css = """
table {
    width: 100%;
    border-collapse: collapse;
}
th, td {
    border: 1px solid black;
    padding: 10px;
    text-align: center;
}
th {
    background-color: #f2f2f2;
    font-weight: bold;
}
"""
matplotlib.use('Agg')  # Use non-interactive backend
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        
    def emit(self, record):
        log_entry = self.format(record)
        self.log_queue.put(log_entry)

# Add queue handler to logger
queue_handler = QueueHandler(log_queue)
queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(queue_handler)

def upload_pdf_to_drive(file_path, folder_id=None):
    """Uploads a PDF file to Google Drive using a service account."""
    
    # Authenticate using Service Account
    # Get service account JSON file path from environment variable
    service_account_file = {
  "type": os.getenv("TYPE"),
  "project_id": os.getenv("PROJECT_ID"),
  "private_key_id": os.getenv("PROJECT_KEY_ID"),
  "private_key": os.getenv("PRIVETE_KEY"),
  "client_email": os.getenv("CLIENT_EMAIL"),
  "client_id": os.getenv("CLIENT_ID"),
  "auth_uri": os.getenv("AUTH_URI"),
  "token_uri": os.getenv("TOCKEN_URI"),
  "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER"),
  "client_x509_cert_url": os.getenv("CLIENT_CERT_URI"),
  "universe_domain": os.getenv("UNIVERSAL_DOMAIN")
}
    if not service_account_file:
        raise ValueError("SERVICE_ACCOUNT_JSON environment variable is not set")
    
    # Authenticate using Service Account
    gauth = GoogleAuth()
    gauth.settings['client_config_file'] = service_account_file
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
        service_account_file,
        ["https://www.googleapis.com/auth/drive"]
    )
    gauth.Authorize()
    drive = GoogleDrive(gauth)

    # Create file metadata
    file_metadata = {'title': file_path.split('/')[-1]}
    if folder_id:
        file_metadata['parents'] = [{'id': folder_id}]

    # Upload file
    file_drive = drive.CreateFile(file_metadata)
    file_drive.SetContentFile(file_path)
    file_drive.Upload()

    # print(f"File uploaded successfully: {file_drive['title']}, ID: {file_drive['id']}")
    return f"https://drive.google.com/file/d/{file_drive['id']}"
def generate_report(name, start_date, end_date):
    """Generate report using Claude and Groq APIs with MarkdownPdf integration"""
    try:
        logger.info(f"Starting report generation for politician: {name}")
        logger.info(f"Tracking period: {start_date} to {end_date}")
        
        # Initialize API clients
        logger.info("Initializing API clients...")
        openai_api_key=os.getenv("OPENAI_API_KEY")
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        groq_api_key=os.getenv("GROQ_API_KEY")
        if not anthropic_api_key or not openai_api_key or not groq_api_key:
            raise ValueError("apikey environment variable is not set")

        openai_client = OpenAI(api_key=openai_api_key)
        groq_client = Groq(api_key=groq_api_key)
        claude_client = anthropic.Anthropic(
            api_key=anthropic_api_key,
        )
        
        # First API call to Claude
        logger.info("Making API call to Claude for detailed report...")
        message = claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=20000,
            temperature=1,  
            messages=[
                {"role": "user", "content": f"Please generate a single comprehensive report that meets the following criteria of {name} politician .do not generate a false or dummy report if possible. Report Format: Individual Profiles and Historical Data:For each politician in the attached documents, create a separate section (or document) that includes:Historical tracking data starting from {start_date} until {end_date}.Quantitative details and numerical trends related to mentions, polling, social media, and overall sentiment.Sentiment Analysis:Identify major specific examples of mentions related to each politician, including contextual details.Calculate and report the annual percentage shift in sentiment, and outline key reasons or areas where improvement is observed.If available, include a detailed caste-wise sentiment analysis as this could be a game changer.Include insights on general public perception, issues (such as polling trends, social trends, etc.), and factors affecting potential winnability, with a comparative analysis where applicable.Social Media and Link Integration:Provide a platform-wise breakdown of social media mentions and links (e.g., Twitter, Facebook, Instagram, etc.).Ensure that all links are clickable and can be opened in a browser for further reference.Include links to additional online resources or relevant articles that support the sentiment analysis and trends observed.Summary and Recommendations:Based on the detailed analysis, suggest which politician appears to have the most favorable trends or potential.Conclude with a summary that encapsulates the key findings and areas for improvement.Final Compilation:Combine all the above information into one single, cohesive document that is well-organized and easy to navigate.Please ensure that your final output is structured with clear headings and subheadings, contains all the numerical data and examples, and integrates clickable links for further verification of social media mentions and related sources."}
            ]
        )
        detailed_report = message.content
        logger.info("Successfully received detailed report from Claude")
        
        # Extracting text from the Detailed_report
        def extract_text(report):
            return "\n\n".join(block.text for block in report)
        
        extracted_text = extract_text(detailed_report)
        # detailed_report_file = f"detailed_report_of_{name}.txt"
        # with open(detailed_report_file, "w", encoding="utf-8") as f:
        #     f.write(extracted_text)
        logger.info(f"Saved detailed report")
        
        # Second API call check the report
        logger.info("Making API call to OpenAI for checking the report...")
        chat_completions = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "system",
            "content": "You are a highly experienced political analyst and close policy advisor.you are responsible for checking the report for the accuracy and improvising the report by doing all neccesry changes like correcting the false information and filling the correct information aslo add a detailed comparision report between all important personalitites in coalition and opposition "
            },
            {
            "role": "user",
            "content":extracted_text  
            }
        ]
        )
        # Second API call to Groq for final refinement
        logger.info("Making API call to Groq for final report compilation...")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role":"system","content":system_prompt},
                {"role": "user", "content": chat_completions.choices[0].message.content}
            ],
            model="deepseek-r1-distill-llama-70b",
        )
        final_report = chat_completion.choices[0].message.content
        logger.info("Successfully received final report from Groq")
        
        # Extract JSON for charts
        import re
        pattern = re.compile(r'(\[.*\])', re.DOTALL)
        logger.info("Extracting JSON array from final report...")
        match = pattern.search(final_report)
        
        if not match:
            logger.error("No JSON array found in the final report")
            raise ValueError("Unable to extract JSON data")
        
        json_text = match.group(1)
        # print("Extracted JSON:")
        # print(json_text)
        
        # Parse the JSON string with error handling
        try:
            charts = json.loads(json_text)
            # print("Parsed Charts:", charts)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise

        # Initialize MarkdownPdf
        pdf = MarkdownPdf()
        
        # Custom CSS for better formatting
        custom_css = """
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0 auto;
            max-width: 800px;
            padding: 20px;
        }
        h1 { color: #333; border-bottom: 2px solid #333; }
        h2 { color: #444; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 15px auto;
        }
        """
        
        # Prepare Markdown content
        markdown_content = f"""# Political Profile Report: {name}

## Report Overview
**Analysis Period:** {start_date} to {end_date}

## Detailed Report Content
{extracted_text}

## Charts and Visualizations
"""
        
        # Generate charts and save images
        chart_markdown = ""
        for chart in charts:
            try:
                chart_type = chart.get("type", "")
                credentials = chart.get("credentials", {})
                title = credentials.get("title", "Unnamed Chart")
                x_axis = credentials.get("x-axis", "X-Axis")
                y_axis = credentials.get("y-axis", "Y-Axis")
                data_points = credentials.get("data", [])
                
                # Create the plot
                plt.figure(figsize=(10, 6))
                plt.title(title)
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)
                
                # Universal handling for different chart types
                if chart_type == "chart" or chart_type == "graph":
                    # Extract keys dynamically
                    x_key = list(data_points[0].keys())[0]
                    y_key = list(data_points[0].keys())[1]
                    
                    x_values = [point.get(x_key, "") for point in data_points]
                    y_values = [point.get(y_key, 0) for point in data_points]
                    
                    # Plot based on chart type
                    if chart_type == "chart":
                        plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue')
                        plt.grid(True)
                    else:  # graph
                        plt.bar(x_values, y_values, color='skyblue')
                        plt.grid(axis='y')
                    
                    plt.xticks(rotation=45, ha='right')
                
                # Save the plot
                plot_filename = f"{title.replace(' ', '_')}.png"
                plt.tight_layout()
                plt.savefig(plot_filename)
                plt.close()

                # Add to markdown content
                chart_markdown += f"\n### {title}\n![{title}]({plot_filename})\n"

            except Exception as chart_error:
                logger.error(f"Error processing chart {title}: {chart_error}")
                continue

        # Combine markdown content
        markdown_content += chart_markdown

        # Add sections to PDF
        pdf.add_section(Section(markdown_content), user_css=custom_css)
        
        # Save PDF
        pdf_filename = f"detailed_report_of_{name}.pdf"
        pdf.save(pdf_filename)
        link= upload_pdf_to_drive(pdf_filename, "1bTBK34syNDWAiQjnGa1ubTQXnh7OcZXw")
        logger.info(f"Saved PDF report to {pdf_filename}")
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)
            png_files = glob.glob("*.png")
            for png_file in png_files:
                os.remove(png_file) 

        logger.info(f"Deleted PDF report from local directory")
        return pdf_filename , link
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
@app.route('/')
def index():
    # Set default start date to January 1, 2023
    default_start_date = "2023-01-01"
    # Set default end date to today's date
    default_end_date = datetime.now().strftime("%Y-%m-%d")
    
    return render_template('index.html', 
                           default_start_date=default_start_date,
                           default_end_date=default_end_date)

@app.route('/generate', methods=['POST'])
def generate():
    name = request.form['politician_name']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    # Start report generation in a background thread
   
    pdf_filename,drive_link =generate_report(name, start_date, end_date)
    if pdf_filename:
        return redirect(drive_link)
    else:
        return "Report generation failed. Please try again later."




if __name__ == '__main__':
    # Create a directory for reports if it doesn't exist
    if not os.path.exists('reports'):
        os.makedirs('reports')
        
    app.run(port=8080)


