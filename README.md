cloudflared tunnel --url http://localhost:5000
# Political Analysis Report Generator

## Overview
This project is a Flask-based web application that generates detailed political analysis reports for a given politician. The system extracts data, enhances the report using AI, generates charts, and compiles the information into a downloadable PDF or text file.

## Features
- Accepts a politician's name and a date range as input.
- Uses AI (Claude and Groq API) to gather and enhance report data.
- Generates visual charts for key data points.
- Converts the report into a downloadable PDF.
- Allows users to check report generation status.
- Provides real-time logs of the processing steps.

## Requirements
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Flask
- NumPy
- Matplotlib
- Pandas
- ReportLab
- BeautifulSoup4
- Markdown

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Rdhobe/reportgenerationAI.git
   cd political-report-generator
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables for API keys:
   ```bash
   export OPENAI_API_KEY='your_openai_api_key'
   export GROQ_API_KEY='your_groq_api_key'
   export ANTHROPIC_API_KEY='your_claude_api_key'
   ```
   On Windows (PowerShell):
   ```powershell
   $env:OPENAI_API_KEY='your_openai_api_key'
   $env:GROQ_API_KEY='your_groq_api_key'
   $env:ANTHROPIC_API_KEY='your_claude_api_key'
   ```

## Usage
### Running the Application
Start the Flask server:
```bash
python app.py
```
Visit `http://127.0.0.1:5000/` in your browser.

### Generating a Report
1. Enter the politician's name and the date range.
2. Click "Generate Report" to start processing.
3. Check the report status via logs or the status endpoint.
4. Once ready, view or download the report.

### API Endpoints
- `/` - Home page with report generation form.
- `/generate` - Starts the report generation process.
- `/view_report/<politician_name>` - Displays the generated report.
- `/download/<politician_name>` - Downloads the report (PDF or text).
- `/download_pdf/<politician_name>` - Downloads the report as a PDF.
- `/check_status/<politician_name>` - Checks the report generation status.
- `/logs` - Streams real-time logs of report generation.

## File Structure
```
political-report-generator/
│── templates/            # HTML templates
│── static/               # Static files (CSS, JS, images)
│── reports/              # Generated reports directory
│── app.py                # Main Flask application
│── requirements.txt       # Dependencies list
│── README.md             # Project documentation
```

## Future Enhancements
- Add support for more AI models.
- Integrate additional data sources for improved analysis.
- Implement user authentication and history tracking.

## License
This project is licensed under the MIT License.

## Contact
For any issues, please open an issue on the GitHub repository or contact the developer.

