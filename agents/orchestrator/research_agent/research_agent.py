from .company_scraper import CompanyScraper
from .job_signal_detector import JobSignalDetector
from .tech_stack_analyzer import TechStackAnalyzer
from dotenv import load_dotenv
import os

# Load .env from the current directory of the file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

FireCrawler_Api_Key = os.getenv("FireCrawler_Api_Key")

class ResearchAgent:
    def __init__(self):
        self.scraper = CompanyScraper(api_key=FireCrawler_Api_Key)
        self.signal_detector = JobSignalDetector(api_key=FireCrawler_Api_Key)
        self.tech_analyzer = TechStackAnalyzer()

    def analyze_company(self, website_url):
        """
        Perform deep research on a company.
        """
        print(f"Analyzing company: {website_url}")
        
        # 1. Scrape website
        scraping_data = self.scraper.research_company(website_url)
        if scraping_data is None:
            scraping_data = {}
        
        # 2. Detect job signals
        job_signals = self.signal_detector.detect_signals(website_url)
        
        # 3. Analyze tech stack
        tech_stack = self.tech_analyzer.analyze(scraping_data)
        
        # Prepare job signals for the email template
        signals_summary = job_signals.get("summary_of_hiring", "N/A") if isinstance(job_signals, dict) else "N/A"
        
        company_data = {
            "website_summary": scraping_data.get("summary", ""),
            "job_signals": job_signals,
            "signals_summary": signals_summary,
            "tech_stack": tech_stack,
            "raw_data": scraping_data
        }
        
        return company_data
