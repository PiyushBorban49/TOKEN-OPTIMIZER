import os
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List, Optional

# Define what a 'Job Opening' looks like for the AI
class JobOpening(BaseModel):
    title: str = Field(description="The exact job title (e.g. Senior Backend Engineer)")
    department: str = Field(description="Department: Engineering, Sales, Marketing, etc.")
    is_senior: bool = Field(description="True if it's a leadership/senior role")
    link: str = Field(description="The direct link to the job posting")

# The final output for the company
class HiringSignals(BaseModel):
    is_hiring: bool = Field(description="Is the company actively hiring right now?")
    priority_level: str = Field(description="High, Medium, or Low based on number of roles")
    open_roles: List[JobOpening] = Field(description="List of relevant jobs found")
    summary_of_hiring: str = Field(description="A 1-sentence summary of their hiring strategy.")

class JobSignalDetector:
    def __init__(self, api_key: str):
        self.app = FirecrawlApp(api_key=api_key)

    def detect_signals(self, company_url: str) -> HiringSignals:
        print(f"Searching for hiring signals at {company_url}...")
        
        try:
            # STEP 1: Use MAP to find the careers page link
            # This is 10x faster than crawling the whole site
            map_result = self.app.map(url=company_url, search="careers")
            
            # Handle MapData object or dictionary
            if isinstance(map_result, dict):
                links = map_result.get('links', [])
            else:
                links = getattr(map_result, 'links', [])
            
            # Find the best candidate for a careers page
            careers_url = next((l for l in links if "careers" in l or "jobs" in l), company_url)
            
            # STEP 2: Use EXTRACT on that specific page
            # We use /* to allow the AI to look at the job listings
            result = self.app.extract(
                urls=[f"{careers_url}"],
                schema=HiringSignals.model_json_schema(),
                prompt="Identify if they are hiring. Extract specific job titles and departments. If they have a 'View All Jobs' page, use that."
            )
            
            # Handle response object
            data = result.get('data') if isinstance(result, dict) else result.data
            
            # Convert Pydantic model to dict if necessary
            if hasattr(data, 'model_dump'):
                return data.model_dump()
            return data

        except Exception as e:
            print(f"❌ Error detecting jobs: {e}")
            return None