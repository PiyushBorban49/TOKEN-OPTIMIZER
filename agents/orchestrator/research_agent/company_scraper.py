import os
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List, Optional

class CompanyIntel(BaseModel):
    summary: str = Field(description="A 2-sentence summary of what the company does.")
    target_customers: List[str] = Field(description="Who is this company's ideal customer profile?")
    recent_news: Optional[str] = Field(description="Any mentions of funding, hiring, or new launches.")
    tech_stack_hints: List[str] = Field(description="Keywords like 'AWS', 'Salesforce', 'React', 'Python'.")
    tone_of_voice: str = Field(description="Is the brand professional, playful, or technical?")

class CompanyScraper:
    def __init__(self, api_key: str):
        self.app = FirecrawlApp(api_key=api_key)

    def research_company(self, url: str):
        print(f"Deep-researching: {url}...")
        
        try:
            result = self.app.extract(
                urls=[url],
                schema=CompanyIntel.model_json_schema(),
                prompt="Extract detailed company information. Look for tech stack in footer or career pages, and news in the blog/press sections."
            )
            
            data = result.get('data') if isinstance(result, dict) else result.data
            
            # Convert Pydantic model to dict if necessary
            if hasattr(data, 'model_dump'):
                return data.model_dump()
            return data
            
        except Exception as e:
            print(f"❌ Error researching {url}: {e}")
            return None
