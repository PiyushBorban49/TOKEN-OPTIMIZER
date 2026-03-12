import os
from groq import Groq

class OutreachAgent:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        if self.api_key:
            self.client = Groq(api_key=self.api_key)

    def generate_email(self, lead, company_data):
        """
        Generate a hyper-personalized cold email.
        """
        print(f"Generating email for: {lead['name']}")
        
        if not self.client:
            # Fallback for mock/no-key runs
            tech_name = company_data.get('tech_stack', ['Sales'])[0] if company_data.get('tech_stack') else 'Sales'
            job_name = company_data.get('job_signals', {}).get('summary_of_hiring', 'new roles') if isinstance(company_data.get('job_signals'), dict) else 'new roles'
            
            return f"Subject: Helping {lead['company']} scale their {tech_name} team\n\nHi {lead['name']},\n\nI noticed {lead['company']} is currently hiring for {job_name}. Most companies at this stage struggle with pipeline growth...\n\nBest,\nAI SDR"

        prompt = f"""
        Write a hyper-personalized cold email.
        Lead: {lead['name']}
        Role: {lead['role']}
        Company: {lead['company']}
        Company Context:
        - Website Summary: {company_data.get('website_summary')}
        - Hiring Signals: {', '.join(company_data.get('job_signals', []))}
        - Tech Stack: {', '.join(company_data.get('tech_stack', []))}
        Goal: Book a 15-min discovery call.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating email: {str(e)}"

    def generate_linkedin_message(self, lead, company_data):
        """
        Generate a short LinkedIn connection request.
        """
        return f"Hi {lead['name']}, noticed you're scaling the team at {lead['company']}. Would love to connect and share how we're help similar firms optimize their sales stack."