from research_agent.research_agent import ResearchAgent
from lead_agent.lead_finder import LeadFinderAgent
from scoring_agent.lead_scoring import LeadScoringAgent
from outreach_agent.email_generator import OutreachAgent
from outreach_agent.followup_engine import FollowupAgent

class AgentManager:
    def __init__(self):
        self.research = ResearchAgent()
        self.leads_agent = LeadFinderAgent()
        self.scorer = LeadScoringAgent()
        self.outreach = OutreachAgent()
        self.followup = FollowupAgent()

    def run(self, company_name, website_url):
        print(f"--- Starting Autonomous SDR Workflow for {company_name} ---")
        
        # 1. Deep Research
        company_data = self.research.analyze_company(website_url)
        if not company_data:
            print("⚠️ Warning: Research failed. Using empty research data.")
            company_data = {"website_summary": "N/A", "job_signals": {}, "tech_stack": [], "raw_data": {}}
        
        # 2. Find High-Intent Leads
        leads = self.leads_agent.find_leads(company_name)
        
        results = []
        for lead in leads:
            # 3. Lead Scoring based on Research
            score = self.scorer.score(company_data)
            
            # 4. Multi-channel Outreach Generation
            email = self.outreach.generate_email(lead, company_data)
            linkedin = self.outreach.generate_linkedin_message(lead, company_data)
            
            results.append({
                "lead": lead,
                "score": score,
                "outreach": {
                    "email": email,
                    "linkedin": linkedin
                },
                "status": "Ready for Send"
            })
            
        print("--- Workflow Completed ---")
        return {
            "company": company_name,
            "research": company_data,
            "results": results
        }

clf = AgentManager()
response = clf.run("REDSUN", "https://ovo-redsun.webflow.io/")
print(response)