class LeadScoringAgent:

    def score(self, company_data):

        score = 50

        if "hiring" in company_data["website_summary"]:
            score += 20

        if "sales" in company_data["website_summary"]:
            score += 10

        if "growth" in company_data["website_summary"]:
            score += 10

        return min(score, 100)