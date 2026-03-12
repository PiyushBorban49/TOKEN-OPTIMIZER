class TechStackAnalyzer:
    def analyze(self, scraping_data):
        """
        Detect tech stack from website signals.
        """
        # Basic keyword detection
        summary = scraping_data.get("summary", "").lower()
        tech = []
        
        if "react" in summary: tech.append("React")
        if "next.js" in summary: tech.append("Next.js")
        if "aws" in summary: tech.append("AWS")
        
        return tech if tech else ["Unknown Tech Stack"]
