class FollowupAgent:
    def __init__(self):
        self.schedule = {
            1: "Intro Email",
            3: "First Follow-up (Value Add)",
            7: "Second Follow-up (Case Study)",
            14: "Final Break-up Email"
        }

    def get_followup(self, day, lead_name):
        """
        Get the follow-up message for a specific day in the sequence.
        """
        messages = {
            3: f"Hi {lead_name}, just wanted to share a case study on how we helped a similar company increase 30% revenue.",
            7: f"Hi {lead_name}, checking if you had a chance to read my last email?",
            14: f"Hi {lead_name}, since I haven't heard back, I'll assume this isn't a priority for now. Feel free to reach out later!"
        }
        return messages.get(day, "No follow-up scheduled for this day.")