from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class PeerReviewAuthorBot:
    def __init__(self, engine):
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
        self.responses = []
        
    def instruct(self, title, abstract, vectorstore):
        """Initialize author bot with paper content"""
        self.title = title
        self.abstract = abstract
        self.vectorstore = vectorstore
        
        self.system_prompt = f"""You are the author of "{title}". Your task:
        - Defend your methodology and results
        - Provide additional evidence from the paper
        - Address reviewer concerns professionally
        - Reference specific sections
        - Maintain academic tone
        
        Abstract: {abstract}"""

    def respond_to_question(self, question):
        prompt = PromptTemplate(
            input_variables=["question"],
            template=f"""{self.system_prompt}
            
            Reviewer Question: {{question}}
            Author Response:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run({"question": question})
        self.responses.append((question, response))
        return response
        
    def summarize_responses(self):
        summary_prompt = """Analyze these author responses:
        {responses}
        
        Identify:
        1. 3 strongest defenses
        2. 2 potential weaknesses
        3. Key evidence provided"""
        
        return LLMChain(llm=self.llm, 
                      prompt=PromptTemplate.from_template(summary_prompt)
                     ).run({"responses": self.responses})

class PeerReviewReviewerBot:
    def __init__(self, engine):
        self.llm = ChatOpenAI(model_name="gpt-4o")
        self.critiques = []
        
    def instruct(self, title, abstract, focus_areas, rigor_level):
        """Initialize reviewer bot with evaluation parameters"""
        # Convert rigor_level (0-100) to temperature (0.1-0.9)
        self.llm.temperature = 0.9 - (rigor_level/100 * 0.8)
        
        self.system_prompt = f"""As peer reviewer of "{title}":
        - Focus: {', '.join(focus_areas)}
        - Rigor: {rigor_level}/100
        - Ask probing questions
        - Identify methodological flaws
        - Verify statistical validity
        - Check ethical compliance
        
        Abstract: {abstract}"""

    def generate_question(self):
        prompt = PromptTemplate(
            input_variables=[],
            template=f"""{self.system_prompt}
            
            Generate a critical review question:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        question = chain.run({})
        self.critiques.append(question)
        return question
        
    def generate_verdict(self):
        verdict_template = """Based on these critiques:
        {critiques}
        
        Final recommendations:
        1. Accept/Revise/Reject
        2. Required revisions
        3. Suggested improvements"""
        
        return LLMChain(llm=self.llm,
                      prompt=PromptTemplate.from_template(verdict_template)
                     ).run({"critiques": self.critiques})
        
    def summarize_critiques(self):
        summary_prompt = """Analyze review critiques:
        {critiques}
        
        Extract:
        1. Methodological concerns
        2. Statistical issues
        3. Ethical considerations
        4. Suggested improvements"""
        
        return LLMChain(llm=self.llm,
                      prompt=PromptTemplate.from_template(summary_prompt)
                     ).run({"critiques": self.critiques})