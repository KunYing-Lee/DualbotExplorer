from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

class DebateBot:
    def __init__(self, engine):
        if engine == 'OpenAI':
            self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        else:
            raise KeyError("Unsupported chat model!")
        self.memory = ConversationBufferMemory(return_messages=True)

    def instruct(self, role, oppo_role, scenario, session_length, starter=False):
        self.role = role
        self.oppo_role = oppo_role
        self.scenario = scenario
        self.session_length = session_length
        self.starter = starter

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self._specify_system_message()),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("""{input}""")
        ])
        
        self.conversation = ConversationChain(
            memory=self.memory, 
            prompt=prompt, 
            llm=self.llm, 
            verbose=False
        )

    def _specify_system_message(self):
        exchange_counts = {
            'Short': 4,
            'Long': 8
        }[self.session_length]
        
        prompt = f"""You are an AI debater participating in a debate about: {self.scenario}
        Role: {self.role['name']}
        - Present clear arguments with supporting evidence
        - Address opponent's points directly
        - Maintain logical consistency
        - Use academic tone but remain accessible
        - Limit responses to 3-5 concise paragraphs
        - You will exchange arguments {exchange_counts} times"""
        
        if self.starter:
            prompt += "\nYou will initiate the debate."
        else:
            prompt += f"\nWait for {self.oppo_role['name']}'s opening statement."
        
        return prompt

class DualDebateBots:
    def __init__(self, engine, role_dict, scenario, session_length):
        self.chatbots = role_dict
        self.engine = engine
        
        for k in role_dict.keys():
            self.chatbots[k].update({'chatbot': DebateBot(engine)})
            
        self.chatbots['role1']['chatbot'].instruct(
            role=self.chatbots['role1'],
            oppo_role=self.chatbots['role2'],
            scenario=scenario,
            session_length=session_length,
            starter=True
        )
        
        self.chatbots['role2']['chatbot'].instruct(
            role=self.chatbots['role2'],
            oppo_role=self.chatbots['role1'],
            scenario=scenario,
            session_length=session_length,
            starter=False
        )
        
        self._reset_conversation_history()

    def step(self):
        output1 = self.chatbots['role1']['chatbot'].conversation.predict(input=self.input1)
        self.conversation_history.append({"bot": self.chatbots['role1']['name'], "text": output1})
        
        output2 = self.chatbots['role2']['chatbot'].conversation.predict(input=output1)
        self.conversation_history.append({"bot": self.chatbots['role2']['name'], "text": output2})
        
        self.input1 = output2
        return output1, output2

    def summary(self, script):
        summary_bot = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
        instruction = """Analyze this debate transcript and create a structured summary:
        1. List Pro's main arguments with supporting points
        2. List Con's main arguments with supporting points
        3. Highlight key areas of disagreement
        4. Note any unresolved questions
        
        Debate transcript: {script}"""
        
        prompt = PromptTemplate(
            input_variables=["script"],
            template=instruction,
        )
        
        summary_chain = LLMChain(llm=summary_bot, prompt=prompt)
        return summary_chain.predict(script=script)

    def _reset_conversation_history(self):
        self.conversation_history = []
        self.input1 = "Begin the debate."