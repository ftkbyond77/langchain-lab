from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

def generate_quiz(material_text):
    prompt_template = """
    Generate 3 quiz questions with answers based on the following content:
    {content}
    """
    prompt = PromptTemplate(input_variables=["content"], template=prompt_template)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    quiz_text = chain.run(content=material_text)
    return quiz_text
