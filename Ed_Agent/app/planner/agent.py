from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

def generate_study_plan(materials):
    """
    materials: list of Material objects
    """
    prompt_template = """
    User has the following study materials: {titles}.
    Please suggest a study plan with priority order based on importance and length.
    """
    titles = ", ".join([m.title for m in materials])
    prompt = PromptTemplate(input_variables=["titles"], template=prompt_template)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    plan = chain.run(titles=titles)
    return plan
