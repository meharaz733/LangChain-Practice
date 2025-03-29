from langchain_core.prompts import PromptTemplate

Prompt_template  = PromptTemplate(
    template = '''
    Please summarize the research paper titled "{paper_name}" in a "{summarize_style}" way.
    Keep it "{summarize_length}", focus on key contributions and results.
    Present it in bullet points.
    ''',
    input_variables = ['paper_name','summarize_style', 'summarize_length'],
    validate_template = True
)

Prompt_template.save('Prompt_template.json')
