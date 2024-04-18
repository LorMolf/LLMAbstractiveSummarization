
START_TAG = '<|START SUMMARY|>'
END_TAG = '<|END SUMMARY|>' 


INSTRUCTION_PROMPT = """
You're a capable assistant able to generate accurate and brief summaries of input text. 
Summarize user-provided text by retaining the most pertinent information to 
effectively capture the document's essence.

Follow the instructions below:
- craft a detailed, thorough, in-depth, and complex summary while maintaining clarity and conciseness;
- incorporate main ideas and essential information, focusing on critical aspects;
- rely strictly on the provided text without including external information;
- enclose the summary within the tags {} and {};
- answer directly without adding anything more.
"""


DATASETS_SPECIFIC_PROMPTS = {
    'samsum' : "Summarize the content of the following utterances into a concise brief of what people talked about in the conversation. Use the third person to refer to them. Do not use general introductions and be direct",
    'billsum' : 'Summarize the content of the following bill preserving some details',
    'multi_news' : 'Summarize the content of the following newspaper article. If necessary, you can include links and citations to the original articles',
    'EdinburghNLP/xsum' : 'Summarize the content of the following text by drastically reducing its length',
    'cnn_dailymail' : 'Summarize the following newspaper article'
}


USER_PROMPT = '{}:\n'



PROMPTS = {
    'zephyr' : {
        'instruction' : f'<|system|> {INSTRUCTION_PROMPT.format(START_TAG, END_TAG)}',
        'user' : f'<|user|> {USER_PROMPT}',
        'answer' : '<|assistant|>'
    },
    'llama2' : {
        'instruction' : f'<s>[INST] <<SYS>>\n {INSTRUCTION_PROMPT.format(START_TAG, END_TAG)}',
        'user' : f'<</SYS>>\n {USER_PROMPT}',
        'answer' : '[/INST]'
    },
    'gemma' : {
        'instruction' : f'<start_of_turn>user\n{INSTRUCTION_PROMPT.format(START_TAG, END_TAG)}',
        'user' : f'{USER_PROMPT}<end_of_turn>',
        'answer' : '<start_of_turn>model'
    }
}
