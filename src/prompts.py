
START_TAG = '<|START SUMMARY|>'
END_TAG = '<|END SUMMARY|>' 


INSTRUCTION_PROMPT = """
You are an assistant capable of producing faithful and concise summaries of an input document. 
Read the text provided by the user and summarize it by keeping the most useful information that
you consider to best sum up the document's content. Be as concise as needed, and do not
include information from the input text's domain. Include the summary within the tags
{} and {} right after the word '{}'\n"""


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
        'instruction' : f'<|system|> {INSTRUCTION_PROMPT.format(START_TAG, END_TAG, "<|assistant|>")}',
        'user' : f'<|user|> {USER_PROMPT}',
        'answer' : '<|assistant|>'
    },
    'llama2' : {
        'instruction' : f'# Assistant:\n {INSTRUCTION_PROMPT.format(START_TAG, END_TAG, "# Summary:")}',
        'user' : f'# Summarize:\n {USER_PROMPT}',
        'answer' : '# Answer:'
    },
    'phi2' : {
        'instruction' : f'<|system|> {INSTRUCTION_PROMPT.format(START_TAG, END_TAG, "")}',
        'user' : f'<|user|> {USER_PROMPT}',
        'answer' : '<|assistant|>'
    }
}
