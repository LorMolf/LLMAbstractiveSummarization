
START_TAG = '[START SUMMARY]'
END_TAG = '[END SUMMARY]' 


INSTRUCTION_PROMPT = """
You are an assistant capable of producing faithful and concise summaries of an input document. 
Read the text provided by the user and summarize it by keeping the most useful information which
you consider to best sum up the content of the document. Be as concise as needed and do not
include information out of the input text's domain. Include the summary within the tags
{} and {} right after the work '{}'\n"""


DATASETS_SPECIFIC_PROMPTS = {
    'samsum' : 'Summarize the content of the following utterances into a concise brief of what people talked about in the conversation. Use third person to refer to them',
    'billsum' : 'Sumarize the content of the following bill preserving some details',
    'multi_news' : 'Summarize the content of the following newspaper article. If necessary, yuo can include links and citation to the original articles',
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
        'instruction' : f'# Instruct: {INSTRUCTION_PROMPT.format(START_TAG, END_TAG, "")}',
        'user' : f'# Summarize:\n {USER_PROMPT}',
        'answer' : '# Output: '
    }
}