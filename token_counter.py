def concatenate_inputs(question, conversation_history, context, response):
    ##To determine the relevant tokens excluding the prompt template we take the following elements:
    # question - Question submitted by the user in the chat interface
    # conversation_history - The conversational history passed to the model as context
    # context - The reference answer passed to the model to help generate an adequate response to the question
    
    # The combined_text variable is initialized simply as the question
    combined_text = question

    # We now start adding the conversational history to the initialized "combined_text"
    # The conversation history variable is a list of lists. Each outer list item corresponds to one conversational turn
    for exchange in conversation_history:
        # We iterate through the conversation_history, working through each element as the variable "exchange"
        # Each conversational turn (list item named exchange in this case) consists of two strings (text elements), 
        # the first one, exchange[0] corresponds to the users submitted question, the second element exchange[1] 
        # corresponds to the bots response.
        # For each conversational turn we extend the combined text by a space, the user question, a space, the bot response
        combined_text = combined_text + " " + exchange[0] + " " + exchange[1]

    # We now take the combined text, which at this point consists of the original user question and the elements of the conversational 
    # history, and add another space as well as the context used to answer the question which is taken from the QA database provided
    combined_text += " " + context

    return combined_text

def count_total_tokens(question, conversation_history, context, response):
    import tiktoken

    # Here we load the encoding model (tokenizer) that is used to convert text into tokens prior to being processed by the Large Language Model. 
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    # Here we pass the individual elements to the function we have defined that allows us to merge all textual elements
    # into a single text element named prompt
    prompt = concatenate_inputs(question, conversation_history, context, response)

    # We new generate the encoding from our original text using the previously defined encoder by passing our 
    # prompt text to it
    encoding = encoder.encode(prompt)
    
    # We now determine the length (number of elements) that make up our encoding using pythons integrated len() function
    total_tokens = len(encoding)
    
    # Here we return the number of tokens that we calculated for storage in the database. 
    return total_tokens
