import ollama


def make_ollama_request(model: str, question: str, stream: bool = False):
    response = ollama.chat(
        model='llama3.1',
        messages=[{
            'role': 'user',
            'content': question,
        }],
        stream=stream)

    if not stream:
        return response['message']['content']
    else:
        return stream
