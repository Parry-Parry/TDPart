from typing import Any, List
from fastchat.model import get_conversation_template

class DocumentFormatter:
    def __init__(self, max_length : int, **kwargs) -> None:
        super().__init__(name='DocumentFormatter', **kwargs)
        self.max_length = max_length
    
    def logic(self, texts : List[str]) -> str:
        texts = [text[:self.max_length] for text in texts]
        return '\n'.join([f'[{i}] {text}' for i, text in enumerate(texts)])

class RankPrompt:
    def __init__(self, 
                 model : str,
                 components : List[str],
                 doc_formatter : bool = False,
                 max_length : int = 200,
                 rankllm : bool = False) -> None:
        super().__init__(name='RankPrompt')
        template = get_conversation_template(model) 
        if rankllm: template.set_system_message("You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.")
        self.prompt = '\n'.join(components)
        self.template = template
        self.formatter = DocumentFormatter(max_length)
        self.use_formatter = doc_formatter
    
    def logic(self, **kwargs) -> Any:
        if self.use_formatter:
            texts = kwargs.pop('texts')
            kwargs['documents'] = self.formatter(texts)
        input_context = self.prompt.format(**kwargs)
        template = self.template.copy()
        template.append_message(template.roles[0], input_context)
        return template.get_prompt() + " ASSISTANT:"