from typing import Any, List
from lightchain import Link
from fastchat.model import get_conversation_template

class DocumentFormatter(Link):
    def __init__(self, max_length : int, **kwargs) -> None:
        super().__init__(name='DocumentFormatter', **kwargs)
        self.max_length = max_length
    
    def logic(self, texts : List[str]) -> str:
        texts = [text[:self.max_length] for text in texts]
        return '\n'.join([f'[{i}] {text}' for i, text in enumerate(texts)])

class RankPrompt(Link):
    def __init__(self, 
                 pre : str, 
                 post : str, 
                 model : str, 
                 max_length : int = 200, 
                 context_size: int = 4096) -> None:
        super().__init__(name='RankPrompt')
        self.model = model
        self.context_size = context_size
        template = get_conversation_template(model)
        self.prompt = '\n'.join([template, pre, '{documents}', post])
        self.formatter = DocumentFormatter(max_length)
    
    def logic(self, query : List[str], texts : List[List[str]]) -> Any:
        return [self.prompt.format(query=query, documents=self.formatter(texts), num=len(texts)) for query, texts in zip(query, texts)]