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
                 components : List[str],
                 doc_formatter : bool = False,
                 model : str = None,
                 max_length : int = 200) -> None:
        super().__init__(name='RankPrompt')
        if model:
            template = get_conversation_template(model) 
            self.prompt = '\n'.join([template, *components])
        else: self.prompt = '\n'.join(components)
        self.formatter = DocumentFormatter(max_length)
        self.use_formatter = doc_formatter
    
    def logic(self, **kwargs) -> Any:
        if self.use_formatter:
            texts = kwargs.pop('texts')
            kwargs['documents'] = self.formatter(texts)
        return self.prompt.format(**kwargs)