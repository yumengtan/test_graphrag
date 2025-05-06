from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseExtractor(ABC):
    """Abstract base class for content extractors"""
    
    @abstractmethod
    def extract(self, source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Extract text content from source"""
        pass
    
    @abstractmethod
    def validate(self, source: str) -> bool:
        """Validate if source can be processed by this extractor"""
        pass
    
    def get_source_type(self) -> str:
        """Get the type of source this extractor handles"""
        return self.__class__.__name__.replace("Extractor", "").lower()