"""
Template registry and management.

Provides a central registry for managing different model card templates
and format-specific rendering engines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from ..core.model_card import ModelCard
from ..core.config import CardConfig


class Template(ABC):
    """Abstract base class for model card templates."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def render(self, model_card: ModelCard, output_format: str = "markdown") -> str:
        """Render model card using this template."""
        pass
    
    @abstractmethod
    def enhance_model_card(
        self, 
        model_card: ModelCard, 
        collected_data: Dict[str, Any], 
        config: CardConfig
    ) -> None:
        """Enhance model card with template-specific content."""
        pass
    
    @abstractmethod
    def get_required_sections(self) -> List[str]:
        """Get list of required sections for this template."""
        pass
    
    @abstractmethod
    def get_optional_sections(self) -> List[str]:
        """Get list of optional sections for this template."""
        pass
    
    def validate_model_card(self, model_card: ModelCard) -> List[str]:
        """Validate model card against template requirements."""
        issues = []
        required_sections = self.get_required_sections()
        
        for section in required_sections:
            if not model_card.get_section(section):
                issues.append(f"Missing required section: {section}")
        
        return issues


class TemplateRegistry:
    """Registry for managing model card templates."""
    
    def __init__(self):
        self._templates: Dict[str, Template] = {}
        self.logger = logging.getLogger(__name__)
        
        # Register default templates
        self._register_default_templates()
    
    def _register_default_templates(self) -> None:
        """Register built-in templates."""
        from .huggingface import HuggingFaceTemplate
        from .google import GoogleModelCardTemplate
        from .eu_cra import EUCRATemplate
        
        self.register("huggingface", HuggingFaceTemplate())
        self.register("google", GoogleModelCardTemplate())
        self.register("eu_cra", EUCRATemplate())
    
    def register(self, name: str, template: Template) -> None:
        """Register a template."""
        self._templates[name] = template
        self.logger.info(f"Registered template: {name}")
    
    def get_template(self, name: str) -> Optional[Template]:
        """Get template by name."""
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all registered template names."""
        return list(self._templates.keys())
    
    def get_template_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a template."""
        template = self.get_template(name)
        if template is None:
            return None
        
        return {
            "name": template.name,
            "description": template.description,
            "required_sections": template.get_required_sections(),
            "optional_sections": template.get_optional_sections()
        }
    
    def validate_template_support(self, name: str, model_card: ModelCard) -> Dict[str, Any]:
        """Validate if model card is compatible with template."""
        template = self.get_template(name)
        if template is None:
            return {
                "supported": False,
                "issues": [f"Template not found: {name}"]
            }
        
        issues = template.validate_model_card(model_card)
        
        return {
            "supported": len(issues) == 0,
            "issues": issues,
            "required_sections": template.get_required_sections(),
            "missing_sections": [
                section for section in template.get_required_sections()
                if not model_card.get_section(section)
            ]
        }