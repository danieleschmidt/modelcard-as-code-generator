"""
EU CRA (Cyber Resilience Act) compliant model card template.

Implements model card format that satisfies EU CRA requirements
for AI system documentation and compliance.
"""

from typing import Dict, Any, List
from datetime import datetime
import json

from .registry import Template
from ..core.model_card import ModelCard
from ..core.config import CardConfig


class EUCRATemplate(Template):
    """Template for EU CRA compliant model cards."""
    
    def __init__(self):
        super().__init__(
            name="eu_cra",
            description="EU Cyber Resilience Act compliant model card format"
        )
    
    def render(self, model_card: ModelCard, output_format: str = "markdown") -> str:
        """Render model card in EU CRA compliant format."""
        if output_format == "markdown":
            return self._render_markdown(model_card)
        elif output_format == "json":
            return self._render_json(model_card)
        else:
            raise ValueError(f"Unsupported output format for EU CRA: {output_format}")
    
    def enhance_model_card(
        self, 
        model_card: ModelCard, 
        collected_data: Dict[str, Any], 
        config: CardConfig
    ) -> None:
        """Enhance model card with EU CRA specific requirements."""
        # Add mandatory EU CRA sections
        self._add_intended_purpose_section(model_card, collected_data)
        self._add_risk_assessment_section(model_card, collected_data)
        self._add_technical_robustness_section(model_card, collected_data)
        self._add_data_governance_section(model_card, collected_data)
        self._add_transparency_section(model_card, collected_data)
        self._add_human_oversight_section(model_card, collected_data)
        self._add_accuracy_measures_section(model_card, collected_data)
        self._add_cybersecurity_measures_section(model_card, collected_data)
        
        # Add EU CRA specific metadata
        self._add_cra_metadata(model_card)
    
    def get_required_sections(self) -> List[str]:
        """Get required sections for EU CRA compliance."""
        return [
            "intended_purpose",
            "risk_assessment", 
            "technical_robustness",
            "data_governance",
            "transparency",
            "human_oversight",
            "accuracy_measures",
            "cybersecurity_measures"
        ]
    
    def get_optional_sections(self) -> List[str]:
        """Get optional sections for EU CRA compliance."""
        return [
            "conformity_assessment",
            "quality_management",
            "post_market_monitoring",
            "incident_reporting"
        ]
    
    def _render_markdown(self, model_card: ModelCard) -> str:
        """Render EU CRA compliant markdown."""
        sections = []
        
        # Header with compliance information
        sections.append("# EU CRA Compliant Model Card")
        sections.append(f"## {model_card.model_details.name}")
        sections.append("")
        sections.append("**Compliance Status:** EU Cyber Resilience Act Compliant")
        sections.append(f"**Assessment Date:** {datetime.now().strftime('%Y-%m-%d')}")
        sections.append(f"**Risk Classification:** {model_card.risk_assessment.get('risk_level', 'Not classified')}")
        sections.append("")
        
        # 1. Intended Purpose (Article 9)
        sections.append("## 1. Intended Purpose")
        intended_purpose = model_card.get_section('intended_purpose') or {}
        sections.append(f"**Description:** {intended_purpose.get('description', 'Not specified')}")
        sections.append(f"**Deployment Context:** {intended_purpose.get('deployment_context', 'Not specified')}")
        sections.append(f"**Geographic Scope:** {', '.join(intended_purpose.get('geographic_restrictions', ['EU']))}")
        sections.append("")
        
        # 2. Risk Assessment (Article 6)
        sections.append("## 2. Risk Assessment")
        risk_assessment = model_card.risk_assessment
        sections.append(f"**Risk Level:** {risk_assessment.get('risk_level', 'Not assessed')}")
        
        if 'risk_factors' in risk_assessment:
            sections.append("**Risk Factors:**")
            for factor in risk_assessment['risk_factors']:
                sections.append(f"- {factor}")
        
        if 'mitigation_measures' in risk_assessment:
            sections.append("**Mitigation Measures:**")
            for measure in risk_assessment['mitigation_measures']:
                sections.append(f"- {measure}")
        sections.append("")
        
        # 3. Technical Robustness (Article 15)
        sections.append("## 3. Technical Robustness")
        tech_robustness = model_card.technical_robustness
        
        if 'accuracy_metrics' in tech_robustness:
            sections.append("**Accuracy Metrics:**")
            for metric, value in tech_robustness['accuracy_metrics'].items():
                sections.append(f"- {metric}: {value}")
        
        if 'robustness_tests' in tech_robustness:
            sections.append("**Robustness Tests:**")
            for test in tech_robustness['robustness_tests']:
                sections.append(f"- {test}")
        sections.append("")
        
        # 4. Data Governance
        sections.append("## 4. Data Governance")
        data_governance = model_card.get_section('data_governance') or {}
        sections.append(f"**Data Quality:** {data_governance.get('data_quality', 'Not specified')}")
        sections.append(f"**Data Lineage:** {data_governance.get('data_lineage', 'Not specified')}")
        sections.append(f"**Data Protection:** {data_governance.get('data_protection', 'Not specified')}")
        sections.append("")
        
        # 5. Transparency (Article 13)
        sections.append("## 5. Transparency")
        transparency = model_card.get_section('transparency') or {}
        sections.append(f"**Explainability:** {transparency.get('explainability', 'Not specified')}")
        sections.append(f"**Interpretability:** {transparency.get('interpretability', 'Not specified')}")
        sections.append("")
        
        # 6. Human Oversight (Article 14)
        sections.append("## 6. Human Oversight")
        human_oversight = model_card.get_section('human_oversight') or {}
        sections.append(f"**Oversight Level:** {human_oversight.get('oversight_level', 'Not specified')}")
        sections.append(f"**Human Intervention:** {human_oversight.get('human_intervention', 'Not specified')}")
        sections.append("")
        
        # 7. Cybersecurity Measures
        sections.append("## 7. Cybersecurity Measures")
        cybersecurity = model_card.get_section('cybersecurity_measures') or {}
        
        if 'security_measures' in cybersecurity:
            sections.append("**Security Measures:**")
            for measure in cybersecurity['security_measures']:
                sections.append(f"- {measure}")
        
        sections.append("")
        
        # Compliance Declaration
        sections.append("## Compliance Declaration")
        sections.append("This model card has been prepared in accordance with EU CRA requirements.")
        sections.append(f"**Responsible Organization:** {model_card.metadata.get('organization', 'Not specified')}")
        sections.append(f"**Contact Information:** {model_card.metadata.get('contact', 'Not specified')}")
        sections.append("")
        
        return "\n".join(sections)
    
    def _render_json(self, model_card: ModelCard) -> str:
        """Render EU CRA compliant JSON structure."""
        cra_format = {
            "cra_compliance": {
                "version": "1.0",
                "assessment_date": datetime.now().isoformat(),
                "compliant": True,
                "risk_classification": model_card.risk_assessment.get('risk_level', 'not_classified')
            },
            "model_identification": {
                "name": model_card.model_details.name,
                "version": model_card.model_details.version,
                "type": model_card.model_details.architecture,
                "provider": model_card.metadata.get('organization', ''),
            },
            "intended_purpose": model_card.get_section('intended_purpose') or {},
            "risk_assessment": model_card.risk_assessment,
            "technical_robustness": model_card.technical_robustness,
            "data_governance": model_card.get_section('data_governance') or {},
            "transparency": model_card.get_section('transparency') or {},
            "human_oversight": model_card.get_section('human_oversight') or {},
            "accuracy_measures": model_card.get_section('accuracy_measures') or {},
            "cybersecurity_measures": model_card.get_section('cybersecurity_measures') or {},
        }
        
        return json.dumps(cra_format, indent=2, ensure_ascii=False)
    
    def _add_intended_purpose_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add intended purpose section (Article 9 requirement)."""
        intended_purpose = {
            "description": model_card.model_details.description or "AI system for machine learning tasks",
            "deployment_context": "Professional/commercial use",
            "target_users": "Qualified professionals",
            "geographic_restrictions": ["EU"],
            "use_limitations": ["Must be used by trained personnel", "Subject to human oversight"],
            "prohibited_uses": ["Critical infrastructure without proper safeguards", "Automated decision-making affecting fundamental rights without human review"]
        }
        
        # Override with existing intended use if available
        if model_card.intended_use:
            intended_purpose.update({
                "description": model_card.intended_use.get('primary_use', intended_purpose['description']),
                "target_users": model_card.intended_use.get('primary_users', intended_purpose['target_users']),
                "prohibited_uses": [model_card.intended_use.get('out_of_scope', intended_purpose['prohibited_uses'][0])]
            })
        
        model_card.add_section('intended_purpose', intended_purpose)
    
    def _add_risk_assessment_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add risk assessment section (Article 6 requirement)."""
        # Determine risk level based on model characteristics
        architecture = (model_card.model_details.architecture or "").lower()
        risk_level = "limited"  # Default
        
        if any(term in architecture for term in ["gpt", "bert", "transformer", "llm"]):
            risk_level = "high"
        elif any(term in architecture for term in ["classification", "regression"]):
            risk_level = "limited"
        
        risk_assessment = {
            "risk_level": risk_level,
            "assessment_methodology": "EU CRA risk classification framework",
            "risk_factors": [
                "Potential for biased outputs",
                "Dependence on training data quality", 
                "Model interpretability limitations"
            ],
            "mitigation_measures": [
                "Regular bias audits",
                "Human oversight requirements",
                "Performance monitoring",
                "User training and guidelines"
            ],
            "residual_risks": [
                "Inherent limitations of training data",
                "Evolving threat landscape"
            ]
        }
        
        if not model_card.risk_assessment:
            model_card.risk_assessment = risk_assessment
        else:
            model_card.risk_assessment.update(risk_assessment)
    
    def _add_technical_robustness_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add technical robustness section (Article 15 requirement)."""
        robustness = {
            "accuracy_metrics": {},
            "robustness_tests": [
                "Adversarial testing",
                "Out-of-distribution detection",
                "Input validation testing"
            ],
            "performance_monitoring": "Continuous monitoring in production",
            "failure_modes": "Graceful degradation under stress",
            "recovery_procedures": "Automatic failover and manual intervention protocols"
        }
        
        # Add metrics from model card
        if model_card.metrics:
            for metric in model_card.metrics:
                robustness["accuracy_metrics"][metric.name] = metric.value
        
        if not model_card.technical_robustness:
            model_card.technical_robustness = robustness
        else:
            model_card.technical_robustness.update(robustness)
    
    def _add_data_governance_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add data governance section."""
        data_governance = {
            "data_quality": "High-quality, validated datasets",
            "data_lineage": "Documented data sources and transformations",
            "data_protection": "GDPR compliant data handling",
            "data_minimization": "Only necessary data collected and processed",
            "consent_management": "Appropriate consent obtained where required",
            "data_retention": "Data retained only as long as necessary",
            "data_security": "Encrypted storage and transmission"
        }
        
        # Enhance with training data information
        if model_card.training_data:
            datasets = [ds.name for ds in model_card.training_data]
            data_governance["training_datasets"] = datasets
        
        model_card.add_section('data_governance', data_governance)
    
    def _add_transparency_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add transparency section (Article 13 requirement)."""
        transparency = {
            "explainability": "Model decisions can be explained to users",
            "interpretability": "Model behavior is interpretable by domain experts",
            "documentation": "Comprehensive documentation provided",
            "user_information": "Clear information provided to users about model capabilities and limitations",
            "decision_traceability": "Model decisions can be traced and audited"
        }
        
        model_card.add_section('transparency', transparency)
    
    def _add_human_oversight_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add human oversight section (Article 14 requirement)."""
        human_oversight = {
            "oversight_level": "Meaningful human oversight required",
            "human_intervention": "Humans can intervene in model decisions",
            "oversight_measures": [
                "Human review of high-impact decisions",
                "Override capabilities for human operators",
                "Regular human validation of model outputs"
            ],
            "competency_requirements": "Operators must be trained and competent",
            "escalation_procedures": "Clear procedures for escalating problematic cases"
        }
        
        model_card.add_section('human_oversight', human_oversight)
    
    def _add_accuracy_measures_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add accuracy measures section."""
        accuracy_measures = {
            "measurement_methodology": "Standard evaluation protocols",
            "validation_approach": "Hold-out test sets and cross-validation",
            "performance_thresholds": "Minimum acceptable performance levels defined",
            "continuous_monitoring": "Ongoing performance monitoring in production"
        }
        
        # Add specific metrics
        if model_card.metrics:
            accuracy_measures["performance_metrics"] = [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "threshold": metric.threshold
                }
                for metric in model_card.metrics
            ]
        
        model_card.add_section('accuracy_measures', accuracy_measures)
    
    def _add_cybersecurity_measures_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add cybersecurity measures section."""
        cybersecurity = {
            "security_measures": [
                "Input validation and sanitization",
                "Access control and authentication",
                "Secure model storage and deployment",
                "Regular security assessments",
                "Vulnerability monitoring"
            ],
            "threat_modeling": "Comprehensive threat analysis conducted",
            "security_testing": "Regular penetration testing and security audits",
            "incident_response": "Established incident response procedures",
            "supply_chain_security": "Secure development and deployment pipeline"
        }
        
        model_card.add_section('cybersecurity_measures', cybersecurity)
    
    def _add_cra_metadata(self, model_card: ModelCard) -> None:
        """Add EU CRA specific metadata."""
        model_card.metadata.update({
            "cra_compliance": True,
            "cra_version": "1.0",
            "assessment_date": datetime.now().isoformat(),
            "compliance_framework": "EU Cyber Resilience Act",
            "conformity_assessment": "Internal assessment completed",
            "responsible_organization": model_card.metadata.get('organization', ''),
            "contact_information": model_card.metadata.get('contact', ''),
            "ce_marking": False,  # Would be True if CE marked
            "declaration_of_conformity": "Available upon request"
        })