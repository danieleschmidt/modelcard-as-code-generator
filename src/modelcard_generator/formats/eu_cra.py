"""EU CRA (Cyber Resilience Act) compliant model card format."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..core.models import CardConfig, CardFormat, ModelCard


class RiskLevel(Enum):
    """EU AI Act risk levels."""
    MINIMAL = "minimal"
    LIMITED = "limited"
    HIGH = "high"
    UNACCEPTABLE = "unacceptable"


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    PENDING_REVIEW = "pending_review"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class IntendedPurpose:
    """Intended purpose with EU CRA requirements."""
    description: str
    deployment_context: str
    geographic_restrictions: List[str] = field(default_factory=list)
    user_categories: List[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Risk assessment for EU CRA compliance."""
    risk_level: RiskLevel
    mitigation_measures: List[str] = field(default_factory=list)
    impact_assessment: Optional[str] = None
    probability_assessment: Optional[str] = None
    residual_risks: List[str] = field(default_factory=list)


@dataclass
class TechnicalRobustness:
    """Technical robustness measures."""
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    robustness_tests: List[str] = field(default_factory=list)
    cybersecurity_measures: List[str] = field(default_factory=list)
    resilience_measures: List[str] = field(default_factory=list)


@dataclass
class DataGovernance:
    """Data governance and protection measures."""
    data_sources: List[str] = field(default_factory=list)
    data_quality_measures: List[str] = field(default_factory=list)
    privacy_protection: List[str] = field(default_factory=list)
    retention_policy: Optional[str] = None
    consent_mechanism: Optional[str] = None


@dataclass
class TransparencyMeasures:
    """Transparency and explainability measures."""
    explainability_methods: List[str] = field(default_factory=list)
    interpretability_features: List[str] = field(default_factory=list)
    decision_logic: Optional[str] = None
    algorithmic_transparency: Optional[str] = None


@dataclass
class HumanOversight:
    """Human oversight requirements."""
    oversight_measures: List[str] = field(default_factory=list)
    human_in_the_loop: bool = False
    human_on_the_loop: bool = False
    meaningful_human_control: bool = False
    escalation_procedures: List[str] = field(default_factory=list)


class EUCRAModelCard(ModelCard):
    """EU CRA compliant model card implementation."""

    def __init__(self, config: Optional[CardConfig] = None):
        if config is None:
            config = CardConfig(format=CardFormat.EU_CRA)
        super().__init__(config)

        # EU CRA specific sections
        self.intended_purpose = IntendedPurpose(
            description="",
            deployment_context=""
        )
        self.risk_assessment = RiskAssessment(risk_level=RiskLevel.LIMITED)
        self.technical_robustness = TechnicalRobustness()
        self.data_governance = DataGovernance()
        self.transparency_measures = TransparencyMeasures()
        self.human_oversight = HumanOversight()

        # Compliance tracking
        self.compliance_status: Dict[str, ComplianceStatus] = {}
        self.regulatory_requirements: Dict[str, Any] = {}
        self.audit_information: Dict[str, Any] = {}

    def set_intended_purpose(
        self,
        description: str,
        deployment_context: str,
        geographic_restrictions: Optional[List[str]] = None,
        user_categories: Optional[List[str]] = None
    ) -> None:
        """Set intended purpose with CRA requirements."""
        self.intended_purpose = IntendedPurpose(
            description=description,
            deployment_context=deployment_context,
            geographic_restrictions=geographic_restrictions or [],
            user_categories=user_categories or []
        )

        # Update base intended use
        self.intended_use = f"{description}\n\nDeployment Context: {deployment_context}"
        if geographic_restrictions:
            self.intended_use += f"\nGeographic Scope: {', '.join(geographic_restrictions)}"

    def set_risk_assessment(
        self,
        risk_level: Union[RiskLevel, str],
        mitigation_measures: List[str],
        impact_assessment: Optional[str] = None,
        probability_assessment: Optional[str] = None,
        residual_risks: Optional[List[str]] = None
    ) -> None:
        """Set risk assessment information."""
        if isinstance(risk_level, str):
            risk_level = RiskLevel(risk_level)

        self.risk_assessment = RiskAssessment(
            risk_level=risk_level,
            mitigation_measures=mitigation_measures,
            impact_assessment=impact_assessment,
            probability_assessment=probability_assessment,
            residual_risks=residual_risks or []
        )

        # Update base limitations
        if mitigation_measures:
            self.limitations.known_limitations.extend([
                f"Risk Level: {risk_level.value}",
                f"Mitigation Required: {', '.join(mitigation_measures)}"
            ])

    def set_technical_robustness(
        self,
        accuracy_metrics: Dict[str, float],
        robustness_tests: List[str],
        cybersecurity_measures: List[str],
        resilience_measures: Optional[List[str]] = None
    ) -> None:
        """Set technical robustness information."""
        self.technical_robustness = TechnicalRobustness(
            accuracy_metrics=accuracy_metrics,
            robustness_tests=robustness_tests,
            cybersecurity_measures=cybersecurity_measures,
            resilience_measures=resilience_measures or []
        )

        # Add metrics to evaluation results
        for metric_name, value in accuracy_metrics.items():
            self.add_metric(metric_name, value)

    def set_data_governance(
        self,
        data_sources: List[str],
        data_quality_measures: List[str],
        privacy_protection: List[str],
        retention_policy: Optional[str] = None,
        consent_mechanism: Optional[str] = None
    ) -> None:
        """Set data governance information."""
        self.data_governance = DataGovernance(
            data_sources=data_sources,
            data_quality_measures=data_quality_measures,
            privacy_protection=privacy_protection,
            retention_policy=retention_policy,
            consent_mechanism=consent_mechanism
        )

        # Update training details
        self.training_details.training_data.extend(data_sources)

    def set_transparency_measures(
        self,
        explainability_methods: List[str],
        interpretability_features: List[str],
        decision_logic: Optional[str] = None,
        algorithmic_transparency: Optional[str] = None
    ) -> None:
        """Set transparency and explainability measures."""
        self.transparency_measures = TransparencyMeasures(
            explainability_methods=explainability_methods,
            interpretability_features=interpretability_features,
            decision_logic=decision_logic,
            algorithmic_transparency=algorithmic_transparency
        )

    def set_human_oversight(
        self,
        oversight_measures: List[str],
        human_in_the_loop: bool = False,
        human_on_the_loop: bool = False,
        meaningful_human_control: bool = False,
        escalation_procedures: Optional[List[str]] = None
    ) -> None:
        """Set human oversight requirements."""
        self.human_oversight = HumanOversight(
            oversight_measures=oversight_measures,
            human_in_the_loop=human_in_the_loop,
            human_on_the_loop=human_on_the_loop,
            meaningful_human_control=meaningful_human_control,
            escalation_procedures=escalation_procedures or []
        )

    def set_compliance_status(self, requirement: str, status: Union[ComplianceStatus, str]) -> None:
        """Set compliance status for a specific requirement."""
        if isinstance(status, str):
            status = ComplianceStatus(status)

        self.compliance_status[requirement] = status

        # Update base compliance info
        self.set_compliance_info(requirement, status.value, {
            "checked_at": datetime.now().isoformat(),
            "framework": "EU_CRA"
        })

    def add_regulatory_requirement(self, requirement_id: str, details: Dict[str, Any]) -> None:
        """Add regulatory requirement details."""
        self.regulatory_requirements[requirement_id] = {
            **details,
            "added_at": datetime.now().isoformat()
        }

    def set_audit_information(self, audit_data: Dict[str, Any]) -> None:
        """Set audit trail information."""
        self.audit_information = {
            **audit_data,
            "last_updated": datetime.now().isoformat()
        }

    def render(self, format_type: str = "markdown") -> str:
        """Render EU CRA compliant model card."""
        if format_type == "markdown":
            return self._render_eu_cra_markdown()
        elif format_type == "json":
            return self._render_eu_cra_json()
        else:
            return super().render(format_type)

    def _render_eu_cra_markdown(self) -> str:
        """Render as EU CRA compliant markdown."""
        lines = []

        # Title and Declaration
        lines.append(f"# {self.model_details.name}")
        lines.append("\n**EU Cyber Resilience Act (CRA) Compliance Declaration**")

        if self.model_details.description:
            lines.append(f"\n{self.model_details.description}")

        # Executive Summary
        lines.append("\n## Executive Summary")
        lines.append(f"This model card provides comprehensive documentation for {self.model_details.name} ")
        lines.append("in compliance with the EU Cyber Resilience Act (CRA) requirements for AI systems.")

        # Model Identification
        lines.append("\n## Model Identification")
        lines.append(f"- **Model Name:** {self.model_details.name}")
        if self.model_details.version:
            lines.append(f"- **Version:** {self.model_details.version}")
        if self.model_details.authors:
            lines.append(f"- **Responsible Party:** {', '.join(self.model_details.authors)}")
        lines.append(f"- **Last Updated:** {self.updated_at.strftime('%Y-%m-%d')}")

        # Intended Purpose (CRA Article 13)
        lines.append("\n## Intended Purpose")
        lines.append(f"**Description:** {self.intended_purpose.description}")
        lines.append(f"\n**Deployment Context:** {self.intended_purpose.deployment_context}")

        if self.intended_purpose.geographic_restrictions:
            lines.append(f"\n**Geographic Restrictions:** {', '.join(self.intended_purpose.geographic_restrictions)}")

        if self.intended_purpose.user_categories:
            lines.append(f"\n**Intended User Categories:** {', '.join(self.intended_purpose.user_categories)}")

        # Risk Assessment (CRA Annex I)
        lines.append("\n## Risk Assessment")
        lines.append(f"**Risk Level:** {self.risk_assessment.risk_level.value.upper()}")

        if self.risk_assessment.impact_assessment:
            lines.append(f"\n**Impact Assessment:** {self.risk_assessment.impact_assessment}")

        if self.risk_assessment.probability_assessment:
            lines.append(f"\n**Probability Assessment:** {self.risk_assessment.probability_assessment}")

        if self.risk_assessment.mitigation_measures:
            lines.append("\n**Mitigation Measures:**")
            for measure in self.risk_assessment.mitigation_measures:
                lines.append(f"- {measure}")

        if self.risk_assessment.residual_risks:
            lines.append("\n**Residual Risks:**")
            for risk in self.risk_assessment.residual_risks:
                lines.append(f"- {risk}")

        # Technical Robustness (CRA Article 11)
        lines.append("\n## Technical Robustness")

        if self.technical_robustness.accuracy_metrics:
            lines.append("\n### Performance Metrics")
            for metric, value in self.technical_robustness.accuracy_metrics.items():
                lines.append(f"- **{metric}:** {value:.4f}")

        if self.technical_robustness.robustness_tests:
            lines.append("\n### Robustness Testing")
            for test in self.technical_robustness.robustness_tests:
                lines.append(f"- {test}")

        if self.technical_robustness.cybersecurity_measures:
            lines.append("\n### Cybersecurity Measures")
            for measure in self.technical_robustness.cybersecurity_measures:
                lines.append(f"- {measure}")

        if self.technical_robustness.resilience_measures:
            lines.append("\n### Resilience Measures")
            for measure in self.technical_robustness.resilience_measures:
                lines.append(f"- {measure}")

        # Data Governance (GDPR Compliance)
        lines.append("\n## Data Governance")

        if self.data_governance.data_sources:
            lines.append("\n### Data Sources")
            for source in self.data_governance.data_sources:
                lines.append(f"- {source}")

        if self.data_governance.data_quality_measures:
            lines.append("\n### Data Quality Measures")
            for measure in self.data_governance.data_quality_measures:
                lines.append(f"- {measure}")

        if self.data_governance.privacy_protection:
            lines.append("\n### Privacy Protection")
            for protection in self.data_governance.privacy_protection:
                lines.append(f"- {protection}")

        if self.data_governance.retention_policy:
            lines.append(f"\n**Data Retention Policy:** {self.data_governance.retention_policy}")

        if self.data_governance.consent_mechanism:
            lines.append(f"\n**Consent Mechanism:** {self.data_governance.consent_mechanism}")

        # Transparency and Explainability (EU AI Act Article 13)
        lines.append("\n## Transparency and Explainability")

        if self.transparency_measures.explainability_methods:
            lines.append("\n### Explainability Methods")
            for method in self.transparency_measures.explainability_methods:
                lines.append(f"- {method}")

        if self.transparency_measures.interpretability_features:
            lines.append("\n### Interpretability Features")
            for feature in self.transparency_measures.interpretability_features:
                lines.append(f"- {feature}")

        if self.transparency_measures.decision_logic:
            lines.append(f"\n**Decision Logic:** {self.transparency_measures.decision_logic}")

        if self.transparency_measures.algorithmic_transparency:
            lines.append(f"\n**Algorithmic Transparency:** {self.transparency_measures.algorithmic_transparency}")

        # Human Oversight (EU AI Act Article 14)
        lines.append("\n## Human Oversight")

        if self.human_oversight.oversight_measures:
            lines.append("\n### Oversight Measures")
            for measure in self.human_oversight.oversight_measures:
                lines.append(f"- {measure}")

        lines.append(f"\n**Human-in-the-loop:** {'Yes' if self.human_oversight.human_in_the_loop else 'No'}")
        lines.append(f"**Human-on-the-loop:** {'Yes' if self.human_oversight.human_on_the_loop else 'No'}")
        lines.append(f"**Meaningful Human Control:** {'Yes' if self.human_oversight.meaningful_human_control else 'No'}")

        if self.human_oversight.escalation_procedures:
            lines.append("\n### Escalation Procedures")
            for procedure in self.human_oversight.escalation_procedures:
                lines.append(f"- {procedure}")

        # Limitations and Constraints
        if self.limitations.known_limitations:
            lines.append("\n## Limitations and Constraints")
            for limitation in self.limitations.known_limitations:
                lines.append(f"- {limitation}")

        if self.limitations.sensitive_use_cases:
            lines.append("\n### Prohibited Use Cases")
            for use_case in self.limitations.sensitive_use_cases:
                lines.append(f"- {use_case}")

        # Compliance Status
        if self.compliance_status:
            lines.append("\n## Compliance Status")
            for requirement, status in self.compliance_status.items():
                status_icon = {
                    ComplianceStatus.COMPLIANT: "✅",
                    ComplianceStatus.PENDING_REVIEW: "⏳",
                    ComplianceStatus.NON_COMPLIANT: "❌",
                    ComplianceStatus.NOT_APPLICABLE: "➖"
                }.get(status, "❓")

                lines.append(f"- {status_icon} **{requirement}:** {status.value.replace('_', ' ').title()}")

        # Regulatory Requirements
        if self.regulatory_requirements:
            lines.append("\n## Regulatory Requirements")
            for req_id, details in self.regulatory_requirements.items():
                lines.append(f"\n### {req_id}")
                for key, value in details.items():
                    if key != "added_at":
                        lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")

        # Audit Information
        if self.audit_information:
            lines.append("\n## Audit Information")
            for key, value in self.audit_information.items():
                if key != "last_updated":
                    lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")

        # Declaration of Conformity
        lines.append("\n## Declaration of Conformity")
        lines.append("This model card constitutes a declaration that the AI system described herein ")
        lines.append("has been assessed for compliance with applicable EU regulations including the ")
        lines.append("Cyber Resilience Act (CRA) and EU AI Act.")

        lines.append("\n**Document Version:** 1.0")
        lines.append(f"**Last Review Date:** {datetime.now().strftime('%Y-%m-%d')}")
        lines.append(f"**Next Review Due:** {(datetime.now().replace(year=datetime.now().year + 1)).strftime('%Y-%m-%d')}")

        return "\n".join(lines)

    def _render_eu_cra_json(self) -> str:
        """Render as EU CRA JSON format."""
        data = {
            "schema_version": "EU_CRA_1.0",
            "model_identification": {
                "name": self.model_details.name,
                "version": self.model_details.version,
                "authors": self.model_details.authors,
                "description": self.model_details.description,
                "last_updated": self.updated_at.isoformat()
            },
            "intended_purpose": {
                "description": self.intended_purpose.description,
                "deployment_context": self.intended_purpose.deployment_context,
                "geographic_restrictions": self.intended_purpose.geographic_restrictions,
                "user_categories": self.intended_purpose.user_categories
            },
            "risk_assessment": {
                "risk_level": self.risk_assessment.risk_level.value,
                "mitigation_measures": self.risk_assessment.mitigation_measures,
                "impact_assessment": self.risk_assessment.impact_assessment,
                "probability_assessment": self.risk_assessment.probability_assessment,
                "residual_risks": self.risk_assessment.residual_risks
            },
            "technical_robustness": {
                "accuracy_metrics": self.technical_robustness.accuracy_metrics,
                "robustness_tests": self.technical_robustness.robustness_tests,
                "cybersecurity_measures": self.technical_robustness.cybersecurity_measures,
                "resilience_measures": self.technical_robustness.resilience_measures
            },
            "data_governance": {
                "data_sources": self.data_governance.data_sources,
                "data_quality_measures": self.data_governance.data_quality_measures,
                "privacy_protection": self.data_governance.privacy_protection,
                "retention_policy": self.data_governance.retention_policy,
                "consent_mechanism": self.data_governance.consent_mechanism
            },
            "transparency_measures": {
                "explainability_methods": self.transparency_measures.explainability_methods,
                "interpretability_features": self.transparency_measures.interpretability_features,
                "decision_logic": self.transparency_measures.decision_logic,
                "algorithmic_transparency": self.transparency_measures.algorithmic_transparency
            },
            "human_oversight": {
                "oversight_measures": self.human_oversight.oversight_measures,
                "human_in_the_loop": self.human_oversight.human_in_the_loop,
                "human_on_the_loop": self.human_oversight.human_on_the_loop,
                "meaningful_human_control": self.human_oversight.meaningful_human_control,
                "escalation_procedures": self.human_oversight.escalation_procedures
            },
            "compliance_status": {
                requirement: status.value
                for requirement, status in self.compliance_status.items()
            },
            "regulatory_requirements": self.regulatory_requirements,
            "audit_information": self.audit_information,
            "limitations": {
                "known_limitations": self.limitations.known_limitations,
                "sensitive_use_cases": self.limitations.sensitive_use_cases,
                "out_of_scope_uses": self.limitations.out_of_scope_uses,
                "recommendations": self.limitations.recommendations
            }
        }

        # Remove None values
        data = self._remove_none_values(data)

        return json.dumps(data, indent=2)

    def validate_cra_compliance(self) -> Dict[str, Any]:
        """Validate compliance with EU CRA requirements."""
        issues = []
        warnings = []

        # Check intended purpose (Article 13)
        if not self.intended_purpose.description:
            issues.append("Missing intended purpose description (CRA Article 13)")

        if not self.intended_purpose.deployment_context:
            issues.append("Missing deployment context (CRA Article 13)")

        # Check risk assessment (Annex I)
        if self.risk_assessment.risk_level == RiskLevel.HIGH and not self.risk_assessment.mitigation_measures:
            issues.append("High-risk AI system requires mitigation measures (CRA Annex I)")

        if self.risk_assessment.risk_level == RiskLevel.UNACCEPTABLE:
            issues.append("Unacceptable risk level - deployment prohibited (EU AI Act Article 5)")

        # Check technical robustness (Article 11)
        if not self.technical_robustness.accuracy_metrics:
            warnings.append("No accuracy metrics provided for technical robustness assessment")

        if not self.technical_robustness.cybersecurity_measures:
            issues.append("Missing cybersecurity measures (CRA Article 11)")

        # Check data governance (GDPR compliance)
        if not self.data_governance.privacy_protection:
            issues.append("Missing privacy protection measures (GDPR compliance)")

        # Check transparency (EU AI Act Article 13)
        if (self.risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.LIMITED] and
            not self.transparency_measures.explainability_methods):
            warnings.append("High/Limited risk systems should provide explainability methods")

        # Check human oversight (EU AI Act Article 14)
        if (self.risk_assessment.risk_level == RiskLevel.HIGH and
            not any([self.human_oversight.human_in_the_loop,
                    self.human_oversight.human_on_the_loop,
                    self.human_oversight.meaningful_human_control])):
            issues.append("High-risk AI systems require human oversight (EU AI Act Article 14)")

        # Check prohibited use cases
        if not self.limitations.sensitive_use_cases:
            warnings.append("No prohibited use cases documented")

        compliance_score = max(0, 1 - (len(issues) * 0.2 + len(warnings) * 0.1))

        return {
            "is_compliant": len(issues) == 0,
            "compliance_score": compliance_score,
            "issues": issues,
            "warnings": warnings,
            "risk_level": self.risk_assessment.risk_level.value,
            "framework": "EU_CRA"
        }

    def _remove_none_values(self, data: Any) -> Any:
        """Recursively remove None values from data structure."""
        if isinstance(data, dict):
            return {k: self._remove_none_values(v) for k, v in data.items() if v is not None}
        elif isinstance(data, list):
            return [self._remove_none_values(item) for item in data if item is not None]
        else:
            return data


def create_eu_cra_template() -> EUCRAModelCard:
    """Create a template EU CRA model card with common sections."""
    card = EUCRAModelCard()

    # Set basic intended purpose template
    card.set_intended_purpose(
        description="[Describe the specific purpose and functionality of the AI system]",
        deployment_context="[Specify the operational environment and conditions]",
        geographic_restrictions=["EU"],
        user_categories=["[Specify intended user categories]"]
    )

    # Set risk assessment template
    card.set_risk_assessment(
        risk_level=RiskLevel.LIMITED,
        mitigation_measures=[
            "[Specify risk mitigation measures]",
            "[Include technical safeguards]"
        ],
        impact_assessment="[Assess potential impact of failures]",
        residual_risks=["[Document remaining risks after mitigation]"]
    )

    # Set technical robustness template
    card.set_technical_robustness(
        accuracy_metrics={},  # To be filled with actual metrics
        robustness_tests=[
            "Adversarial testing",
            "Out-of-distribution testing",
            "Stress testing"
        ],
        cybersecurity_measures=[
            "Input validation",
            "Output sanitization",
            "Secure communication protocols"
        ]
    )

    # Set data governance template
    card.set_data_governance(
        data_sources=["[Specify training data sources]"],
        data_quality_measures=[
            "Data validation procedures",
            "Quality control checks",
            "Bias detection mechanisms"
        ],
        privacy_protection=[
            "Data anonymization",
            "Access controls",
            "Encryption in transit and at rest"
        ],
        retention_policy="[Specify data retention policy]",
        consent_mechanism="[Describe consent collection process]"
    )

    # Set transparency measures template
    card.set_transparency_measures(
        explainability_methods=["[Specify XAI methods used]"],
        interpretability_features=["[List interpretability features]"],
        decision_logic="[Describe decision-making process]"
    )

    # Set human oversight template
    card.set_human_oversight(
        oversight_measures=["[Specify human oversight mechanisms]"],
        human_in_the_loop=False,
        human_on_the_loop=True,
        meaningful_human_control=True,
        escalation_procedures=["[Define escalation procedures]"]
    )

    return card
