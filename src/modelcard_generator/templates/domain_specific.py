"""Domain-specific model card templates for specialized use cases."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from ..core.models import CardConfig, CardFormat, ModelCard
from .library import Template

logger = logging.getLogger(__name__)


@dataclass
class ClinicalValidation:
    """Clinical validation metrics for medical AI models."""
    sensitivity: float
    specificity: float
    auc: float
    ppv: Optional[float] = None  # Positive Predictive Value
    npv: Optional[float] = None  # Negative Predictive Value
    cohort_size: Optional[int] = None
    validation_sites: Optional[List[str]] = None
    follow_up_period: Optional[str] = None


@dataclass
class RegulatoryInfo:
    """Regulatory approval information."""
    status: str  # pending, approved, denied, withdrawn
    agency: str  # FDA, EMA, etc.
    pathway: Optional[str] = None  # 510(k), De Novo, PMA, etc.
    submission_date: Optional[datetime] = None
    approval_date: Optional[datetime] = None
    indication: Optional[str] = None


@dataclass
class FairnessMetrics:
    """Fairness metrics for financial AI models."""
    demographic_parity: float
    equal_opportunity: float
    equalized_odds: Optional[float] = None
    calibration: Optional[float] = None
    individual_fairness: Optional[float] = None
    protected_attributes: List[str] = None


class MedicalAITemplate(Template):
    """Template for medical AI model cards with clinical validation and regulatory compliance."""

    def __init__(self):
        super().__init__(
            name="medical_ai",
            required_sections=[
                "intended_use",
                "clinical_validation",
                "regulatory_status",
                "contraindications",
                "risk_assessment",
                "data_privacy"
            ]
        )

    def create(self,
               model_name: str,
               intended_use: str,
               clinical_validation: ClinicalValidation,
               regulatory_info: RegulatoryInfo,
               contraindications: List[str],
               intended_population: str,
               data_sources: List[str],
               **kwargs) -> ModelCard:
        """Create medical AI model card.
        
        Args:
            model_name: Name of the medical AI model
            intended_use: Clinical intended use statement
            clinical_validation: Clinical validation metrics
            regulatory_info: Regulatory approval information
            contraindications: List of contraindications
            intended_population: Target patient population
            data_sources: List of data sources used for training
            **kwargs: Additional configuration options
            
        Returns:
            Medical AI model card
        """
        config = CardConfig(
            format=CardFormat.HUGGINGFACE,
            include_ethical_considerations=True,
            regulatory_standard="fda_medical_device"
        )

        card = ModelCard(config)

        # Model details
        card.model_details.name = model_name
        card.model_details.description = f"Medical AI model for {intended_use}"
        card.model_details.version = kwargs.get("version", "1.0.0")
        card.model_details.license = kwargs.get("license", "Proprietary")
        card.model_details.tags = ["medical-ai", "healthcare", "clinical-decision-support"]

        # Add medical-specific sections
        self._add_intended_use_section(card, intended_use, intended_population)
        self._add_clinical_validation_section(card, clinical_validation)
        self._add_regulatory_section(card, regulatory_info)
        self._add_contraindications_section(card, contraindications)
        self._add_risk_assessment_section(card, clinical_validation)
        self._add_data_privacy_section(card, data_sources)

        # Training details specific to medical AI
        card.training_details.training_data = data_sources
        card.training_details.preprocessing = kwargs.get(
            "preprocessing",
            "Clinical data preprocessing including deidentification, normalization, and quality checks"
        )

        # Add clinical metrics
        self._add_clinical_metrics(card, clinical_validation)

        # Metadata
        card.metadata.update({
            "domain": "medical",
            "regulatory_pathway": regulatory_info.pathway,
            "target_population": intended_population,
            "clinical_validation_completed": True,
            "hipaa_compliant": True
        })

        return card

    def _add_intended_use_section(self, card: ModelCard, intended_use: str, population: str) -> None:
        """Add intended use section with clinical details."""
        section = f"""## Intended Use
        
### Clinical Indication
{intended_use}

### Target Population
{population}

### Clinical Environment
This model is intended for use by qualified healthcare professionals in clinical settings with appropriate oversight and quality assurance measures.

### Limitations
- Not intended for use as sole basis for clinical decision-making
- Requires clinical interpretation and validation
- Performance may vary across different patient populations
- Should be used in conjunction with other clinical information
"""
        card.add_section("intended_use", section.strip())

    def _add_clinical_validation_section(self, card: ModelCard, validation: ClinicalValidation) -> None:
        """Add clinical validation section."""
        section = f"""## Clinical Validation

### Performance Metrics
- **Sensitivity**: {validation.sensitivity:.3f} ({validation.sensitivity*100:.1f}%)
- **Specificity**: {validation.specificity:.3f} ({validation.specificity*100:.1f}%)
- **AUC**: {validation.auc:.3f}"""

        if validation.ppv is not None:
            section += f"\n- **Positive Predictive Value**: {validation.ppv:.3f} ({validation.ppv*100:.1f}%)"

        if validation.npv is not None:
            section += f"\n- **Negative Predictive Value**: {validation.npv:.3f} ({validation.npv*100:.1f}%)"

        if validation.cohort_size:
            section += f"\n\n### Study Cohort\n- **Size**: {validation.cohort_size:,} patients"

        if validation.validation_sites:
            section += f"\n- **Sites**: {', '.join(validation.validation_sites)}"

        if validation.follow_up_period:
            section += f"\n- **Follow-up Period**: {validation.follow_up_period}"

        section += """

### Statistical Analysis
All performance metrics were calculated using appropriate statistical methods with confidence intervals. Cross-validation was performed to assess generalizability across different patient populations and clinical sites.
"""

        card.add_section("clinical_validation", section.strip())

    def _add_regulatory_section(self, card: ModelCard, regulatory: RegulatoryInfo) -> None:
        """Add regulatory status section."""
        section = f"""## Regulatory Status

### Current Status
**{regulatory.status.title()}** by {regulatory.agency}"""

        if regulatory.pathway:
            section += f"\n\n### Regulatory Pathway\n{regulatory.pathway}"

        if regulatory.submission_date:
            section += f"\n\n### Key Dates\n- **Submission**: {regulatory.submission_date.strftime('%B %d, %Y')}"

        if regulatory.approval_date:
            section += f"\n- **Approval**: {regulatory.approval_date.strftime('%B %d, %Y')}"

        if regulatory.indication:
            section += f"\n\n### Approved Indication\n{regulatory.indication}"

        section += """

### Compliance
This model card complies with regulatory requirements for medical device software documentation and transparency."""

        card.add_section("regulatory_status", section.strip())

    def _add_contraindications_section(self, card: ModelCard, contraindications: List[str]) -> None:
        """Add contraindications and warnings section."""
        section = """## Contraindications and Warnings

### Contraindications
The following are contraindications for use of this model:
"""
        for contraindication in contraindications:
            section += f"- {contraindication}\n"

        section += """
### Warnings and Precautions
- Model performance has not been validated in pediatric populations unless specifically indicated
- Use caution in patients with comorbid conditions not represented in training data
- Regular model performance monitoring is required in clinical deployment
- Healthcare providers should be trained on appropriate use and interpretation

### Adverse Event Reporting
Any adverse events or performance issues should be reported to the model manufacturer and relevant regulatory authorities.
"""

        card.add_section("contraindications", section.strip())

    def _add_risk_assessment_section(self, card: ModelCard, validation: ClinicalValidation) -> None:
        """Add clinical risk assessment section."""
        # Calculate clinical risk metrics
        false_positive_rate = 1 - validation.specificity
        false_negative_rate = 1 - validation.sensitivity

        section = f"""## Risk Assessment

### Clinical Risk Analysis
- **False Positive Rate**: {false_positive_rate:.3f} ({false_positive_rate*100:.1f}%)
  - *Risk*: Unnecessary procedures, patient anxiety, increased healthcare costs
- **False Negative Rate**: {false_negative_rate:.3f} ({false_negative_rate*100:.1f}%)
  - *Risk*: Delayed diagnosis, missed treatment opportunities

### Risk Mitigation Strategies
1. **Clinical Oversight**: All model outputs require review by qualified clinicians
2. **Quality Assurance**: Regular performance monitoring and validation
3. **Training**: Comprehensive training for all users on model limitations
4. **Documentation**: Maintain detailed logs of all model predictions and outcomes

### Risk-Benefit Analysis
The clinical benefits of using this model have been demonstrated to outweigh the risks when used appropriately in the intended clinical setting with proper safeguards.
"""

        card.add_section("risk_assessment", section.strip())

    def _add_data_privacy_section(self, card: ModelCard, data_sources: List[str]) -> None:
        """Add data privacy and security section."""
        section = """## Data Privacy and Security

### Data Sources
Training data was obtained from the following sources:
"""
        for source in data_sources:
            section += f"- {source}\n"

        section += """
### Privacy Protection Measures
- All training data was deidentified according to HIPAA Safe Harbor standards
- No direct patient identifiers are stored or transmitted by the model
- Data encryption in transit and at rest
- Access controls and audit logging implemented

### Compliance
- **HIPAA**: Fully compliant with HIPAA privacy and security rules
- **GDPR**: Compliant with EU General Data Protection Regulation where applicable
- **FDA**: Follows FDA guidance on software as medical device cybersecurity

### Data Retention
Patient data is not retained by the model after processing. All temporary data is securely deleted according to institutional policies.
"""

        card.add_section("data_privacy", section.strip())

    def _add_clinical_metrics(self, card: ModelCard, validation: ClinicalValidation) -> None:
        """Add clinical performance metrics to the model card."""
        card.add_metric("sensitivity", validation.sensitivity)
        card.add_metric("specificity", validation.specificity)
        card.add_metric("auc", validation.auc)

        if validation.ppv is not None:
            card.add_metric("positive_predictive_value", validation.ppv)

        if validation.npv is not None:
            card.add_metric("negative_predictive_value", validation.npv)

        # Calculate additional clinical metrics
        false_positive_rate = 1 - validation.specificity
        false_negative_rate = 1 - validation.sensitivity

        card.add_metric("false_positive_rate", false_positive_rate)
        card.add_metric("false_negative_rate", false_negative_rate)


class FinancialAITemplate(Template):
    """Template for financial AI model cards with fairness and regulatory compliance."""

    def __init__(self):
        super().__init__(
            name="financial_ai",
            required_sections=[
                "business_purpose",
                "fairness_assessment",
                "regulatory_compliance",
                "explainability",
                "monitoring",
                "governance"
            ]
        )

    def create(self,
               model_name: str,
               business_purpose: str,
               fairness_metrics: FairnessMetrics,
               regulatory_compliance: List[str],
               explainability_method: str,
               protected_attributes: List[str],
               **kwargs) -> ModelCard:
        """Create financial AI model card.
        
        Args:
            model_name: Name of the financial AI model
            business_purpose: Business purpose and use case
            fairness_metrics: Fairness assessment metrics
            regulatory_compliance: List of regulatory frameworks
            explainability_method: Method used for model explainability
            protected_attributes: List of protected attributes monitored
            **kwargs: Additional configuration options
            
        Returns:
            Financial AI model card
        """
        config = CardConfig(
            format=CardFormat.HUGGINGFACE,
            include_ethical_considerations=True,
            regulatory_standard="financial_services"
        )

        card = ModelCard(config)

        # Model details
        card.model_details.name = model_name
        card.model_details.description = f"Financial AI model for {business_purpose}"
        card.model_details.version = kwargs.get("version", "1.0.0")
        card.model_details.license = kwargs.get("license", "Proprietary")
        card.model_details.tags = ["financial-ai", "risk-assessment", "compliance"]

        # Add financial-specific sections
        self._add_business_purpose_section(card, business_purpose)
        self._add_fairness_section(card, fairness_metrics, protected_attributes)
        self._add_regulatory_section(card, regulatory_compliance)
        self._add_explainability_section(card, explainability_method)
        self._add_monitoring_section(card, protected_attributes)
        self._add_governance_section(card)

        # Add fairness metrics
        self._add_fairness_metrics(card, fairness_metrics)

        # Metadata
        card.metadata.update({
            "domain": "financial_services",
            "regulatory_frameworks": regulatory_compliance,
            "fairness_validated": True,
            "explainable": True,
            "protected_attributes": protected_attributes
        })

        return card

    def _add_business_purpose_section(self, card: ModelCard, purpose: str) -> None:
        """Add business purpose section."""
        section = f"""## Business Purpose and Use Case

### Primary Use Case
{purpose}

### Business Value
This model provides automated decision support to improve efficiency, consistency, and accuracy in financial decision-making while maintaining fairness and regulatory compliance.

### Scope of Use
- Intended for use by qualified financial professionals
- Operates within established business processes and controls
- Subject to human oversight and review
- Used in conjunction with other risk management tools

### Out of Scope
- Not intended for fully automated decision-making without human review
- Not validated for use outside specified business context
- Should not be used as sole basis for adverse actions affecting consumers
"""

        card.add_section("business_purpose", section.strip())

    def _add_fairness_section(self, card: ModelCard, metrics: FairnessMetrics, attributes: List[str]) -> None:
        """Add fairness assessment section."""
        section = f"""## Fairness Assessment

### Fairness Metrics
- **Demographic Parity**: {metrics.demographic_parity:.4f}
- **Equal Opportunity**: {metrics.equal_opportunity:.4f}"""

        if metrics.equalized_odds is not None:
            section += f"\n- **Equalized Odds**: {metrics.equalized_odds:.4f}"

        if metrics.calibration is not None:
            section += f"\n- **Calibration**: {metrics.calibration:.4f}"

        if metrics.individual_fairness is not None:
            section += f"\n- **Individual Fairness**: {metrics.individual_fairness:.4f}"

        section += """

### Protected Attributes Monitored
The following protected attributes are monitored for fairness:
"""
        for attr in attributes:
            section += f"- {attr}\n"

        section += """
### Fairness Methodology
Fairness assessments were conducted using industry-standard metrics and methodologies. Regular monitoring ensures ongoing compliance with fairness standards.

### Bias Mitigation
- Pre-processing: Data preprocessing techniques to reduce historical biases
- In-processing: Fairness constraints incorporated during model training
- Post-processing: Output calibration to ensure equitable outcomes across groups
"""

        card.add_section("fairness_assessment", section.strip())

    def _add_regulatory_section(self, card: ModelCard, compliance: List[str]) -> None:
        """Add regulatory compliance section."""
        section = """## Regulatory Compliance

### Applicable Regulations
This model has been designed and validated to comply with:
"""
        for regulation in compliance:
            section += f"- {regulation}\n"

        section += """
### Compliance Measures
- **Fair Credit Reporting Act (FCRA)**: Model outputs meet FCRA accuracy and fairness requirements
- **Equal Credit Opportunity Act (ECOA)**: Systematic testing for disparate impact across protected classes
- **Fair Housing Act (FHA)**: Where applicable, housing-related decisions comply with FHA requirements
- **Model Risk Management**: Following regulatory guidance on model risk management (SR 11-7)

### Documentation and Audit Trail
- Comprehensive model documentation maintained
- Decision logs and audit trails preserved
- Regular model validation and testing conducted
- Compliance monitoring and reporting processes established
"""

        card.add_section("regulatory_compliance", section.strip())

    def _add_explainability_section(self, card: ModelCard, method: str) -> None:
        """Add explainability section."""
        section = f"""## Model Explainability

### Explainability Method
**{method}**

### Feature Importance
The model provides feature importance scores indicating which factors most influence predictions. This enables:
- Understanding of key decision drivers
- Validation of business logic
- Identification of potential biases
- Compliance with explainability requirements

### Individual Explanations
For each prediction, the model can provide:
- Top contributing factors
- Factor importance scores  
- Directional impact (positive/negative)
- Confidence intervals

### Business Logic Validation
Model explanations are regularly reviewed to ensure:
- Consistency with business expertise
- Alignment with regulatory expectations
- Absence of prohibited factors
- Reasonable and defensible decision logic
"""

        card.add_section("explainability", section.strip())

    def _add_monitoring_section(self, card: ModelCard, attributes: List[str]) -> None:
        """Add ongoing monitoring section."""
        section = """## Ongoing Monitoring

### Performance Monitoring
- **Model Performance**: Regular assessment of predictive accuracy
- **Population Stability**: Monitoring for data drift and population shifts
- **Fairness Metrics**: Continuous monitoring of fairness across protected classes
- **Business Outcomes**: Tracking of business KPIs and success metrics

### Monitoring Frequency
- Real-time: Key performance indicators and system health
- Daily: Fairness metrics and population statistics
- Weekly: Model performance and drift detection
- Monthly: Comprehensive model validation and review
- Quarterly: Model governance and compliance assessment

### Alert Thresholds
Automated alerts trigger when:
- Performance metrics drop below acceptable thresholds
- Fairness metrics exceed tolerance levels
- Data drift indicators signal significant changes
- System errors or anomalies are detected

### Protected Attributes Monitoring
Continuous monitoring of outcomes across protected attributes:
"""

        for attr in attributes:
            section += f"- {attr}: Automated fairness metric calculation\n"

        card.add_section("monitoring", section.strip())

    def _add_governance_section(self, card: ModelCard) -> None:
        """Add governance section."""
        section = """## Model Governance

### Governance Structure
- **Model Owner**: Responsible for model performance and compliance
- **Model Risk Management**: Independent validation and ongoing oversight  
- **Compliance Team**: Regulatory compliance monitoring and reporting
- **Business Users**: Appropriate use and interpretation of model outputs

### Review and Approval Process
1. **Development**: Model development following established standards
2. **Validation**: Independent validation of model performance and fairness
3. **Approval**: Governance committee approval before production deployment
4. **Monitoring**: Ongoing performance and compliance monitoring
5. **Review**: Regular model review and revalidation

### Documentation Standards
- Model development documentation
- Validation and testing reports
- Fairness assessment documentation
- Ongoing monitoring reports
- Compliance and audit documentation

### Change Management
All model changes follow established change management procedures:
- Impact assessment and risk analysis
- Stakeholder review and approval
- Testing and validation requirements  
- Documentation and audit trail maintenance
"""

        card.add_section("governance", section.strip())

    def _add_fairness_metrics(self, card: ModelCard, metrics: FairnessMetrics) -> None:
        """Add fairness metrics to the model card."""
        card.add_metric("demographic_parity", metrics.demographic_parity)
        card.add_metric("equal_opportunity", metrics.equal_opportunity)

        if metrics.equalized_odds is not None:
            card.add_metric("equalized_odds", metrics.equalized_odds)

        if metrics.calibration is not None:
            card.add_metric("calibration", metrics.calibration)

        if metrics.individual_fairness is not None:
            card.add_metric("individual_fairness", metrics.individual_fairness)


class BiometricAITemplate(Template):
    """Template for biometric AI model cards with privacy and security focus."""

    def __init__(self):
        super().__init__(
            name="biometric_ai",
            required_sections=[
                "privacy_protection",
                "consent_mechanism",
                "data_retention",
                "security_measures",
                "algorithmic_fairness",
                "liveness_detection"
            ]
        )

    def create(self,
               model_name: str,
               biometric_type: str,
               privacy_measures: List[str],
               retention_policy: str,
               accuracy_by_demographics: Dict[str, float],
               **kwargs) -> ModelCard:
        """Create biometric AI model card.
        
        Args:
            model_name: Name of the biometric AI model
            biometric_type: Type of biometric (face, fingerprint, iris, etc.)
            privacy_measures: List of privacy protection measures
            retention_policy: Data retention policy
            accuracy_by_demographics: Accuracy metrics by demographic groups
            **kwargs: Additional configuration options
            
        Returns:
            Biometric AI model card
        """
        config = CardConfig(
            format=CardFormat.HUGGINGFACE,
            include_ethical_considerations=True,
            regulatory_standard="biometric_privacy"
        )

        card = ModelCard(config)

        # Model details
        card.model_details.name = model_name
        card.model_details.description = f"Biometric AI model for {biometric_type} recognition and verification"
        card.model_details.version = kwargs.get("version", "1.0.0")
        card.model_details.license = kwargs.get("license", "Proprietary")
        card.model_details.tags = ["biometric-ai", "privacy-preserving", "identity-verification"]

        # Add biometric-specific sections
        self._add_privacy_section(card, privacy_measures)
        self._add_consent_section(card)
        self._add_retention_section(card, retention_policy)
        self._add_security_section(card)
        self._add_fairness_section(card, accuracy_by_demographics)
        self._add_liveness_section(card, biometric_type)

        # Metadata
        card.metadata.update({
            "domain": "biometric_identification",
            "biometric_type": biometric_type,
            "privacy_preserving": True,
            "liveness_detection": True,
            "demographic_fairness_validated": True
        })

        return card

    def _add_privacy_section(self, card: ModelCard, measures: List[str]) -> None:
        """Add privacy protection section."""
        section = """## Privacy Protection

### Privacy-by-Design Principles
This biometric AI model implements privacy-by-design principles:

#### Privacy Protection Measures
"""
        for measure in measures:
            section += f"- {measure}\n"

        section += """
#### Technical Safeguards
- **Template Protection**: Biometric templates are irreversibly encrypted
- **Data Minimization**: Only necessary biometric features are extracted and stored
- **Anonymization**: Personal identifiers are separated from biometric data
- **Secure Processing**: All biometric processing occurs in secure, isolated environments

#### Legal Compliance
- GDPR Article 9 compliance for special category data processing
- CCPA compliance for biometric identifier protection
- BIPA (Biometric Information Privacy Act) compliance where applicable
"""

        card.add_section("privacy_protection", section.strip())

    def _add_consent_section(self, card: ModelCard) -> None:
        """Add consent mechanism section."""
        section = """## Consent Mechanism

### Informed Consent Requirements
- **Clear Notice**: Users receive clear information about biometric data collection
- **Purpose Specification**: Specific purposes for biometric processing are disclosed
- **Explicit Consent**: Active, informed consent required before processing
- **Withdrawal Rights**: Users can withdraw consent and request data deletion

### Consent Documentation
- Consent timestamp and version tracking
- Audit trail of consent decisions
- Regular consent renewal processes
- Clear opt-out mechanisms

### Special Populations
- Enhanced protections for minors
- Accessibility considerations for disabled users  
- Multi-language consent options
- Cultural sensitivity in consent processes
"""

        card.add_section("consent_mechanism", section.strip())

    def _add_retention_section(self, card: ModelCard, policy: str) -> None:
        """Add data retention section."""
        section = f"""## Data Retention Policy

### Retention Schedule
{policy}

### Deletion Procedures
- **Automated Deletion**: Systematic deletion based on retention schedules
- **Secure Erasure**: Cryptographically secure deletion methods
- **Backup Purging**: Deletion from all backup systems and archives
- **Third-Party Coordination**: Ensuring deletion across all processing partners

### User Rights
- **Right to Deletion**: Users can request immediate deletion of biometric data
- **Data Portability**: Users can request their biometric templates in portable format
- **Access Rights**: Users can access information about their stored biometric data
- **Correction Rights**: Users can request correction of inaccurate data

### Audit and Compliance
- Regular audits of data retention compliance
- Documentation of all deletion activities
- Compliance reporting to relevant authorities
"""

        card.add_section("data_retention", section.strip())

    def _add_security_section(self, card: ModelCard) -> None:
        """Add security measures section."""
        section = """## Security Measures

### Cryptographic Protection
- **End-to-End Encryption**: All biometric data encrypted in transit and at rest
- **Key Management**: Hardware security modules for cryptographic key protection
- **Template Encryption**: Biometric templates stored in encrypted, irreversible format
- **Zero-Knowledge Architecture**: No plaintext biometric data stored

### Access Controls
- **Multi-Factor Authentication**: Required for all system access
- **Role-Based Access**: Principle of least privilege applied
- **Audit Logging**: Comprehensive logging of all access and processing activities
- **Regular Access Reviews**: Periodic review and validation of access rights

### Infrastructure Security
- **Secure Enclaves**: Processing in hardware-protected secure environments
- **Network Isolation**: Biometric processing networks isolated from general systems
- **Intrusion Detection**: Real-time monitoring for security threats
- **Vulnerability Management**: Regular security assessments and updates

### Incident Response
- **Breach Detection**: Automated systems for detecting potential breaches
- **Response Procedures**: Established procedures for security incidents
- **Notification Processes**: Clear processes for notifying affected users and authorities
- **Recovery Planning**: Comprehensive disaster recovery and business continuity plans
"""

        card.add_section("security_measures", section.strip())

    def _add_fairness_section(self, card: ModelCard, accuracy_by_demographics: Dict[str, float]) -> None:
        """Add algorithmic fairness section."""
        section = """## Algorithmic Fairness

### Demographic Performance Analysis
Performance metrics across different demographic groups:

"""
        for group, accuracy in accuracy_by_demographics.items():
            section += f"- **{group}**: {accuracy:.3f} ({accuracy*100:.1f}% accuracy)\n"

        section += """
### Bias Mitigation Strategies
- **Diverse Training Data**: Training datasets balanced across demographic groups
- **Fairness Constraints**: Mathematical fairness constraints incorporated in training
- **Regular Testing**: Ongoing testing for demographic bias and performance disparities
- **Threshold Optimization**: Per-group threshold optimization to ensure equitable outcomes

### Continuous Monitoring
- Real-time monitoring of performance across demographic groups
- Automated alerts for performance disparities
- Regular retraining to address emerging biases
- Stakeholder feedback incorporation

### Accessibility Considerations
- Support for users with disabilities
- Alternative authentication methods for users unable to provide biometric samples
- Inclusive design principles applied throughout development
"""

        card.add_section("algorithmic_fairness", section.strip())

    def _add_liveness_section(self, card: ModelCard, biometric_type: str) -> None:
        """Add liveness detection section."""
        section = f"""## Liveness Detection

### Anti-Spoofing Measures
Advanced liveness detection specifically designed for {biometric_type}:

- **Multi-Modal Detection**: Multiple sensors and detection methods
- **Challenge-Response**: Interactive challenges to verify live presence  
- **Temporal Analysis**: Analysis of natural biometric variations over time
- **3D Analysis**: Depth and spatial analysis to detect presentation attacks

### Attack Vector Protection
- **Photo Attacks**: Protection against printed photos and digital displays
- **Video Attacks**: Detection of video replay attacks
- **Mask Attacks**: Detection of artificial masks or prosthetics
- **Deep Fake Protection**: Advanced detection of synthetic biometric presentations

### Performance Metrics
- **False Accept Rate (FAR)**: Rate of accepting spoofed presentations
- **False Reject Rate (FRR)**: Rate of rejecting genuine live presentations  
- **Attack Detection Rate**: Percentage of presentation attacks successfully detected
- **Real-Time Performance**: Sub-second liveness determination

### Continuous Improvement
- Regular updates to address new attack vectors
- Machine learning-based adaptation to evolving threats
- Security research collaboration and threat intelligence integration
"""

        card.add_section("liveness_detection", section.strip())
