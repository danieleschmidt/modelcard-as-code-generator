"""
Compliance checker for regulatory standards.

Validates model cards against various regulatory frameworks
including GDPR, EU AI Act, CCPA, and other standards.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import logging

from ..core.model_card import ModelCard


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    GDPR = "gdpr"
    EU_AI_ACT = "eu_ai_act"
    CCPA = "ccpa"
    ISO_23053 = "iso_23053"
    SOX = "sox"
    HIPAA = "hipaa"


@dataclass
class ComplianceRequirement:
    """Represents a compliance requirement."""
    id: str
    description: str
    required: bool
    section: Optional[str] = None
    field: Optional[str] = None
    check_function: Optional[str] = None


@dataclass
class ComplianceResult:
    """Result of compliance checking."""
    standard: str
    compliant: bool
    score: float  # 0.0 to 1.0
    missing_requirements: List[str]
    satisfied_requirements: List[str]
    warnings: List[str]
    recommendations: List[str]


class ComplianceChecker:
    """Checks model card compliance against regulatory standards."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_requirements()
    
    def _initialize_requirements(self) -> None:
        """Initialize compliance requirements for different standards."""
        self.requirements = {
            ComplianceStandard.GDPR.value: [
                ComplianceRequirement(
                    id="gdpr_purpose_limitation",
                    description="Purpose limitation - clear intended use specified",
                    required=True,
                    section="intended_use",
                    check_function="check_purpose_limitation"
                ),
                ComplianceRequirement(
                    id="gdpr_data_minimization",
                    description="Data minimization - only necessary data used",
                    required=True,
                    section="data_governance",
                    check_function="check_data_minimization"
                ),
                ComplianceRequirement(
                    id="gdpr_transparency",
                    description="Transparency - clear information about processing",
                    required=True,
                    section="transparency",
                    check_function="check_transparency"
                ),
                ComplianceRequirement(
                    id="gdpr_accuracy",
                    description="Accuracy - measures to ensure data accuracy",
                    required=True,
                    section="accuracy_measures",
                    check_function="check_accuracy_measures"
                ),
                ComplianceRequirement(
                    id="gdpr_accountability",
                    description="Accountability - demonstrate compliance",
                    required=True,
                    check_function="check_accountability"
                ),
            ],
            
            ComplianceStandard.EU_AI_ACT.value: [
                ComplianceRequirement(
                    id="ai_act_risk_assessment",
                    description="Risk assessment conducted and documented",
                    required=True,
                    section="risk_assessment",
                    check_function="check_risk_assessment"
                ),
                ComplianceRequirement(
                    id="ai_act_quality_management",
                    description="Quality management system in place",
                    required=True,
                    check_function="check_quality_management"
                ),
                ComplianceRequirement(
                    id="ai_act_data_governance",
                    description="Data governance and data management practices",
                    required=True,
                    section="data_governance",
                    check_function="check_data_governance"
                ),
                ComplianceRequirement(
                    id="ai_act_transparency",
                    description="Transparency and provision of information",
                    required=True,
                    section="transparency",
                    check_function="check_ai_transparency"
                ),
                ComplianceRequirement(
                    id="ai_act_human_oversight",
                    description="Human oversight measures",
                    required=True,
                    section="human_oversight",
                    check_function="check_human_oversight"
                ),
                ComplianceRequirement(
                    id="ai_act_accuracy_robustness",
                    description="Accuracy, robustness and cybersecurity",
                    required=True,
                    section="technical_robustness",
                    check_function="check_accuracy_robustness"
                ),
            ],
            
            ComplianceStandard.CCPA.value: [
                ComplianceRequirement(
                    id="ccpa_data_collection_notice",
                    description="Notice of data collection and use",
                    required=True,
                    section="data_governance",
                    check_function="check_data_collection_notice"
                ),
                ComplianceRequirement(
                    id="ccpa_purpose_specification",
                    description="Specific purposes for data collection",
                    required=True,
                    section="intended_use",
                    check_function="check_purpose_specification"
                ),
                ComplianceRequirement(
                    id="ccpa_consumer_rights",
                    description="Consumer rights and mechanisms",
                    required=True,
                    check_function="check_consumer_rights"
                ),
            ],
            
            ComplianceStandard.ISO_23053.value: [
                ComplianceRequirement(
                    id="iso_trustworthiness_framework",
                    description="Framework for AI trustworthiness",
                    required=True,
                    check_function="check_trustworthiness_framework"
                ),
                ComplianceRequirement(
                    id="iso_risk_management",
                    description="AI risk management process",
                    required=True,
                    section="risk_assessment",
                    check_function="check_iso_risk_management"
                ),
                ComplianceRequirement(
                    id="iso_verification_validation",
                    description="Verification and validation processes",
                    required=True,
                    check_function="check_verification_validation"
                ),
            ],
        }
    
    def check_compliance(
        self,
        model_card: ModelCard,
        standard: str,
        strict: bool = False
    ) -> ComplianceResult:
        """
        Check model card compliance against a specific standard.
        
        Args:
            model_card: ModelCard to check
            standard: Compliance standard to check against
            strict: Whether to use strict compliance checking
            
        Returns:
            ComplianceResult with compliance status and details
        """
        try:
            if standard not in self.requirements:
                raise ValueError(f"Unsupported compliance standard: {standard}")
            
            requirements = self.requirements[standard]
            satisfied = []
            missing = []
            warnings = []
            recommendations = []
            
            # Check each requirement
            for req in requirements:
                is_satisfied = self._check_requirement(model_card, req, warnings)
                
                if is_satisfied:
                    satisfied.append(req.id)
                else:
                    if req.required:
                        missing.append(req.id)
                    else:
                        warnings.append(f"Optional requirement not met: {req.description}")
            
            # Calculate compliance score
            total_required = sum(1 for req in requirements if req.required)
            satisfied_required = sum(1 for req_id in satisfied 
                                   if any(req.id == req_id and req.required for req in requirements))
            
            score = satisfied_required / total_required if total_required > 0 else 1.0
            compliant = len(missing) == 0 if strict else score >= 0.8
            
            # Generate recommendations
            recommendations.extend(self._generate_recommendations(standard, missing, warnings))
            
            return ComplianceResult(
                standard=standard,
                compliant=compliant,
                score=score,
                missing_requirements=missing,
                satisfied_requirements=satisfied,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Compliance check failed for {standard}: {e}")
            return ComplianceResult(
                standard=standard,
                compliant=False,
                score=0.0,
                missing_requirements=[],
                satisfied_requirements=[],
                warnings=[f"Compliance check failed: {str(e)}"],
                recommendations=["Manual compliance review required due to check failure"]
            )
    
    def _check_requirement(
        self,
        model_card: ModelCard,
        requirement: ComplianceRequirement,
        warnings: List[str]
    ) -> bool:
        """Check if a specific requirement is satisfied."""
        try:
            # Use check function if specified
            if requirement.check_function:
                check_method = getattr(self, requirement.check_function, None)
                if check_method:
                    return check_method(model_card, requirement, warnings)
            
            # Default check based on section/field presence
            if requirement.section:
                section_data = model_card.get_section(requirement.section)
                return section_data is not None and bool(section_data)
            
            # If no specific check, assume satisfied
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to check requirement {requirement.id}: {e}")
            warnings.append(f"Could not verify requirement: {requirement.description}")
            return False
    
    # GDPR compliance checks
    
    def check_purpose_limitation(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check GDPR purpose limitation compliance."""
        intended_use = model_card.intended_use
        if not intended_use:
            return False
        
        has_primary_use = bool(intended_use.get('primary_use'))
        has_scope_definition = bool(intended_use.get('out_of_scope'))
        
        if not has_primary_use:
            warnings.append("Primary use not clearly defined")
        if not has_scope_definition:
            warnings.append("Out-of-scope uses not defined")
        
        return has_primary_use and has_scope_definition
    
    def check_data_minimization(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check GDPR data minimization compliance."""
        data_governance = model_card.get_section('data_governance')
        if not data_governance:
            return False
        
        has_minimization = 'data_minimization' in data_governance
        has_necessity_justification = 'necessity_justification' in data_governance
        
        if not has_minimization:
            warnings.append("Data minimization practices not documented")
        
        return has_minimization or has_necessity_justification
    
    def check_transparency(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check GDPR transparency compliance."""
        transparency = model_card.get_section('transparency')
        has_transparency_section = transparency is not None
        
        has_model_description = bool(model_card.model_details.description)
        has_intended_use = bool(model_card.intended_use)
        
        if not has_model_description:
            warnings.append("Model description insufficient for transparency")
        
        return has_transparency_section and has_model_description and has_intended_use
    
    def check_accuracy_measures(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check GDPR accuracy measures compliance."""
        has_metrics = len(model_card.metrics) > 0
        has_accuracy_section = model_card.get_section('accuracy_measures') is not None
        has_validation = bool(model_card.evaluation_data)
        
        if not has_metrics:
            warnings.append("No performance metrics provided")
        if not has_validation:
            warnings.append("No evaluation data documented")
        
        return has_metrics and (has_accuracy_section or has_validation)
    
    def check_accountability(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check GDPR accountability compliance."""
        has_organization = 'organization' in model_card.metadata
        has_contact = 'contact' in model_card.metadata
        has_documentation = bool(model_card.model_details.description)
        
        if not has_organization:
            warnings.append("Responsible organization not identified")
        if not has_contact:
            warnings.append("Contact information not provided")
        
        return has_organization or has_contact or has_documentation
    
    # EU AI Act compliance checks
    
    def check_risk_assessment(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check EU AI Act risk assessment compliance."""
        risk_assessment = model_card.risk_assessment
        if not risk_assessment:
            return False
        
        has_risk_level = 'risk_level' in risk_assessment
        has_mitigation = 'mitigation_measures' in risk_assessment
        has_assessment_method = 'assessment_methodology' in risk_assessment
        
        if not has_risk_level:
            warnings.append("Risk level not classified")
        if not has_mitigation:
            warnings.append("Risk mitigation measures not documented")
        
        return has_risk_level and has_mitigation
    
    def check_quality_management(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check EU AI Act quality management compliance."""
        has_quality_section = model_card.get_section('quality_management') is not None
        has_testing_procedures = bool(model_card.evaluation_data)
        has_metrics = len(model_card.metrics) > 0
        
        if not has_testing_procedures:
            warnings.append("Testing procedures not documented")
        
        return has_quality_section or (has_testing_procedures and has_metrics)
    
    def check_data_governance(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check EU AI Act data governance compliance."""
        data_governance = model_card.get_section('data_governance')
        if not data_governance:
            return False
        
        has_data_quality = 'data_quality' in data_governance
        has_data_lineage = 'data_lineage' in data_governance
        has_training_data = len(model_card.training_data) > 0
        
        if not has_data_quality:
            warnings.append("Data quality measures not documented")
        if not has_data_lineage:
            warnings.append("Data lineage not documented")
        
        return has_data_quality and (has_data_lineage or has_training_data)
    
    def check_ai_transparency(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check EU AI Act transparency compliance."""
        transparency = model_card.get_section('transparency')
        if not transparency:
            return False
        
        has_explainability = 'explainability' in transparency
        has_user_info = 'user_information' in transparency
        has_limitations = bool(model_card.caveats_and_recommendations)
        
        if not has_explainability:
            warnings.append("Explainability measures not documented")
        if not has_limitations:
            warnings.append("Model limitations not documented")
        
        return has_explainability or has_user_info or has_limitations
    
    def check_human_oversight(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check EU AI Act human oversight compliance."""
        oversight = model_card.get_section('human_oversight')
        if not oversight:
            return False
        
        has_oversight_level = 'oversight_level' in oversight
        has_intervention = 'human_intervention' in oversight
        has_measures = 'oversight_measures' in oversight
        
        if not has_oversight_level:
            warnings.append("Human oversight level not specified")
        
        return has_oversight_level or has_intervention or has_measures
    
    def check_accuracy_robustness(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check EU AI Act accuracy and robustness compliance."""
        tech_robustness = model_card.technical_robustness
        if not tech_robustness:
            return False
        
        has_accuracy = 'accuracy_metrics' in tech_robustness
        has_robustness = 'robustness_tests' in tech_robustness
        has_security = model_card.get_section('cybersecurity_measures') is not None
        
        if not has_accuracy:
            warnings.append("Accuracy metrics not documented")
        if not has_robustness:
            warnings.append("Robustness testing not documented")
        
        return has_accuracy and (has_robustness or has_security)
    
    # CCPA compliance checks
    
    def check_data_collection_notice(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check CCPA data collection notice compliance."""
        data_governance = model_card.get_section('data_governance')
        has_notice = data_governance and 'data_collection_notice' in data_governance
        has_training_data_info = len(model_card.training_data) > 0
        
        if not has_notice and not has_training_data_info:
            warnings.append("Data collection practices not documented")
        
        return has_notice or has_training_data_info
    
    def check_purpose_specification(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check CCPA purpose specification compliance."""
        intended_use = model_card.intended_use
        has_specific_purpose = intended_use and 'primary_use' in intended_use
        
        if not has_specific_purpose:
            warnings.append("Specific data use purposes not documented")
        
        return has_specific_purpose
    
    def check_consumer_rights(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check CCPA consumer rights compliance."""
        has_contact = 'contact' in model_card.metadata
        has_rights_section = model_card.get_section('consumer_rights') is not None
        
        if not has_contact:
            warnings.append("Contact information for consumer rights not provided")
        
        return has_contact or has_rights_section
    
    # ISO 23053 compliance checks
    
    def check_trustworthiness_framework(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check ISO 23053 trustworthiness framework compliance."""
        has_ethical_considerations = bool(model_card.ethical_considerations)
        has_risk_assessment = bool(model_card.risk_assessment)
        has_limitations = bool(model_card.caveats_and_recommendations)
        
        trustworthiness_components = sum([
            has_ethical_considerations,
            has_risk_assessment,
            has_limitations
        ])
        
        if trustworthiness_components < 2:
            warnings.append("Insufficient trustworthiness framework components")
        
        return trustworthiness_components >= 2
    
    def check_iso_risk_management(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check ISO 23053 risk management compliance."""
        risk_assessment = model_card.risk_assessment
        if not risk_assessment:
            return False
        
        has_process = 'assessment_methodology' in risk_assessment
        has_controls = 'mitigation_measures' in risk_assessment
        has_monitoring = 'monitoring' in risk_assessment
        
        return has_process and has_controls
    
    def check_verification_validation(self, model_card: ModelCard, req: ComplianceRequirement, warnings: List[str]) -> bool:
        """Check ISO 23053 verification and validation compliance."""
        has_evaluation = len(model_card.evaluation_data) > 0
        has_metrics = len(model_card.metrics) > 0
        has_testing = model_card.get_section('validation_testing') is not None
        
        verification_components = sum([has_evaluation, has_metrics, has_testing])
        
        if verification_components < 2:
            warnings.append("Insufficient verification and validation documentation")
        
        return verification_components >= 2
    
    def _generate_recommendations(
        self,
        standard: str,
        missing_requirements: List[str],
        warnings: List[str]
    ) -> List[str]:
        """Generate recommendations for improving compliance."""
        recommendations = []
        
        if not missing_requirements and not warnings:
            recommendations.append(f"Model card is compliant with {standard} requirements")
            return recommendations
        
        # Standard-specific recommendations
        if standard == ComplianceStandard.GDPR.value:
            if any('purpose' in req for req in missing_requirements):
                recommendations.append("Clearly define the intended use and purpose limitations")
            if any('data_minimization' in req for req in missing_requirements):
                recommendations.append("Document data minimization practices and necessity justification")
            if any('transparency' in req for req in missing_requirements):
                recommendations.append("Enhance transparency documentation with detailed model explanations")
        
        elif standard == ComplianceStandard.EU_AI_ACT.value:
            if any('risk' in req for req in missing_requirements):
                recommendations.append("Complete comprehensive risk assessment and classification")
            if any('oversight' in req for req in missing_requirements):
                recommendations.append("Define human oversight measures and intervention capabilities")
            if any('data_governance' in req for req in missing_requirements):
                recommendations.append("Implement robust data governance and quality management practices")
        
        # General recommendations
        if missing_requirements:
            recommendations.append(f"Address {len(missing_requirements)} missing requirements for full compliance")
        
        if warnings:
            recommendations.append("Review and address compliance warnings for enhanced documentation")
        
        return recommendations
    
    def check_multiple_standards(
        self,
        model_card: ModelCard,
        standards: List[str],
        strict: bool = False
    ) -> Dict[str, ComplianceResult]:
        """Check compliance against multiple standards."""
        results = {}
        
        for standard in standards:
            results[standard] = self.check_compliance(model_card, standard, strict)
        
        return results
    
    def get_compliance_summary(self, results: Dict[str, ComplianceResult]) -> Dict[str, Any]:
        """Get summary of compliance results across multiple standards."""
        total_standards = len(results)
        compliant_standards = sum(1 for result in results.values() if result.compliant)
        average_score = sum(result.score for result in results.values()) / total_standards if total_standards > 0 else 0
        
        all_missing = []
        all_recommendations = []
        
        for result in results.values():
            all_missing.extend(result.missing_requirements)
            all_recommendations.extend(result.recommendations)
        
        return {
            "overall_compliant": compliant_standards == total_standards,
            "compliance_rate": compliant_standards / total_standards if total_standards > 0 else 0,
            "average_score": average_score,
            "compliant_standards": compliant_standards,
            "total_standards": total_standards,
            "total_missing_requirements": len(all_missing),
            "priority_recommendations": list(set(all_recommendations))[:5]  # Top 5 unique recommendations
        }