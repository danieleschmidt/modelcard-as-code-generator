"""Global-first deployment and internationalization capabilities."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .logging_config import get_logger
from .advanced_monitoring import metrics_collector

logger = get_logger(__name__)


class InternationalizationManager:
    """Comprehensive internationalization and localization support."""
    
    def __init__(self, default_locale: str = "en"):
        self.default_locale = default_locale
        self.translations: Dict[str, Dict[str, str]] = {}
        self.supported_locales: Set[str] = {"en", "es", "fr", "de", "ja", "zh", "pt", "it", "ru", "ko"}
        self.locale_formats: Dict[str, Dict[str, str]] = {}
        self.currency_symbols: Dict[str, str] = {}
        
        self._load_default_translations()
        self._load_locale_formats()
    
    def _load_default_translations(self) -> None:
        """Load default translations for model card templates."""
        self.translations = {
            "en": {
                "model_details": "Model Details",
                "model_name": "Model Name",
                "model_version": "Model Version",
                "model_description": "Model Description",
                "intended_use": "Intended Use",
                "training_data": "Training Data",
                "evaluation_results": "Evaluation Results",
                "limitations": "Limitations",
                "ethical_considerations": "Ethical Considerations",
                "bias_risks": "Bias Risks",
                "fairness_metrics": "Fairness Metrics",
                "carbon_footprint": "Carbon Footprint",
                "license": "License",
                "citation": "Citation",
                "contact": "Contact Information",
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1_score": "F1 Score",
                "performance_summary": "Performance Summary",
                "known_limitations": "Known Limitations",
                "out_of_scope": "Out of Scope Uses",
                "recommendations": "Recommendations"
            },
            "es": {
                "model_details": "Detalles del Modelo",
                "model_name": "Nombre del Modelo",
                "model_version": "Versión del Modelo",
                "model_description": "Descripción del Modelo",
                "intended_use": "Uso Previsto",
                "training_data": "Datos de Entrenamiento",
                "evaluation_results": "Resultados de Evaluación",
                "limitations": "Limitaciones",
                "ethical_considerations": "Consideraciones Éticas",
                "bias_risks": "Riesgos de Sesgo",
                "fairness_metrics": "Métricas de Equidad",
                "carbon_footprint": "Huella de Carbono",
                "license": "Licencia",
                "citation": "Cita",
                "contact": "Información de Contacto",
                "accuracy": "Precisión",
                "precision": "Exactitud",
                "recall": "Exhaustividad",
                "f1_score": "Puntuación F1",
                "performance_summary": "Resumen de Rendimiento",
                "known_limitations": "Limitaciones Conocidas",
                "out_of_scope": "Usos Fuera del Alcance",
                "recommendations": "Recomendaciones"
            },
            "fr": {
                "model_details": "Détails du Modèle",
                "model_name": "Nom du Modèle",
                "model_version": "Version du Modèle",
                "model_description": "Description du Modèle",
                "intended_use": "Utilisation Prévue",
                "training_data": "Données d'Entraînement",
                "evaluation_results": "Résultats d'Évaluation",
                "limitations": "Limitations",
                "ethical_considerations": "Considérations Éthiques",
                "bias_risks": "Risques de Biais",
                "fairness_metrics": "Métriques d'Équité",
                "carbon_footprint": "Empreinte Carbone",
                "license": "Licence",
                "citation": "Citation",
                "contact": "Informations de Contact",
                "accuracy": "Précision",
                "precision": "Exactitude",
                "recall": "Rappel",
                "f1_score": "Score F1",
                "performance_summary": "Résumé des Performances",
                "known_limitations": "Limitations Connues",
                "out_of_scope": "Utilisations Hors Portée",
                "recommendations": "Recommandations"
            },
            "de": {
                "model_details": "Modelldetails",
                "model_name": "Modellname",
                "model_version": "Modellversion",
                "model_description": "Modellbeschreibung",
                "intended_use": "Beabsichtigte Verwendung",
                "training_data": "Trainingsdaten",
                "evaluation_results": "Evaluierungsergebnisse",
                "limitations": "Einschränkungen",
                "ethical_considerations": "Ethische Überlegungen",
                "bias_risks": "Verzerrungsrisiken",
                "fairness_metrics": "Fairness-Metriken",
                "carbon_footprint": "CO₂-Fußabdruck",
                "license": "Lizenz",
                "citation": "Zitation",
                "contact": "Kontaktinformationen",
                "accuracy": "Genauigkeit",
                "precision": "Präzision",
                "recall": "Trefferquote",
                "f1_score": "F1-Score",
                "performance_summary": "Leistungszusammenfassung",
                "known_limitations": "Bekannte Einschränkungen",
                "out_of_scope": "Außerhalb des Anwendungsbereichs",
                "recommendations": "Empfehlungen"
            },
            "ja": {
                "model_details": "モデル詳細",
                "model_name": "モデル名",
                "model_version": "モデルバージョン",
                "model_description": "モデル説明",
                "intended_use": "意図された用途",
                "training_data": "トレーニングデータ",
                "evaluation_results": "評価結果",
                "limitations": "制限事項",
                "ethical_considerations": "倫理的考慮事項",
                "bias_risks": "バイアスリスク",
                "fairness_metrics": "公平性メトリクス",
                "carbon_footprint": "カーボンフットプリント",
                "license": "ライセンス",
                "citation": "引用",
                "contact": "連絡先情報",
                "accuracy": "精度",
                "precision": "適合率",
                "recall": "再現率",
                "f1_score": "F1スコア",
                "performance_summary": "パフォーマンス要約",
                "known_limitations": "既知の制限事項",
                "out_of_scope": "範囲外の用途",
                "recommendations": "推奨事項"
            },
            "zh": {
                "model_details": "模型详情",
                "model_name": "模型名称",
                "model_version": "模型版本",
                "model_description": "模型描述",
                "intended_use": "预期用途",
                "training_data": "训练数据",
                "evaluation_results": "评估结果",
                "limitations": "限制",
                "ethical_considerations": "伦理考虑",
                "bias_risks": "偏见风险",
                "fairness_metrics": "公平性指标",
                "carbon_footprint": "碳足迹",
                "license": "许可证",
                "citation": "引用",
                "contact": "联系信息",
                "accuracy": "准确率",
                "precision": "精确率",
                "recall": "召回率",
                "f1_score": "F1分数",
                "performance_summary": "性能摘要",
                "known_limitations": "已知限制",
                "out_of_scope": "超出范围的用途",
                "recommendations": "建议"
            }
        }
    
    def _load_locale_formats(self) -> None:
        """Load locale-specific formatting rules."""
        self.locale_formats = {
            "en": {"date": "%Y-%m-%d", "datetime": "%Y-%m-%d %H:%M:%S", "decimal": ".", "thousands": ","},
            "es": {"date": "%d/%m/%Y", "datetime": "%d/%m/%Y %H:%M:%S", "decimal": ",", "thousands": "."},
            "fr": {"date": "%d/%m/%Y", "datetime": "%d/%m/%Y %H:%M:%S", "decimal": ",", "thousands": " "},
            "de": {"date": "%d.%m.%Y", "datetime": "%d.%m.%Y %H:%M:%S", "decimal": ",", "thousands": "."},
            "ja": {"date": "%Y年%m月%d日", "datetime": "%Y年%m月%d日 %H:%M:%S", "decimal": ".", "thousands": ","},
            "zh": {"date": "%Y年%m月%d日", "datetime": "%Y年%m月%d日 %H:%M:%S", "decimal": ".", "thousands": ","}
        }
        
        self.currency_symbols = {
            "en": "$", "es": "€", "fr": "€", "de": "€", "ja": "¥", "zh": "¥",
            "pt": "R$", "it": "€", "ru": "₽", "ko": "₩"
        }
    
    def translate(self, key: str, locale: str = None, fallback: str = None) -> str:
        """Translate a key to the specified locale."""
        locale = locale or self.default_locale
        
        if locale in self.translations and key in self.translations[locale]:
            return self.translations[locale][key]
        
        # Try default locale
        if locale != self.default_locale and key in self.translations[self.default_locale]:
            return self.translations[self.default_locale][key]
        
        # Return fallback or key itself
        return fallback or key
    
    def format_date(self, date_obj: datetime, locale: str = None) -> str:
        """Format date according to locale conventions."""
        locale = locale or self.default_locale
        format_str = self.locale_formats.get(locale, self.locale_formats["en"])["date"]
        return date_obj.strftime(format_str)
    
    def format_number(self, number: float, locale: str = None, decimal_places: int = 2) -> str:
        """Format number according to locale conventions."""
        locale = locale or self.default_locale
        formats = self.locale_formats.get(locale, self.locale_formats["en"])
        
        # Format with decimal places
        formatted = f"{number:.{decimal_places}f}"
        
        # Split into integer and decimal parts
        parts = formatted.split(".")
        integer_part = parts[0]
        decimal_part = parts[1] if len(parts) > 1 else ""
        
        # Add thousands separators
        if len(integer_part) > 3:
            thousands_sep = formats["thousands"]
            integer_part = thousands_sep.join([
                integer_part[max(0, i-3):i] for i in range(len(integer_part), 0, -3)
            ][::-1])
        
        # Combine with decimal separator
        if decimal_part and int(decimal_part) > 0:
            decimal_sep = formats["decimal"]
            return f"{integer_part}{decimal_sep}{decimal_part}"
        else:
            return integer_part
    
    def get_supported_locales(self) -> List[str]:
        """Get list of supported locales."""
        return sorted(list(self.supported_locales))
    
    def add_custom_translation(self, locale: str, key: str, translation: str) -> None:
        """Add custom translation."""
        if locale not in self.translations:
            self.translations[locale] = {}
        self.translations[locale][key] = translation
        self.supported_locales.add(locale)
    
    def load_translations_from_file(self, file_path: Path) -> None:
        """Load translations from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                for locale, translations_dict in translations.items():
                    if locale not in self.translations:
                        self.translations[locale] = {}
                    self.translations[locale].update(translations_dict)
                    self.supported_locales.add(locale)
            logger.info(f"Loaded translations from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load translations from {file_path}: {e}")


class ComplianceFramework:
    """Multi-region compliance framework for global deployment."""
    
    def __init__(self):
        self.frameworks = {
            "GDPR": {
                "regions": ["EU", "EEA"],
                "requirements": [
                    "data_minimization",
                    "consent_management",
                    "right_to_erasure",
                    "data_portability",
                    "privacy_by_design",
                    "data_protection_impact_assessment"
                ],
                "documentation": [
                    "privacy_policy",
                    "data_processing_records",
                    "consent_records",
                    "breach_notification_procedures"
                ]
            },
            "CCPA": {
                "regions": ["California", "US"],
                "requirements": [
                    "consumer_disclosure",
                    "opt_out_mechanisms",
                    "data_deletion_rights",
                    "non_discrimination",
                    "privacy_rights_notice"
                ],
                "documentation": [
                    "privacy_notice",
                    "data_collection_disclosure",
                    "consumer_request_procedures"
                ]
            },
            "PDPA": {
                "regions": ["Singapore", "Thailand"],
                "requirements": [
                    "consent_requirements",
                    "data_breach_notification",
                    "data_protection_measures",
                    "cross_border_transfers"
                ],
                "documentation": [
                    "privacy_notice",
                    "data_protection_policies",
                    "breach_response_plan"
                ]
            },
            "EU_AI_ACT": {
                "regions": ["EU"],
                "requirements": [
                    "risk_assessment",
                    "transparency_requirements",
                    "human_oversight",
                    "accuracy_robustness",
                    "bias_monitoring",
                    "quality_management_system"
                ],
                "documentation": [
                    "ai_system_documentation",
                    "risk_assessment_report",
                    "conformity_assessment",
                    "quality_management_documentation"
                ]
            },
            "ISO_27001": {
                "regions": ["Global"],
                "requirements": [
                    "information_security_policy",
                    "risk_management",
                    "access_control",
                    "cryptography",
                    "incident_management",
                    "business_continuity"
                ],
                "documentation": [
                    "isms_documentation",
                    "security_policies",
                    "risk_register",
                    "incident_response_plan"
                ]
            }
        }
        
        self.regional_requirements = {
            "EU": ["GDPR", "EU_AI_ACT", "ISO_27001"],
            "US": ["CCPA", "ISO_27001"],
            "APAC": ["PDPA", "ISO_27001"],
            "Global": ["ISO_27001"]
        }
    
    def get_applicable_frameworks(self, regions: List[str]) -> Dict[str, Any]:
        """Get applicable compliance frameworks for target regions."""
        applicable = {}
        
        for region in regions:
            if region in self.regional_requirements:
                for framework in self.regional_requirements[region]:
                    if framework not in applicable:
                        applicable[framework] = self.frameworks[framework].copy()
                        applicable[framework]["applicable_regions"] = []
                    applicable[framework]["applicable_regions"].append(region)
        
        return applicable
    
    def generate_compliance_checklist(self, regions: List[str]) -> Dict[str, Any]:
        """Generate compliance checklist for target regions."""
        frameworks = self.get_applicable_frameworks(regions)
        
        checklist = {
            "frameworks": list(frameworks.keys()),
            "requirements": {},
            "documentation": {},
            "assessment": {
                "total_requirements": 0,
                "completed": 0,
                "compliance_score": 0.0
            }
        }
        
        for framework_name, framework in frameworks.items():
            for requirement in framework["requirements"]:
                if requirement not in checklist["requirements"]:
                    checklist["requirements"][requirement] = {
                        "status": "pending",
                        "frameworks": [],
                        "priority": "medium"
                    }
                checklist["requirements"][requirement]["frameworks"].append(framework_name)
            
            for doc in framework["documentation"]:
                if doc not in checklist["documentation"]:
                    checklist["documentation"][doc] = {
                        "status": "pending",
                        "frameworks": [],
                        "required_by": []
                    }
                checklist["documentation"][doc]["frameworks"].append(framework_name)
        
        checklist["assessment"]["total_requirements"] = len(checklist["requirements"])
        
        return checklist
    
    def validate_compliance(self, model_card_data: Dict[str, Any], regions: List[str]) -> Dict[str, Any]:
        """Validate model card compliance with regional frameworks."""
        frameworks = self.get_applicable_frameworks(regions)
        validation_results = {
            "overall_compliance": True,
            "framework_results": {},
            "missing_requirements": [],
            "recommendations": []
        }
        
        for framework_name, framework in frameworks.items():
            framework_result = {
                "compliant": True,
                "missing_requirements": [],
                "score": 0.0
            }
            
            required_fields = self._get_required_fields(framework_name)
            present_fields = 0
            
            for field in required_fields:
                if self._is_field_present(model_card_data, field):
                    present_fields += 1
                else:
                    framework_result["missing_requirements"].append(field)
                    framework_result["compliant"] = False
                    validation_results["overall_compliance"] = False
            
            framework_result["score"] = present_fields / len(required_fields) if required_fields else 1.0
            validation_results["framework_results"][framework_name] = framework_result
        
        # Generate recommendations
        validation_results["recommendations"] = self._generate_compliance_recommendations(validation_results)
        
        return validation_results
    
    def _get_required_fields(self, framework: str) -> List[str]:
        """Get required fields for compliance framework."""
        field_mappings = {
            "GDPR": ["data_sources", "data_processing_purpose", "legal_basis", "retention_period"],
            "CCPA": ["data_categories", "business_purpose", "third_party_sharing"],
            "PDPA": ["consent_obtained", "data_protection_measures", "transfer_restrictions"],
            "EU_AI_ACT": ["risk_level", "intended_purpose", "human_oversight", "bias_assessment"],
            "ISO_27001": ["security_measures", "access_controls", "incident_procedures"]
        }
        return field_mappings.get(framework, [])
    
    def _is_field_present(self, data: Dict[str, Any], field: str) -> bool:
        """Check if required field is present in model card data."""
        # Simplified field checking - in production this would be more sophisticated
        field_variations = {
            "data_sources": ["training_data", "datasets", "data_sources"],
            "intended_purpose": ["intended_use", "use_cases", "purpose"],
            "bias_assessment": ["bias_risks", "fairness_metrics", "ethical_considerations"]
        }
        
        check_fields = field_variations.get(field, [field])
        
        for check_field in check_fields:
            if check_field in data and data[check_field]:
                return True
        
        return False
    
    def _generate_compliance_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving compliance."""
        recommendations = []
        
        for framework, result in validation_results["framework_results"].items():
            if not result["compliant"]:
                missing = len(result["missing_requirements"])
                recommendations.append(
                    f"Add {missing} missing requirements for {framework} compliance: "
                    f"{', '.join(result['missing_requirements'][:3])}"
                    f"{'...' if missing > 3 else ''}"
                )
        
        if not validation_results["overall_compliance"]:
            recommendations.append("Consider implementing automated compliance checking in your CI/CD pipeline")
            recommendations.append("Review data collection and processing practices for privacy compliance")
        
        return recommendations


class GlobalDeploymentManager:
    """Manage global deployment with region-specific optimizations."""
    
    def __init__(self):
        self.i18n = InternationalizationManager()
        self.compliance = ComplianceFramework()
        self.regions = {
            "us-east-1": {"continent": "NA", "compliance": ["CCPA"], "locale": "en"},
            "eu-west-1": {"continent": "EU", "compliance": ["GDPR", "EU_AI_ACT"], "locale": "en"},
            "eu-central-1": {"continent": "EU", "compliance": ["GDPR", "EU_AI_ACT"], "locale": "de"},
            "ap-southeast-1": {"continent": "APAC", "compliance": ["PDPA"], "locale": "en"},
            "ap-northeast-1": {"continent": "APAC", "compliance": ["PDPA"], "locale": "ja"},
            "ap-east-1": {"continent": "APAC", "compliance": ["PDPA"], "locale": "zh"}
        }
    
    async def prepare_global_deployment(self, model_card_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare model card for global deployment."""
        deployment_plan = {
            "regions": {},
            "compliance_status": {},
            "localization_status": {},
            "deployment_readiness": {}
        }
        
        for region, config in self.regions.items():
            # Prepare region-specific deployment
            region_deployment = await self._prepare_region_deployment(
                region, config, model_card_data
            )
            deployment_plan["regions"][region] = region_deployment
            
            # Check compliance
            compliance_result = self.compliance.validate_compliance(
                model_card_data, [config["continent"]]
            )
            deployment_plan["compliance_status"][region] = compliance_result
            
            # Check localization
            localization_status = self._check_localization_status(
                model_card_data, config["locale"]
            )
            deployment_plan["localization_status"][region] = localization_status
            
            # Calculate deployment readiness
            readiness_score = self._calculate_deployment_readiness(
                compliance_result, localization_status
            )
            deployment_plan["deployment_readiness"][region] = readiness_score
        
        # Generate deployment summary
        deployment_plan["summary"] = self._generate_deployment_summary(deployment_plan)
        
        return deployment_plan
    
    async def _prepare_region_deployment(
        self, region: str, config: Dict[str, Any], model_card_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare deployment for specific region."""
        return {
            "region": region,
            "locale": config["locale"],
            "compliance_frameworks": config["compliance"],
            "cdn_endpoints": self._get_cdn_endpoints(region),
            "data_residency": self._get_data_residency_requirements(region),
            "monitoring_config": self._get_monitoring_config(region),
            "backup_regions": self._get_backup_regions(region)
        }
    
    def _check_localization_status(self, model_card_data: Dict[str, Any], locale: str) -> Dict[str, Any]:
        """Check localization status for locale."""
        return {
            "locale": locale,
            "translation_coverage": 0.95,  # Simplified - would check actual translations
            "date_formats": "configured",
            "number_formats": "configured",
            "currency_support": locale in self.i18n.currency_symbols,
            "rtl_support": locale in ["ar", "he", "fa"]
        }
    
    def _calculate_deployment_readiness(
        self, compliance_result: Dict[str, Any], localization_status: Dict[str, Any]
    ) -> float:
        """Calculate deployment readiness score."""
        compliance_score = sum(
            result["score"] for result in compliance_result["framework_results"].values()
        ) / len(compliance_result["framework_results"]) if compliance_result["framework_results"] else 1.0
        
        localization_score = localization_status["translation_coverage"]
        
        # Weighted average
        readiness_score = (compliance_score * 0.7) + (localization_score * 0.3)
        
        return readiness_score
    
    def _generate_deployment_summary(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate deployment summary."""
        total_regions = len(deployment_plan["regions"])
        ready_regions = sum(
            1 for score in deployment_plan["deployment_readiness"].values()
            if score >= 0.9
        )
        
        avg_readiness = sum(deployment_plan["deployment_readiness"].values()) / total_regions
        
        return {
            "total_regions": total_regions,
            "ready_regions": ready_regions,
            "avg_readiness_score": avg_readiness,
            "deployment_recommendation": (
                "ready" if avg_readiness >= 0.9 else
                "needs_improvement" if avg_readiness >= 0.7 else
                "not_ready"
            ),
            "next_steps": self._get_next_steps(deployment_plan)
        }
    
    def _get_cdn_endpoints(self, region: str) -> List[str]:
        """Get CDN endpoints for region."""
        return [f"https://cdn-{region}.example.com", f"https://backup-cdn-{region}.example.com"]
    
    def _get_data_residency_requirements(self, region: str) -> Dict[str, Any]:
        """Get data residency requirements for region."""
        return {
            "data_must_stay_in_region": region.startswith("eu-"),
            "encryption_required": True,
            "audit_logging": True,
            "retention_period_days": 365 if region.startswith("eu-") else 180
        }
    
    def _get_monitoring_config(self, region: str) -> Dict[str, Any]:
        """Get monitoring configuration for region."""
        return {
            "metrics_endpoint": f"https://metrics-{region}.example.com",
            "log_aggregation": f"https://logs-{region}.example.com",
            "alerting_webhook": f"https://alerts-{region}.example.com",
            "health_check_interval": 30
        }
    
    def _get_backup_regions(self, region: str) -> List[str]:
        """Get backup regions for failover."""
        backup_map = {
            "us-east-1": ["us-west-2", "us-central-1"],
            "eu-west-1": ["eu-central-1", "eu-north-1"],
            "ap-southeast-1": ["ap-northeast-1", "ap-south-1"]
        }
        return backup_map.get(region, [])
    
    def _get_next_steps(self, deployment_plan: Dict[str, Any]) -> List[str]:
        """Get recommended next steps for deployment."""
        steps = []
        
        # Check compliance issues
        non_compliant_regions = [
            region for region, status in deployment_plan["compliance_status"].items()
            if not status["overall_compliance"]
        ]
        
        if non_compliant_regions:
            steps.append(f"Address compliance issues in {len(non_compliant_regions)} regions")
        
        # Check readiness scores
        low_readiness_regions = [
            region for region, score in deployment_plan["deployment_readiness"].items()
            if score < 0.9
        ]
        
        if low_readiness_regions:
            steps.append(f"Improve deployment readiness in {len(low_readiness_regions)} regions")
        
        # Check localization
        incomplete_localization = [
            region for region, status in deployment_plan["localization_status"].items()
            if status["translation_coverage"] < 0.95
        ]
        
        if incomplete_localization:
            steps.append(f"Complete localization for {len(incomplete_localization)} regions")
        
        if not steps:
            steps.append("All regions ready for deployment")
        
        return steps


# Global instances
i18n_manager = InternationalizationManager()
compliance_framework = ComplianceFramework()
global_deployment_manager = GlobalDeploymentManager()