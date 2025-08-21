#!/usr/bin/env python3
"""Global-First Implementation: Multi-region and i18n support."""

import json
import os
from pathlib import Path

def create_i18n_support():
    """Create internationalization support."""
    
    print("🌍 Implementing Global-First Features")
    print("="*50)
    
    # Create i18n directory structure
    i18n_dir = Path("src/modelcard_generator/i18n")
    i18n_dir.mkdir(exist_ok=True)
    
    # Create translation files for major languages
    translations = {
        "en": {
            "model_card": {
                "title": "Model Card",
                "model_details": "Model Details",
                "intended_use": "Intended Use",
                "training_details": "Training Details",
                "evaluation_results": "Evaluation Results",
                "ethical_considerations": "Ethical Considerations",
                "limitations": "Limitations",
                "bias_risks": "Bias Risks",
                "bias_mitigation": "Bias Mitigation",
                "fairness_metrics": "Fairness Metrics",
                "version": "Version",
                "authors": "Authors",
                "license": "License",
                "description": "Description",
                "framework": "Framework",
                "architecture": "Architecture",
                "hyperparameters": "Hyperparameters",
                "training_data": "Training Data",
                "preprocessing": "Preprocessing"
            },
            "validation": {
                "required_field_missing": "Required field '{field}' is missing or empty",
                "invalid_type": "Invalid type for field '{field}'",
                "too_short": "Content is too short",
                "too_long": "Content is too long",
                "invalid_format": "Invalid format",
                "security_issue": "Security issue detected",
                "bias_documentation_missing": "Bias documentation is missing",
                "compliance_issue": "Compliance issue detected"
            },
            "cli": {
                "generating": "Generating model card...",
                "validating": "Validating model card...",
                "saving": "Saving model card to {path}",
                "success": "Model card generated successfully",
                "error": "Error: {message}",
                "help": "Show this help message",
                "version": "Show version information"
            }
        },
        "es": {
            "model_card": {
                "title": "Tarjeta del Modelo",
                "model_details": "Detalles del Modelo",
                "intended_use": "Uso Previsto",
                "training_details": "Detalles del Entrenamiento",
                "evaluation_results": "Resultados de Evaluación",
                "ethical_considerations": "Consideraciones Éticas",
                "limitations": "Limitaciones",
                "bias_risks": "Riesgos de Sesgo",
                "bias_mitigation": "Mitigación de Sesgo",
                "fairness_metrics": "Métricas de Equidad",
                "version": "Versión",
                "authors": "Autores",
                "license": "Licencia",
                "description": "Descripción",
                "framework": "Marco de Trabajo",
                "architecture": "Arquitectura",
                "hyperparameters": "Hiperparámetros",
                "training_data": "Datos de Entrenamiento",
                "preprocessing": "Preprocesamiento"
            },
            "validation": {
                "required_field_missing": "El campo requerido '{field}' falta o está vacío",
                "invalid_type": "Tipo inválido para el campo '{field}'",
                "too_short": "El contenido es demasiado corto",
                "too_long": "El contenido es demasiado largo",
                "invalid_format": "Formato inválido",
                "security_issue": "Problema de seguridad detectado",
                "bias_documentation_missing": "Falta documentación de sesgo",
                "compliance_issue": "Problema de cumplimiento detectado"
            },
            "cli": {
                "generating": "Generando tarjeta del modelo...",
                "validating": "Validando tarjeta del modelo...",
                "saving": "Guardando tarjeta del modelo en {path}",
                "success": "Tarjeta del modelo generada exitosamente",
                "error": "Error: {message}",
                "help": "Mostrar este mensaje de ayuda",
                "version": "Mostrar información de versión"
            }
        },
        "fr": {
            "model_card": {
                "title": "Carte de Modèle",
                "model_details": "Détails du Modèle",
                "intended_use": "Utilisation Prévue",
                "training_details": "Détails d'Entraînement",
                "evaluation_results": "Résultats d'Évaluation",
                "ethical_considerations": "Considérations Éthiques",
                "limitations": "Limitations",
                "bias_risks": "Risques de Biais",
                "bias_mitigation": "Atténuation des Biais",
                "fairness_metrics": "Métriques d'Équité",
                "version": "Version",
                "authors": "Auteurs",
                "license": "Licence",
                "description": "Description",
                "framework": "Framework",
                "architecture": "Architecture",
                "hyperparameters": "Hyperparamètres",
                "training_data": "Données d'Entraînement",
                "preprocessing": "Prétraitement"
            },
            "validation": {
                "required_field_missing": "Le champ requis '{field}' est manquant ou vide",
                "invalid_type": "Type invalide pour le champ '{field}'",
                "too_short": "Le contenu est trop court",
                "too_long": "Le contenu est trop long",
                "invalid_format": "Format invalide",
                "security_issue": "Problème de sécurité détecté",
                "bias_documentation_missing": "Documentation des biais manquante",
                "compliance_issue": "Problème de conformité détecté"
            },
            "cli": {
                "generating": "Génération de la carte de modèle...",
                "validating": "Validation de la carte de modèle...",
                "saving": "Sauvegarde de la carte de modèle vers {path}",
                "success": "Carte de modèle générée avec succès",
                "error": "Erreur: {message}",
                "help": "Afficher ce message d'aide",
                "version": "Afficher les informations de version"
            }
        },
        "de": {
            "model_card": {
                "title": "Modellkarte",
                "model_details": "Modelldetails",
                "intended_use": "Vorgesehene Verwendung",
                "training_details": "Trainingsdetails",
                "evaluation_results": "Bewertungsergebnisse",
                "ethical_considerations": "Ethische Überlegungen",
                "limitations": "Einschränkungen",
                "bias_risks": "Verzerrungsrisiken",
                "bias_mitigation": "Verzerrungsminderung",
                "fairness_metrics": "Fairness-Metriken",
                "version": "Version",
                "authors": "Autoren",
                "license": "Lizenz",
                "description": "Beschreibung",
                "framework": "Framework",
                "architecture": "Architektur",
                "hyperparameters": "Hyperparameter",
                "training_data": "Trainingsdaten",
                "preprocessing": "Vorverarbeitung"
            },
            "validation": {
                "required_field_missing": "Erforderliches Feld '{field}' fehlt oder ist leer",
                "invalid_type": "Ungültiger Typ für Feld '{field}'",
                "too_short": "Inhalt ist zu kurz",
                "too_long": "Inhalt ist zu lang",
                "invalid_format": "Ungültiges Format",
                "security_issue": "Sicherheitsproblem erkannt",
                "bias_documentation_missing": "Verzerrungsdokumentation fehlt",
                "compliance_issue": "Compliance-Problem erkannt"
            },
            "cli": {
                "generating": "Modellkarte wird generiert...",
                "validating": "Modellkarte wird validiert...",
                "saving": "Modellkarte wird gespeichert unter {path}",
                "success": "Modellkarte erfolgreich generiert",
                "error": "Fehler: {message}",
                "help": "Diese Hilfenachricht anzeigen",
                "version": "Versionsinformationen anzeigen"
            }
        },
        "ja": {
            "model_card": {
                "title": "モデルカード",
                "model_details": "モデル詳細",
                "intended_use": "意図された用途",
                "training_details": "訓練詳細",
                "evaluation_results": "評価結果",
                "ethical_considerations": "倫理的考慮事項",
                "limitations": "制限事項",
                "bias_risks": "バイアスリスク",
                "bias_mitigation": "バイアス軽減",
                "fairness_metrics": "公平性メトリクス",
                "version": "バージョン",
                "authors": "作者",
                "license": "ライセンス",
                "description": "説明",
                "framework": "フレームワーク",
                "architecture": "アーキテクチャ",
                "hyperparameters": "ハイパーパラメータ",
                "training_data": "訓練データ",
                "preprocessing": "前処理"
            },
            "validation": {
                "required_field_missing": "必須フィールド '{field}' が不足または空です",
                "invalid_type": "フィールド '{field}' の型が無効です",
                "too_short": "内容が短すぎます",
                "too_long": "内容が長すぎます",
                "invalid_format": "無効な形式",
                "security_issue": "セキュリティ問題が検出されました",
                "bias_documentation_missing": "バイアス文書が不足しています",
                "compliance_issue": "コンプライアンス問題が検出されました"
            },
            "cli": {
                "generating": "モデルカードを生成中...",
                "validating": "モデルカードを検証中...",
                "saving": "モデルカードを{path}に保存中",
                "success": "モデルカードが正常に生成されました",
                "error": "エラー: {message}",
                "help": "このヘルプメッセージを表示",
                "version": "バージョン情報を表示"
            }
        },
        "zh": {
            "model_card": {
                "title": "模型卡",
                "model_details": "模型详情",
                "intended_use": "预期用途",
                "training_details": "训练详情",
                "evaluation_results": "评估结果",
                "ethical_considerations": "伦理考虑",
                "limitations": "限制",
                "bias_risks": "偏见风险",
                "bias_mitigation": "偏见缓解",
                "fairness_metrics": "公平性指标",
                "version": "版本",
                "authors": "作者",
                "license": "许可证",
                "description": "描述",
                "framework": "框架",
                "architecture": "架构",
                "hyperparameters": "超参数",
                "training_data": "训练数据",
                "preprocessing": "预处理"
            },
            "validation": {
                "required_field_missing": "必填字段 '{field}' 缺失或为空",
                "invalid_type": "字段 '{field}' 类型无效",
                "too_short": "内容太短",
                "too_long": "内容太长",
                "invalid_format": "格式无效",
                "security_issue": "检测到安全问题",
                "bias_documentation_missing": "缺少偏见文档",
                "compliance_issue": "检测到合规问题"
            },
            "cli": {
                "generating": "正在生成模型卡...",
                "validating": "正在验证模型卡...",
                "saving": "正在将模型卡保存到 {path}",
                "success": "模型卡生成成功",
                "error": "错误: {message}",
                "help": "显示此帮助信息",
                "version": "显示版本信息"
            }
        }
    }
    
    # Save translation files
    for lang_code, translations_data in translations.items():
        lang_file = i18n_dir / f"{lang_code}.json"
        with open(lang_file, "w", encoding="utf-8") as f:
            json.dump(translations_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Created translation file: {lang_file}")
    
    return i18n_dir


def create_i18n_module():
    """Create i18n module for translations."""
    
    i18n_module = Path("src/modelcard_generator/i18n/__init__.py")
    
    module_content = '''"""Internationalization (i18n) support for model card generator."""

import json
import os
from pathlib import Path
from typing import Dict, Optional

DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "ja", "zh"]

_translations: Dict[str, Dict] = {}
_current_language = DEFAULT_LANGUAGE


def load_translations() -> None:
    """Load all translation files."""
    global _translations
    
    i18n_dir = Path(__file__).parent
    
    for lang_code in SUPPORTED_LANGUAGES:
        lang_file = i18n_dir / f"{lang_code}.json"
        if lang_file.exists():
            try:
                with open(lang_file, "r", encoding="utf-8") as f:
                    _translations[lang_code] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load translations for {lang_code}: {e}")


def set_language(lang_code: str) -> None:
    """Set the current language."""
    global _current_language
    
    if lang_code in SUPPORTED_LANGUAGES:
        _current_language = lang_code
    else:
        print(f"Warning: Language {lang_code} not supported, using {DEFAULT_LANGUAGE}")
        _current_language = DEFAULT_LANGUAGE


def get_language() -> str:
    """Get the current language."""
    return _current_language


def _(key: str, **kwargs) -> str:
    """Get translated text."""
    if not _translations:
        load_translations()
    
    # Navigate through nested keys (e.g., "model_card.title")
    keys = key.split(".")
    translation = _translations.get(_current_language, {})
    
    for k in keys:
        if isinstance(translation, dict) and k in translation:
            translation = translation[k]
        else:
            # Fallback to English
            fallback = _translations.get(DEFAULT_LANGUAGE, {})
            for k in keys:
                if isinstance(fallback, dict) and k in fallback:
                    fallback = fallback[k]
                else:
                    return key  # Return key if no translation found
            return str(fallback).format(**kwargs) if kwargs else str(fallback)
    
    return str(translation).format(**kwargs) if kwargs else str(translation)


def get_supported_languages() -> list:
    """Get list of supported languages."""
    return SUPPORTED_LANGUAGES.copy()


def detect_system_language() -> str:
    """Detect system language from environment."""
    # Check environment variables
    for var in ["LANG", "LANGUAGE", "LC_ALL", "LC_MESSAGES"]:
        lang = os.environ.get(var)
        if lang:
            # Extract language code (e.g., "en_US.UTF-8" -> "en")
            lang_code = lang.split("_")[0].split(".")[0].lower()
            if lang_code in SUPPORTED_LANGUAGES:
                return lang_code
    
    return DEFAULT_LANGUAGE


# Auto-detect language on import
auto_detected = detect_system_language()
set_language(auto_detected)

# Load translations on import
load_translations()


class LocalizedModelCard:
    """Model card with localization support."""
    
    def __init__(self, language: Optional[str] = None):
        if language:
            set_language(language)
        self.language = get_language()
    
    def get_section_title(self, section: str) -> str:
        """Get localized section title."""
        return _(f"model_card.{section}")
    
    def get_validation_message(self, message_key: str, **kwargs) -> str:
        """Get localized validation message."""
        return _(f"validation.{message_key}", **kwargs)
    
    def get_cli_message(self, message_key: str, **kwargs) -> str:
        """Get localized CLI message."""
        return _(f"cli.{message_key}", **kwargs)


# Global instance
localized_card = LocalizedModelCard()
'''
    
    with open(i18n_module, "w", encoding="utf-8") as f:
        f.write(module_content)
    
    print(f"✅ Created i18n module: {i18n_module}")
    return i18n_module


def create_global_deployment_config():
    """Create global deployment configuration."""
    
    deployment_dir = Path("deployment/global")
    deployment_dir.mkdir(parents=True, exist_ok=True)
    
    # Multi-region deployment configuration
    global_config = {
        "regions": {
            "us-east-1": {
                "name": "US East (N. Virginia)",
                "languages": ["en", "es"],
                "compliance": ["CCPA", "GDPR"],
                "data_residency": "US",
                "endpoints": [
                    "https://api-us-east.modelcard-generator.com"
                ]
            },
            "eu-west-1": {
                "name": "EU West (Ireland)",
                "languages": ["en", "fr", "de"],
                "compliance": ["GDPR", "EU_AI_ACT"],
                "data_residency": "EU",
                "endpoints": [
                    "https://api-eu-west.modelcard-generator.com"
                ]
            },
            "ap-northeast-1": {
                "name": "Asia Pacific (Tokyo)",
                "languages": ["en", "ja"],
                "compliance": ["PDPA"],
                "data_residency": "APAC",
                "endpoints": [
                    "https://api-ap-northeast.modelcard-generator.com"
                ]
            },
            "ap-southeast-1": {
                "name": "Asia Pacific (Singapore)",
                "languages": ["en", "zh"],
                "compliance": ["PDPA", "CYBERSECURITY_ACT"],
                "data_residency": "APAC",
                "endpoints": [
                    "https://api-ap-southeast.modelcard-generator.com"
                ]
            }
        },
        "compliance_frameworks": {
            "GDPR": {
                "required_fields": [
                    "data_protection_measures",
                    "consent_mechanism",
                    "data_retention_policy",
                    "right_to_erasure"
                ],
                "validation_rules": [
                    "personal_data_documentation",
                    "privacy_impact_assessment",
                    "data_processor_agreements"
                ]
            },
            "CCPA": {
                "required_fields": [
                    "personal_information_categories",
                    "data_selling_disclosure",
                    "consumer_rights"
                ],
                "validation_rules": [
                    "consumer_opt_out",
                    "data_deletion_requests"
                ]
            },
            "EU_AI_ACT": {
                "required_fields": [
                    "ai_system_classification",
                    "risk_assessment",
                    "conformity_assessment",
                    "human_oversight"
                ],
                "validation_rules": [
                    "high_risk_requirements",
                    "transparency_obligations",
                    "accuracy_requirements"
                ]
            },
            "PDPA": {
                "required_fields": [
                    "personal_data_handling",
                    "consent_withdrawal",
                    "data_breach_notification"
                ],
                "validation_rules": [
                    "purpose_limitation",
                    "data_minimization"
                ]
            }
        },
        "data_residency": {
            "US": {
                "allowed_regions": ["us-east-1", "us-west-1", "us-west-2"],
                "encryption": "required",
                "audit_logging": "required"
            },
            "EU": {
                "allowed_regions": ["eu-west-1", "eu-central-1", "eu-north-1"],
                "encryption": "required",
                "audit_logging": "required",
                "schrems_ii_compliance": "required"
            },
            "APAC": {
                "allowed_regions": ["ap-northeast-1", "ap-southeast-1", "ap-south-1"],
                "encryption": "required",
                "audit_logging": "required"
            }
        }
    }
    
    config_file = deployment_dir / "global-config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(global_config, f, indent=2)
    
    print(f"✅ Created global deployment config: {config_file}")
    
    # Create region-specific Kubernetes manifests
    for region_id, region_config in global_config["regions"].items():
        region_dir = deployment_dir / region_id
        region_dir.mkdir(exist_ok=True)
        
        # Kubernetes deployment manifest
        k8s_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"modelcard-generator-{region_id}",
                "namespace": "modelcard-system",
                "labels": {
                    "app": "modelcard-generator",
                    "region": region_id,
                    "tier": "production"
                }
            },
            "spec": {
                "replicas": 3,
                "selector": {
                    "matchLabels": {
                        "app": "modelcard-generator",
                        "region": region_id
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "modelcard-generator",
                            "region": region_id
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "modelcard-generator",
                                "image": "modelcard-generator:latest",
                                "ports": [
                                    {
                                        "containerPort": 8080,
                                        "name": "http"
                                    }
                                ],
                                "env": [
                                    {
                                        "name": "REGION",
                                        "value": region_id
                                    },
                                    {
                                        "name": "DEFAULT_LANGUAGE",
                                        "value": region_config["languages"][0]
                                    },
                                    {
                                        "name": "SUPPORTED_LANGUAGES",
                                        "value": ",".join(region_config["languages"])
                                    },
                                    {
                                        "name": "COMPLIANCE_FRAMEWORKS",
                                        "value": ",".join(region_config["compliance"])
                                    },
                                    {
                                        "name": "DATA_RESIDENCY",
                                        "value": region_config["data_residency"]
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "memory": "512Mi",
                                        "cpu": "250m"
                                    },
                                    "limits": {
                                        "memory": "1Gi",
                                        "cpu": "500m"
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": 8080
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/ready",
                                        "port": 8080
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        k8s_file = region_dir / "deployment.yaml"
        import yaml
        with open(k8s_file, "w") as f:
            yaml.dump(k8s_manifest, f, default_flow_style=False)
        
        print(f"✅ Created {region_id} deployment manifest: {k8s_file}")
    
    return deployment_dir


def create_compliance_templates():
    """Create compliance-specific model card templates."""
    
    templates_dir = Path("src/modelcard_generator/templates/compliance")
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    # GDPR Compliance Template
    gdpr_template = {
        "name": "GDPR Compliant Model Card",
        "description": "Template for GDPR compliance requirements",
        "required_sections": [
            "model_details",
            "intended_use",
            "training_details",
            "evaluation_results",
            "ethical_considerations",
            "limitations",
            "data_protection_measures",
            "consent_mechanism",
            "data_retention_policy",
            "right_to_erasure"
        ],
        "template": {
            "model_details": {
                "name": "[Model Name]",
                "version": "[Version]",
                "description": "[Model Description]",
                "authors": ["[Author Names]"],
                "license": "[License]",
                "data_controller": "[Data Controller Information]",
                "data_protection_officer": "[DPO Contact Information]"
            },
            "data_protection_measures": {
                "encryption": "[Encryption methods used]",
                "access_controls": "[Access control measures]",
                "data_minimization": "[Data minimization practices]",
                "purpose_limitation": "[Purpose limitation documentation]"
            },
            "consent_mechanism": {
                "consent_type": "[Type of consent obtained]",
                "consent_withdrawal": "[Process for consent withdrawal]",
                "legal_basis": "[Legal basis for processing]"
            },
            "data_retention_policy": {
                "retention_period": "[Data retention period]",
                "deletion_process": "[Data deletion process]",
                "archival_policy": "[Data archival policy]"
            },
            "right_to_erasure": {
                "process": "[Process for handling erasure requests]",
                "timeline": "[Timeline for erasure]",
                "exceptions": "[Exceptions to erasure rights]"
            }
        }
    }
    
    with open(templates_dir / "gdpr_template.json", "w") as f:
        json.dump(gdpr_template, f, indent=2)
    
    # EU AI Act Template
    eu_ai_act_template = {
        "name": "EU AI Act Compliant Model Card",
        "description": "Template for EU AI Act compliance requirements",
        "required_sections": [
            "model_details",
            "ai_system_classification",
            "risk_assessment",
            "intended_use",
            "training_details",
            "evaluation_results",
            "conformity_assessment",
            "human_oversight",
            "transparency_obligations",
            "accuracy_requirements"
        ],
        "template": {
            "ai_system_classification": {
                "risk_level": "[minimal/limited/high/unacceptable]",
                "classification_rationale": "[Rationale for classification]",
                "applicable_requirements": "[List of applicable requirements]"
            },
            "risk_assessment": {
                "identified_risks": ["[List of identified risks]"],
                "risk_mitigation_measures": ["[List of mitigation measures]"],
                "residual_risks": ["[List of residual risks]"],
                "risk_monitoring": "[Risk monitoring procedures]"
            },
            "conformity_assessment": {
                "assessment_procedure": "[Conformity assessment procedure used]",
                "notified_body": "[Notified body information if applicable]",
                "ce_marking": "[CE marking information]",
                "declaration_of_conformity": "[Declaration of conformity details]"
            },
            "human_oversight": {
                "oversight_measures": "[Human oversight measures]",
                "human_intervention": "[Process for human intervention]",
                "monitoring_procedures": "[Human monitoring procedures]"
            },
            "transparency_obligations": {
                "user_information": "[Information provided to users]",
                "system_capabilities": "[System capabilities disclosure]",
                "system_limitations": "[System limitations disclosure]"
            },
            "accuracy_requirements": {
                "accuracy_metrics": "[Accuracy metrics and thresholds]",
                "testing_procedures": "[Testing and validation procedures]",
                "performance_monitoring": "[Performance monitoring measures]"
            }
        }
    }
    
    with open(templates_dir / "eu_ai_act_template.json", "w") as f:
        json.dump(eu_ai_act_template, f, indent=2)
    
    print(f"✅ Created compliance templates in: {templates_dir}")
    return templates_dir


def test_global_features():
    """Test global features implementation."""
    
    print("\n🧪 Testing Global Features")
    print("-" * 30)
    
    # Test i18n functionality
    try:
        from src.modelcard_generator.i18n import _, set_language, get_supported_languages
        
        # Test English (default)
        title_en = _("model_card.title")
        print(f"✅ English title: {title_en}")
        
        # Test Spanish
        set_language("es")
        title_es = _("model_card.title")
        print(f"✅ Spanish title: {title_es}")
        
        # Test French
        set_language("fr")
        title_fr = _("model_card.title")
        print(f"✅ French title: {title_fr}")
        
        # Test validation message with parameters
        set_language("en")
        validation_msg = _("validation.required_field_missing", field="model_name")
        print(f"✅ Validation message: {validation_msg}")
        
        # Test supported languages
        supported = get_supported_languages()
        print(f"✅ Supported languages: {supported}")
        
        return True
        
    except Exception as e:
        print(f"❌ Global features test failed: {e}")
        return False


if __name__ == "__main__":
    
    # Create i18n support
    i18n_dir = create_i18n_support()
    
    # Create i18n module
    i18n_module = create_i18n_module()
    
    # Create global deployment configuration
    deployment_dir = create_global_deployment_config()
    
    # Create compliance templates
    templates_dir = create_compliance_templates()
    
    # Test implementation
    success = test_global_features()
    
    print("\n🌍 Global-First Implementation Summary")
    print("="*50)
    print(f"✅ Internationalization: 6 languages supported (en, es, fr, de, ja, zh)")
    print(f"✅ Multi-region deployment: 4 regions configured")
    print(f"✅ Compliance frameworks: GDPR, CCPA, EU AI Act, PDPA")
    print(f"✅ Data residency: US, EU, APAC regions")
    print(f"✅ Kubernetes manifests: Per-region deployments")
    print(f"✅ Compliance templates: GDPR and EU AI Act")
    print(f"✅ Feature testing: {'Passed' if success else 'Failed'}")
    
    if success:
        print("\n🎉 Global-First Implementation: COMPLETE!")
    else:
        print("\n💥 Global-First Implementation: INCOMPLETE")