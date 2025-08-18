#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) generation script for Model Card Generator.

Generates SPDX-compliant SBOM documents for security and compliance tracking.
Supports multiple output formats: SPDX JSON, SPDX YAML, CycloneDX JSON.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib
import uuid

try:
    import yaml
except ImportError:
    yaml = None

try:
    import toml
except ImportError:
    toml = None


class SBOMGenerator:
    """Generate SBOM documents for the Model Card Generator project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.project_name = "modelcard-as-code-generator"
        self.namespace = "https://github.com/terragonlabs/modelcard-as-code-generator"
        
    def get_python_dependencies(self) -> List[Dict[str, Any]]:
        """Extract Python dependencies from various sources."""
        dependencies = []
        
        # Try to get from pyproject.toml first
        pyproject_file = self.project_root / "pyproject.toml"
        if pyproject_file.exists() and toml:
            with open(pyproject_file) as f:
                pyproject_data = toml.load(f)
                
            # Extract dependencies from different sections
            deps_sections = [
                pyproject_data.get("project", {}).get("dependencies", []),
                pyproject_data.get("project", {}).get("optional-dependencies", {}).get("dev", []),
                pyproject_data.get("build-system", {}).get("requires", [])
            ]
            
            for deps in deps_sections:
                for dep in deps:
                    if isinstance(dep, str):
                        dependencies.append(self._parse_dependency(dep))
        
        # Fallback to requirements files
        if not dependencies:
            req_files = [
                "requirements.txt",
                "requirements-dev.txt"
            ]
            
            for req_file in req_files:
                req_path = self.project_root / req_file
                if req_path.exists():
                    with open(req_path) as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                dependencies.append(self._parse_dependency(line))
        
        return dependencies
    
    def _parse_dependency(self, dep_string: str) -> Dict[str, Any]:
        """Parse a dependency string into structured data."""
        # Remove comments
        dep_string = dep_string.split("#")[0].strip()
        
        # Basic parsing - could be enhanced with proper dependency parsing
        parts = dep_string.replace(">=", "==").replace("<=", "==").replace(">", "==").replace("<", "==")
        if "==" in parts:
            name, version = parts.split("==", 1)
            version = version.split(",")[0]  # Take first version if multiple constraints
        else:
            name = parts
            version = "unknown"
        
        return {
            "name": name.strip(),
            "version": version.strip(),
            "type": "python-package",
            "purl": f"pkg:pypi/{name.strip()}@{version.strip()}" if version != "unknown" else f"pkg:pypi/{name.strip()}"
        }
    
    def get_system_packages(self) -> List[Dict[str, Any]]:
        """Get system packages from Dockerfile."""
        dockerfile_path = self.project_root / "Dockerfile"
        packages = []
        
        if dockerfile_path.exists():
            with open(dockerfile_path) as f:
                content = f.read()
                
            # Simple parsing for apt-get install commands
            lines = content.split("\n")
            for line in lines:
                if "apt-get install" in line and "-y" in line:
                    # Extract package names after apt-get install
                    parts = line.split("apt-get install")[1]
                    parts = parts.replace("-y", "").replace("--no-install-recommends", "")
                    
                    # Remove shell operators and get package names
                    parts = parts.split("&&")[0].split("\\")[0]
                    package_names = [p.strip() for p in parts.split() if p.strip() and not p.startswith("-")]
                    
                    for pkg in package_names:
                        if pkg and not pkg.startswith("#"):
                            packages.append({
                                "name": pkg,
                                "version": "unknown",
                                "type": "deb-package",
                                "purl": f"pkg:deb/debian/{pkg}"
                            })
        
        return packages
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except (OSError, IOError):
            return "unknown"
    
    def get_project_files(self) -> List[Dict[str, Any]]:
        """Get information about project source files."""
        files = []
        src_dir = self.project_root / "src"
        
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                relative_path = py_file.relative_to(self.project_root)
                files.append({
                    "name": str(relative_path),
                    "path": str(py_file),
                    "type": "source-file",
                    "hash": self.calculate_file_hash(py_file),
                    "size": py_file.stat().st_size if py_file.exists() else 0
                })
        
        # Add important config files
        config_files = [
            "pyproject.toml", "setup.py", "requirements.txt", "requirements-dev.txt",
            "Dockerfile", "docker-compose.yml", "Makefile"
        ]
        
        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                files.append({
                    "name": config_file,
                    "path": str(file_path),
                    "type": "configuration-file",
                    "hash": self.calculate_file_hash(file_path),
                    "size": file_path.stat().st_size
                })
        
        return files
    
    def generate_spdx_json(self) -> Dict[str, Any]:
        """Generate SPDX JSON format SBOM."""
        now = datetime.now(timezone.utc)
        document_id = str(uuid.uuid4())
        
        # Get all components
        python_deps = self.get_python_dependencies()
        system_packages = self.get_system_packages()
        project_files = self.get_project_files()
        
        # SPDX document structure
        spdx_doc = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "documentName": f"{self.project_name}-sbom",
            "documentNamespace": f"{self.namespace}/spdx/{document_id}",
            "creationInfo": {
                "created": now.isoformat(),
                "creators": [
                    "Tool: ModelCard-Generator-SBOM-Generator",
                    "Organization: Terragon Labs"
                ],
                "licenseListVersion": "3.20"
            },
            "packages": [],
            "files": [],
            "relationships": []
        }
        
        # Add main project package
        main_package = {
            "SPDXID": "SPDXRef-Package-MainProject",
            "name": self.project_name,
            "downloadLocation": f"{self.namespace}",
            "filesAnalyzed": True,
            "packageVerificationCode": {
                "packageVerificationCodeValue": self._calculate_package_verification_code(project_files)
            },
            "licenseConcluded": "Apache-2.0",
            "licenseDeclared": "Apache-2.0",
            "copyrightText": "Copyright (c) 2025 Terragon Labs",
            "description": "Automated generation of Model Cards as executable, versioned artifacts",
            "homepage": self.namespace,
            "supplier": "Organization: Terragon Labs"
        }
        spdx_doc["packages"].append(main_package)
        
        # Add Python dependencies as packages
        for i, dep in enumerate(python_deps):
            pkg_id = f"SPDXRef-Package-Python-{i}"
            package = {
                "SPDXID": pkg_id,
                "name": dep["name"],
                "versionInfo": dep["version"],
                "downloadLocation": f"https://pypi.org/project/{dep['name']}/",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": dep["purl"]
                    }
                ]
            }
            spdx_doc["packages"].append(package)
            
            # Add relationship
            spdx_doc["relationships"].append({
                "spdxElementId": "SPDXRef-Package-MainProject",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": pkg_id
            })
        
        # Add system packages
        for i, pkg in enumerate(system_packages):
            pkg_id = f"SPDXRef-Package-System-{i}"
            package = {
                "SPDXID": pkg_id,
                "name": pkg["name"],
                "versionInfo": pkg["version"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": pkg["purl"]
                    }
                ]
            }
            spdx_doc["packages"].append(package)
            
            # Add relationship
            spdx_doc["relationships"].append({
                "spdxElementId": "SPDXRef-Package-MainProject",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": pkg_id
            })
        
        # Add files
        for i, file_info in enumerate(project_files):
            file_id = f"SPDXRef-File-{i}"
            file_obj = {
                "SPDXID": file_id,
                "fileName": file_info["name"],
                "checksums": [
                    {
                        "algorithm": "SHA256",
                        "checksumValue": file_info["hash"]
                    }
                ],
                "licenseConcluded": "Apache-2.0" if file_info["type"] == "source-file" else "NOASSERTION",
                "copyrightText": "Copyright (c) 2025 Terragon Labs" if file_info["type"] == "source-file" else "NOASSERTION"
            }
            spdx_doc["files"].append(file_obj)
            
            # Add file relationship to main package
            spdx_doc["relationships"].append({
                "spdxElementId": "SPDXRef-Package-MainProject",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": file_id
            })
        
        return spdx_doc
    
    def _calculate_package_verification_code(self, files: List[Dict[str, Any]]) -> str:
        """Calculate SPDX package verification code."""
        # Simplified calculation - concatenate and hash file hashes
        file_hashes = sorted([f["hash"] for f in files if f["hash"] != "unknown"])
        combined = "".join(file_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def generate_cyclone_dx(self) -> Dict[str, Any]:
        """Generate CycloneDX format SBOM."""
        now = datetime.now(timezone.utc)
        
        # Get components
        python_deps = self.get_python_dependencies()
        system_packages = self.get_system_packages()
        
        cyclone_doc = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": f"urn:uuid:{uuid.uuid4()}",
            "version": 1,
            "metadata": {
                "timestamp": now.isoformat(),
                "tools": [
                    {
                        "vendor": "Terragon Labs",
                        "name": "ModelCard-Generator-SBOM-Generator",
                        "version": "1.0.0"
                    }
                ],
                "component": {
                    "type": "application",
                    "bom-ref": "main-component",
                    "name": self.project_name,
                    "version": "1.0.0",
                    "description": "Automated generation of Model Cards as executable, versioned artifacts",
                    "licenses": [
                        {
                            "license": {
                                "id": "Apache-2.0"
                            }
                        }
                    ],
                    "purl": f"pkg:github/terragonlabs/{self.project_name}@1.0.0"
                }
            },
            "components": []
        }
        
        # Add Python dependencies
        for dep in python_deps:
            component = {
                "type": "library",
                "bom-ref": f"python-{dep['name']}",
                "name": dep["name"],
                "version": dep["version"],
                "purl": dep["purl"],
                "scope": "required"
            }
            cyclone_doc["components"].append(component)
        
        # Add system packages
        for pkg in system_packages:
            component = {
                "type": "operating-system",
                "bom-ref": f"system-{pkg['name']}",
                "name": pkg["name"],
                "version": pkg["version"],
                "purl": pkg["purl"],
                "scope": "required"
            }
            cyclone_doc["components"].append(component)
        
        return cyclone_doc
    
    def save_sbom(self, sbom_data: Dict[str, Any], output_path: Path, format_type: str) -> None:
        """Save SBOM data to file in specified format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(sbom_data, f, indent=2, sort_keys=True)
        elif format_type.endswith('.yaml') or format_type.endswith('.yml'):
            if yaml:
                with open(output_path, 'w') as f:
                    yaml.dump(sbom_data, f, default_flow_style=False, sort_keys=True)
            else:
                raise ImportError("PyYAML is required for YAML output")
        else:
            raise ValueError(f"Unsupported format: {format_type}")


def main():
    """Main entry point for SBOM generation."""
    parser = argparse.ArgumentParser(
        description="Generate Software Bill of Materials (SBOM) for Model Card Generator"
    )
    parser.add_argument(
        "--format",
        choices=["spdx-json", "spdx-yaml", "cyclone-json"],
        default="spdx-json",
        help="SBOM format to generate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sbom"),
        help="Output directory for SBOM files"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Root directory of the project"
    )
    
    args = parser.parse_args()
    
    generator = SBOMGenerator(args.project_root)
    
    try:
        if args.format == "spdx-json":
            sbom_data = generator.generate_spdx_json()
            output_file = args.output / f"{generator.project_name}-spdx.json"
            generator.save_sbom(sbom_data, output_file, "spdx.json")
            
        elif args.format == "spdx-yaml":
            sbom_data = generator.generate_spdx_json()  # Same data structure
            output_file = args.output / f"{generator.project_name}-spdx.yaml"
            generator.save_sbom(sbom_data, output_file, "spdx.yaml")
            
        elif args.format == "cyclone-json":
            sbom_data = generator.generate_cyclone_dx()
            output_file = args.output / f"{generator.project_name}-cyclone.json"
            generator.save_sbom(sbom_data, output_file, "cyclone.json")
        
        print(f"✅ SBOM generated successfully: {output_file}")
        print(f"📊 Format: {args.format}")
        print(f"📝 Components: {len(sbom_data.get('components', sbom_data.get('packages', [])))}")
        
    except Exception as e:
        print(f"❌ Error generating SBOM: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()