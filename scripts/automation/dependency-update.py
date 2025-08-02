#!/usr/bin/env python3
"""
Automated dependency update script for Model Card Generator.
Checks for outdated dependencies and creates update PRs.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import requests

class DependencyUpdater:
    """Automate dependency updates and security patches."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo_name = os.getenv('GITHUB_REPOSITORY', 'danieleschmidt/modelcard-as-code-generator')
        self.dry_run = os.getenv('DRY_RUN', 'false').lower() == 'true'
        
    def check_outdated_dependencies(self) -> Dict[str, Any]:
        """Check for outdated dependencies."""
        print("🔍 Checking for outdated dependencies...")
        
        outdated = {}
        
        try:
            # Check Python dependencies
            pip_result = self._run_command(['pip', 'list', '--outdated', '--format=json'])
            if pip_result:
                pip_outdated = json.loads(pip_result)
                outdated['python'] = pip_outdated
                
                print(f"📦 Found {len(pip_outdated)} outdated Python packages")
                for pkg in pip_outdated:
                    print(f"  - {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
            
            # Check for security vulnerabilities
            safety_result = self._run_command(['safety', 'check', '--json'], ignore_errors=True)
            if safety_result:
                try:
                    safety_data = json.loads(safety_result)
                    outdated['security_issues'] = safety_data
                    print(f"🔒 Found {len(safety_data)} security vulnerabilities")
                except json.JSONDecodeError:
                    print("⚠️  Could not parse safety output")
            
        except Exception as e:
            print(f"❌ Error checking dependencies: {e}")
        
        return outdated
    
    def update_dependencies(self, update_type: str = 'minor') -> bool:
        """Update dependencies based on type (patch, minor, major)."""
        print(f"🔄 Updating dependencies (type: {update_type})...")
        
        if self.dry_run:
            print("🔍 DRY RUN: Would update dependencies but not making changes")
            return True
        
        try:
            # Create update branch
            branch_name = f"deps/auto-update-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self._run_command(['git', 'checkout', '-b', branch_name])
            
            updates_made = False
            
            # Update Python dependencies
            if self._should_update_python_deps():
                updates_made |= self._update_python_dependencies(update_type)
            
            # Update GitHub Actions
            if self._should_update_github_actions():
                updates_made |= self._update_github_actions()
            
            # Update Docker base images
            if self._should_update_docker_images():
                updates_made |= self._update_docker_images()
            
            if updates_made:
                # Commit changes
                self._run_command(['git', 'add', '-A'])
                commit_msg = f"deps: automated dependency update ({update_type})\n\n🤖 Generated with automated dependency updater"
                self._run_command(['git', 'commit', '-m', commit_msg])
                
                # Push branch
                self._run_command(['git', 'push', '-u', 'origin', branch_name])
                
                # Create pull request
                if self.github_token:
                    self._create_dependency_pr(branch_name, update_type)
                
                print("✅ Dependency update completed")
                return True
            else:
                print("ℹ️  No updates needed")
                # Clean up branch
                self._run_command(['git', 'checkout', 'main'])
                self._run_command(['git', 'branch', '-D', branch_name])
                return False
                
        except Exception as e:
            print(f"❌ Error updating dependencies: {e}")
            return False
    
    def _should_update_python_deps(self) -> bool:
        """Check if Python dependencies should be updated."""
        return Path('pyproject.toml').exists()
    
    def _should_update_github_actions(self) -> bool:
        """Check if GitHub Actions should be updated."""
        return Path('.github/workflows').exists()
    
    def _should_update_docker_images(self) -> bool:
        """Check if Docker images should be updated."""
        return Path('Dockerfile').exists()
    
    def _update_python_dependencies(self, update_type: str) -> bool:
        """Update Python dependencies in pyproject.toml."""
        print("🐍 Updating Python dependencies...")
        
        updates_made = False
        
        try:
            # Get current dependencies
            current_deps = self._get_current_python_deps()
            
            # Check each dependency for updates
            for dep_name, current_version in current_deps.items():
                latest_version = self._get_latest_python_version(dep_name)
                
                if latest_version and self._should_update_version(
                    current_version, latest_version, update_type
                ):
                    print(f"  📦 Updating {dep_name}: {current_version} → {latest_version}")
                    self._update_python_dep_in_file(dep_name, latest_version)
                    updates_made = True
                    
        except Exception as e:
            print(f"❌ Error updating Python dependencies: {e}")
        
        return updates_made
    
    def _update_github_actions(self) -> bool:
        """Update GitHub Actions to latest versions."""
        print("⚙️ Updating GitHub Actions...")
        
        updates_made = False
        workflow_dir = Path('.github/workflows')
        
        if not workflow_dir.exists():
            return False
        
        try:
            for workflow_file in workflow_dir.glob('*.yml'):
                content = workflow_file.read_text()
                original_content = content
                
                # Update common actions
                action_updates = {
                    'actions/checkout@v3': 'actions/checkout@v4',
                    'actions/setup-python@v4': 'actions/setup-python@v5',
                    'actions/cache@v3': 'actions/cache@v4',
                    'codecov/codecov-action@v3': 'codecov/codecov-action@v4',
                }
                
                for old_action, new_action in action_updates.items():
                    if old_action in content:
                        content = content.replace(old_action, new_action)
                        print(f"  🔄 Updated {old_action} → {new_action} in {workflow_file.name}")
                
                if content != original_content:
                    workflow_file.write_text(content)
                    updates_made = True
                    
        except Exception as e:
            print(f"❌ Error updating GitHub Actions: {e}")
        
        return updates_made
    
    def _update_docker_images(self) -> bool:
        """Update Docker base images."""
        print("🐳 Updating Docker images...")
        
        updates_made = False
        dockerfile = Path('Dockerfile')
        
        if not dockerfile.exists():
            return False
        
        try:
            content = dockerfile.read_text()
            original_content = content
            
            # Update Python base images
            image_updates = {
                'python:3.11-slim-bookworm': self._get_latest_python_image(),
            }
            
            for old_image, new_image in image_updates.items():
                if new_image and old_image in content:
                    content = content.replace(old_image, new_image)
                    print(f"  🐳 Updated {old_image} → {new_image}")
                    updates_made = True
            
            if content != original_content:
                dockerfile.write_text(content)
                
        except Exception as e:
            print(f"❌ Error updating Docker images: {e}")
        
        return updates_made
    
    def _get_current_python_deps(self) -> Dict[str, str]:
        """Extract current Python dependencies from pyproject.toml."""
        try:
            import toml
            with open('pyproject.toml') as f:
                data = toml.load(f)
            
            dependencies = data.get('project', {}).get('dependencies', [])
            deps_dict = {}
            
            for dep in dependencies:
                # Parse dependency string (e.g., "click>=8.0.0")
                import re
                match = re.match(r'^([a-zA-Z0-9_-]+)([>=<]+)?(.+)?', dep)
                if match:
                    name = match.group(1)
                    version = match.group(3) if match.group(3) else 'latest'
                    deps_dict[name] = version
            
            return deps_dict
            
        except Exception as e:
            print(f"❌ Error parsing dependencies: {e}")
            return {}
    
    def _get_latest_python_version(self, package_name: str) -> str:
        """Get latest version of a Python package from PyPI."""
        try:
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data['info']['version']
        except Exception as e:
            print(f"⚠️  Could not get latest version for {package_name}: {e}")
        
        return None
    
    def _get_latest_python_image(self) -> str:
        """Get latest Python Docker image tag."""
        # For simplicity, return current stable version
        # In a real implementation, you'd query the Docker Hub API
        return "python:3.11-slim-bookworm"
    
    def _should_update_version(self, current: str, latest: str, update_type: str) -> bool:
        """Determine if version should be updated based on update type."""
        try:
            from packaging import version
            
            current_ver = version.parse(current)
            latest_ver = version.parse(latest)
            
            if latest_ver <= current_ver:
                return False
            
            if update_type == 'patch':
                return (current_ver.major == latest_ver.major and 
                       current_ver.minor == latest_ver.minor)
            elif update_type == 'minor':
                return current_ver.major == latest_ver.major
            elif update_type == 'major':
                return True
            
        except Exception:
            # Fallback: simple string comparison
            return current != latest
        
        return False
    
    def _update_python_dep_in_file(self, dep_name: str, new_version: str):
        """Update a specific dependency in pyproject.toml."""
        try:
            import toml
            
            with open('pyproject.toml') as f:
                data = toml.load(f)
            
            dependencies = data.get('project', {}).get('dependencies', [])
            
            # Find and update the dependency
            for i, dep in enumerate(dependencies):
                if dep.startswith(dep_name):
                    # Update the version
                    import re
                    pattern = r'^([a-zA-Z0-9_-]+)([>=<]+.+)?'
                    match = re.match(pattern, dep)
                    if match:
                        dependencies[i] = f"{match.group(1)}>={new_version}"
                        break
            
            # Write back to file
            with open('pyproject.toml', 'w') as f:
                toml.dump(data, f)
                
        except Exception as e:
            print(f"❌ Error updating {dep_name}: {e}")
    
    def _create_dependency_pr(self, branch_name: str, update_type: str):
        """Create a pull request for dependency updates."""
        if not self.github_token:
            print("⚠️  No GitHub token available, cannot create PR")
            return
        
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            title = f"deps: automated {update_type} dependency updates"
            body = f"""## Automated Dependency Updates

This PR contains automated {update_type} dependency updates.

### Changes
- Updated Python packages to latest {update_type} versions
- Updated GitHub Actions to latest versions
- Updated Docker base images (if applicable)

### Testing
- [ ] All CI checks pass
- [ ] No breaking changes detected
- [ ] Security scans complete

### Notes
This PR was generated automatically by the dependency update system.
Please review the changes and merge if all tests pass.

🤖 Generated with automated dependency updater
"""
            
            pr_data = {
                'title': title,
                'body': body,
                'head': branch_name,
                'base': 'main'
            }
            
            url = f"https://api.github.com/repos/{self.repo_name}/pulls"
            response = requests.post(url, json=pr_data, headers=headers)
            
            if response.status_code == 201:
                pr_url = response.json()['html_url']
                print(f"✅ Created PR: {pr_url}")
            else:
                print(f"❌ Failed to create PR: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Error creating PR: {e}")
    
    def _run_command(self, cmd: List[str], ignore_errors: bool = False) -> str:
        """Run a shell command and return output."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.project_root,
                timeout=300  # 5 minutes
            )
            
            if result.returncode != 0 and not ignore_errors:
                print(f"⚠️  Command failed: {' '.join(cmd)}")
                print(f"   Error: {result.stderr}")
                return ""
            
            return result.stdout
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Command timed out: {' '.join(cmd)}")
            return ""
        except Exception as e:
            if not ignore_errors:
                print(f"❌ Command error: {e}")
            return ""


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated dependency updater")
    parser.add_argument(
        '--type',
        choices=['patch', 'minor', 'major'],
        default='minor',
        help='Type of updates to apply'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check for updates, do not apply them'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without making changes'
    )
    
    args = parser.parse_args()
    
    # Set dry run environment variable
    if args.dry_run:
        os.environ['DRY_RUN'] = 'true'
    
    updater = DependencyUpdater()
    
    if args.check_only:
        # Just check for outdated dependencies
        outdated = updater.check_outdated_dependencies()
        
        if outdated.get('python'):
            print(f"\n📦 {len(outdated['python'])} Python packages can be updated")
        
        if outdated.get('security_issues'):
            print(f"🔒 {len(outdated['security_issues'])} security issues found")
        
        # Exit with non-zero if updates are available
        has_updates = bool(outdated.get('python') or outdated.get('security_issues'))
        sys.exit(1 if has_updates else 0)
    
    else:
        # Perform updates
        success = updater.update_dependencies(args.type)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()