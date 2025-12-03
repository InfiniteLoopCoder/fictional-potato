#!/usr/bin/env python
"""
Verify GRPO pipeline setup and environment
"""
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (requires 3.8+)")
        return False


def check_dependencies():
    """Check required packages"""
    print("\nChecking dependencies...")

    required = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "aiohttp",
        "tqdm",
        "numpy",
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)

    return len(missing) == 0


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    Version: {torch.version.cuda}")
            print(f"    Devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"    GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
            return True
        else:
            print(f"  ⚠ CUDA not available (CPU only)")
            return False
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def check_project_structure():
    """Check project file structure"""
    print("\nChecking project structure...")

    required_files = [
        "config.py",
        "main.py",
        "requirements.txt",
        "README.md",
        "data/__init__.py",
        "data/download_mbpp.py",
        "synthesis/__init__.py",
        "synthesis/teacher_query.py",
        "synthesis/generate_traces.py",
        "training/__init__.py",
        "training/grpo_trainer.py",
        "training/losses.py",
        "training/utils.py",
        "evaluation/__init__.py",
        "evaluation/code_executor.py",
        "evaluation/pass_at_k.py",
    ]

    missing = []
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            missing.append(file)

    return len(missing) == 0


def check_config():
    """Check configuration"""
    print("\nChecking configuration...")
    try:
        from config import get_default_config

        config = get_default_config()
        print(f"  ✓ Config loaded successfully")
        print(f"    Teacher API: {config.teacher.api_url}")
        print(f"    Student Model: {config.student.model_name}")
        print(f"    LoRA Rank: {config.lora.r}")
        return True
    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        return False


def check_teacher_connection():
    """Check teacher model connection"""
    print("\nChecking teacher model connection...")
    print("  (This requires vLLM server to be running)")

    try:
        import asyncio
        from config import get_default_config
        from synthesis.teacher_query import test_teacher_connection

        config = get_default_config()
        result = asyncio.run(test_teacher_connection(config.teacher))

        if result:
            print("  ✓ Teacher model connection successful")
            return True
        else:
            print("  ✗ Teacher model connection failed")
            return False
    except Exception as e:
        print(f"  ⚠ Could not test connection: {e}")
        print("    Run manually: python synthesis/teacher_query.py")
        return False


def print_summary(results):
    """Print summary of checks"""
    print("\n" + "=" * 70)
    print("SETUP VERIFICATION SUMMARY")
    print("=" * 70)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {check}")

    print("=" * 70)

    if all_passed:
        print("✓ All checks passed! You're ready to run the pipeline.")
        print("\nNext steps:")
        print("  1. Start vLLM server (if not running)")
        print("  2. Test teacher connection: python synthesis/teacher_query.py")
        print("  3. Run pipeline: python main.py --stage all")
        print("  4. Or use shell script: ./run_pipeline.sh all")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nTo install dependencies:")
        print("  pip install -r requirements.txt")

    print("=" * 70)


def main():
    """Main verification function"""
    print("=" * 70)
    print("Teacher-Guided GRPO Pipeline - Setup Verification")
    print("=" * 70)

    results = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "CUDA": check_cuda(),
        "Project Structure": check_project_structure(),
        "Configuration": check_config(),
    }

    # Optional: Check teacher connection
    try:
        results["Teacher Connection"] = check_teacher_connection()
    except:
        results["Teacher Connection"] = False

    print_summary(results)


if __name__ == "__main__":
    main()
