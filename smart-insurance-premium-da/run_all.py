"""
Main Runner - Execute All Analysis Phases
Runs all 7 phases of the insurance premium data analysis pipeline.
"""

import subprocess
import sys
import time

def run_script(script_name, phase_num, phase_name):
    """Run a single Python script and handle errors."""
    print("\n" + "=" * 80)
    print(f"RUNNING PHASE {phase_num}: {phase_name}")
    print("=" * 80)
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ Phase {phase_num} completed successfully in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Phase {phase_num} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_name}")
        return False

def main():
    """Execute all analysis phases in sequence."""
    print("=" * 80)
    print("INSURANCE PREMIUM DATA ANALYSIS - FULL PIPELINE")
    print("=" * 80)
    print("\nThis will run all 7 phases of the analysis:")
    print("  Phase 1: Data Loading")
    print("  Phase 2: EDA & Distributions")
    print("  Phase 3: NumPy Aggregations")
    print("  Phase 4: Segmentation Analysis")
    print("  Phase 5: Advanced Visualizations")
    print("  Phase 6: Feature Engineering")
    print("  Phase 7: AI-Assisted Insights")
    
    # Define all phases
    phases = [
        ("01_data_loading.py", 1, "Data Loading"),
        ("02_eda_distributions.py", 2, "EDA & Distributions"),
        ("03_numpy_aggregations.py", 3, "NumPy Aggregations"),
        ("04_segmentation.py", 4, "Segmentation Analysis"),
        ("05_visualizations.py", 5, "Advanced Visualizations"),
        ("06_feature_engineering.py", 6, "Feature Engineering"),
        ("07_ai_insights.py", 7, "AI-Assisted Insights")
    ]
    
    results = []
    start_time = time.time()
    
    for script, phase_num, phase_name in phases:
        success = run_script(script, phase_num, phase_name)
        results.append((phase_num, phase_name, success))
        
        if not success:
            print(f"\n⚠ Warning: Phase {phase_num} failed. Continuing with remaining phases...")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    
    for phase_num, phase_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  Phase {phase_num} ({phase_name}): {status}")
    
    successful = sum(1 for _, _, success in results if success)
    total = len(results)
    
    print(f"\nCompleted: {successful}/{total} phases")
    print(f"Total execution time: {total_time:.2f}s")
    
    if successful == total:
        print("\All phases completed successfully!")
        print("\nGenerated outputs:")
        print("  • outputs/distributions/ - Distribution plots")
        print("  • outputs/segmentation/ - Segmentation analysis")
        print("  • outputs/visualizations/ - Advanced visualizations")
        print("  • outputs/features/ - Feature engineering results")
        print("  • outputs/reports/ - AI-generated insights reports")
    else:
        print(f"\n⚠ {total - successful} phase(s) failed. Check output above for details.")
        sys.exit(1)
    
    print("=" * 80)

if __name__ == "__main__":
    main()
