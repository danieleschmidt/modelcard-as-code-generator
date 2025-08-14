"""Standalone research runner for algorithm optimization."""

import asyncio
import time
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from modelcard_generator.research.advanced_optimizer import execute_autonomous_research


async def main():
    """Run the research study."""
    print("🧬 TERRAGON LABS - AUTONOMOUS RESEARCH EXECUTION")
    print("🔬 Model Card Generation Algorithm Optimization Study")
    print("=" * 80)
    
    try:
        results = await execute_autonomous_research()
        
        print("\n" + "=" * 80)
        print("📋 RESEARCH SUMMARY")
        print("=" * 80)
        
        print(f"📊 Baseline Throughput: {results['results']['baseline_throughput']:.2f} cards/sec")
        print(f"⚡ Optimized Throughput: {results['results']['optimized_throughput']:.2f} cards/sec")
        print(f"🚀 Performance Gain: {results['results']['improvement_factor']:.2f}x")
        print(f"📈 Statistical Significance: p = {results['results']['statistical_significance']:.6f}")
        
        print("\n🎯 KEY FINDINGS:")
        for conclusion in results['conclusions']:
            print(f"   • {conclusion}")
        
        print("\n🔮 FUTURE RESEARCH DIRECTIONS:")
        for direction in results['future_work']:
            print(f"   • {direction}")
        
        print("\n✅ RESEARCH EXECUTION COMPLETED SUCCESSFULLY")
        print("📚 Results ready for peer review and academic publication")
        
    except Exception as e:
        print(f"❌ Research execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)