"""
Main entry point for Intrusion Detection System
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import IntrusionDetectionPipeline
from src.core.config import Config


def main():
    """Main function"""
    
    print("="*60)
    print("INTRUSION DETECTION SYSTEM")
    print("Based on YOLO + DeepSORT")
    print("="*60)
    
    Config.ensure_directories()
    
    pipeline = IntrusionDetectionPipeline()
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("SYSTEM SHUTDOWN")
    print("="*60)


if __name__ == "__main__":
    main()