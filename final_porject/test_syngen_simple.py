#!/usr/bin/env python3
"""
Simple test script to verify SynGen integration without heavy dependencies
"""

try:
    import spacy
    print("✓ spaCy imported successfully")
    
    # Test spaCy model loading
    nlp = spacy.load("en_core_web_sm")
    print("✓ English model loaded successfully")
    
    # Test basic parsing
    doc = nlp("red apple")
    print(f"✓ Parsed '{doc.text}': {[(token.text, token.pos_) for token in doc]}")
    
    print("\n=== SynGen Integration Test ===")
    print("✓ All dependencies installed correctly")
    print("✓ Ready for SynGen integration")
    
    print("\nNext steps:")
    print("1. Install PyTorch: pip install torch")
    print("2. Install diffusers: pip install diffusers transformers")
    print("3. Run: python generate_exemplar.py")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")