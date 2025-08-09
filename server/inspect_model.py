#!/usr/bin/env python3
"""Inspect ONNX model metadata"""

import onnx
from pathlib import Path

def inspect_model():
    model_path = Path("./models/sherpa-nemo-10lang/model.onnx")
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    model = onnx.load(str(model_path))
    
    print("\n=== MODEL METADATA ===")
    for prop in model.metadata_props:
        print(f"{prop.key}: {prop.value}")
    
    print("\n=== MODEL INPUTS ===")
    for inp in model.graph.input:
        print(f"Name: {inp.name}")
        print(f"Type: {inp.type}")
        if inp.type.tensor_type.shape:
            dims = [str(d.dim_value) if d.dim_value else str(d.dim_param) for d in inp.type.tensor_type.shape.dim]
            print(f"Shape: [{', '.join(dims)}]")
        print()
    
    print("\n=== MODEL OUTPUTS ===")
    for out in model.graph.output:
        print(f"Name: {out.name}")
        print(f"Type: {out.type}")
        if out.type.tensor_type.shape:
            dims = [str(d.dim_value) if d.dim_value else str(d.dim_param) for d in out.type.tensor_type.shape.dim]
            print(f"Shape: [{', '.join(dims)}]")
        print()

if __name__ == "__main__":
    inspect_model()