"""
DVC-specific configuration for pipeline tracking.
"""
from pathlib import Path

# DVC pipeline configuration
DVC_CONFIG = {
    "stages": {
        "simple_pipeline": {
            "cmd": "python src/pipeline/task1_pipeline.py",
            "deps": [
                "src/pipeline/task1_pipeline.py",
                "data/00_raw/MachineLearningRating_v3.txt"
            ],
            "outs": [
                "data/01_interim/raw_data.txt",
                "data/01_interim/cleaned_data.txt", 
                "data/01_interim/features_data.txt"
            ]
        }
    },
    "plots": ["reports/figures/"],
    "metrics": ["reports/analysis.json"]
}

def get_dvc_yaml_content():
    """Generate dvc.yaml content from configuration."""
    yaml_content = "# dvc.yaml - Generated from config\n"
    yaml_content += "stages:\n"
    
    for stage_name, stage_config in DVC_CONFIG["stages"].items():
        yaml_content += f"  {stage_name}:\n"
        yaml_content += f"    cmd: {stage_config['cmd']}\n"
        
        if "deps" in stage_config:
            yaml_content += "    deps:\n"
            for dep in stage_config["deps"]:
                yaml_content += f"    - {dep}\n"
        
        if "outs" in stage_config:
            yaml_content += "    outs:\n"
            for out in stage_config["outs"]:
                yaml_content += f"    - {out}\n"
    
    return yaml_content