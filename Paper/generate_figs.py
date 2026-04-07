import os
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# APA 7.0 figure settings as mandated by visualization_agent
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-safe palette (viridis style focus)
CB_PALETTE = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#BBBBBB', '#000000']

def create_causal_tracing_pdf():
    print("Generating causal_tracing.pdf...")
    with open('../data/layer_attribution_scores.json', 'r') as f:
        scores = json.load(f)
        
    layers = []
    values = []
    
    # Sort layers correctly
    for i in range(35):  # Gemma-4-2B has 35 layers
        k = f"layer_{i}"
        if k in scores:
            layers.append(i)
            # Flip to positive magnitude for easier visual interpretation of "Information Loss"
            values.append(abs(scores[k]))
            
    fig, ax = plt.subplots(figsize=(6.9, 3.5)) # Double column proportional
    
    # Color logic: deep layers get prominent color showing accumulation
    colors = ['#BBBBBB' if v < 0.015 else '#CC3311' for v in values]
    
    bars = ax.bar(layers, values, color=colors, width=0.8)
    
    ax.set_xlabel('Transformer Layer Depth')
    ax.set_ylabel('Attribution Patching (Taylor Approx) Magnitude')
    ax.set_title('Figure 1\n\nCausal Breakdown Magnitude per Layer under 4-bit Quantization', loc='left', pad=15)
    
    # Highlight the specific intervention zone
    ax.axvspan(28.5, 31.5, color='#EEEEEE', alpha=0.5, zorder=0)
    ax.text(30, max(values)*0.9, 'Optimal Intervention Zone', ha='center', fontsize=8, color='#555555')
    
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/causal_tracing.pdf')
    plt.close()

def create_recovery_benchmark_pdf():
    print("Generating recovery_benchmark.pdf...")
    with open('../data/final_benchmark_metrics.json', 'r') as f:
        metrics = json.load(f)
        
    categories = ['Capability Recovery', 'Normative Task Retention']
    values = [metrics['capability_recovery_percentage'], metrics['interference_survival_percentage']]
    
    fig, ax = plt.subplots(figsize=(3.3, 3.5)) # Single column proportional
    
    bars = ax.bar(categories, values, color=['#0077BB', '#009988'], width=0.5)
    
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 110)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
        
    # Draw standard baseline indicator
    ax.axhline(0, color='black', linewidth=1)
    
    # Add minor details for APA
    ax.set_title('Figure 2\n\nIntervention Success Metrics', loc='left', pad=15)
    
    import textwrap
    wrapped_labels = [textwrap.fill(label, 15) for label in categories]
    ax.set_xticklabels(wrapped_labels)
    
    plt.savefig('figs/recovery_benchmark.pdf')
    plt.close()

if __name__ == "__main__":
    create_causal_tracing_pdf()
    create_recovery_benchmark_pdf()
    print("Generated publication-ready APA 7.0 figures.")
