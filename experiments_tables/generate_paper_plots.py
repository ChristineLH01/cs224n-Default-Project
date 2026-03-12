import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Set publication quality styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

def compute_gpu_metrics(df):
    df['num_gpus'] = df.get('train_meta.num_gpus', 1).fillna(1).astype(int)
    # Total train time in GPU-hours
    if 'train_meta.total_train_time_sec' in df.columns:
        df['gpu_hours'] = df['train_meta.total_train_time_sec'] * df['num_gpus'] / 3600.0
    else:
        df['gpu_hours'] = 0.0
    return df

def create_tradeoff_plot(csv_path, metric_col, metric_label, condition_col, condition_val, output_filename, is_chrf=False):
    if not os.path.exists(csv_path):
        print(f"Cannot find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    df = compute_gpu_metrics(df)
    
    # Capitalize method name to match old script assumptions: "Full", "Lora", etc.
    df['Method'] = df['method'].str.capitalize()
    df['Model'] = df['model_size']
    df['GPU_Hours'] = df['gpu_hours']
    
    # Filter by dataset condition
    df = df[df[condition_col] == condition_val]
    
    # We want to plot the Max metrics over LR, epoch, etc per Method/Model group
    grouped = df.groupby(['Model', 'Method']).agg({
        metric_col: 'max',
        'GPU_Hours': 'mean'
    }).reset_index()
    
    methods_order = ["Full", "Lora", "Rslora", "Adalora", "Dora"]
    method_labels = ["Full", "LoRA", "rsLoRA", "AdaLoRA", "DoRA"]
    
    colors = sns.color_palette("Set1", len(methods_order))
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    for ax, model, title in zip(axes, ['gpt2', 'gpt2-medium'], ["GPT-2", "GPT-2 Medium"]):
        model_df = grouped[grouped['Model'] == model]
        if model_df.empty:
            continue
            
        for i, (method, label) in enumerate(zip(methods_order, method_labels)):
            row = model_df[model_df['Method'] == method]
            if not row.empty:
                x = row['GPU_Hours'].iloc[0]
                y = row[metric_col].iloc[0]
                
                # Small jitter strictly for identical points so both dots are visibly separate
                jitter_y = 0
                if model == "gpt2" and not is_chrf:
                    if method == "Adalora":
                        jitter_y = -0.0003
                    elif method == "Dora":
                        jitter_y = 0.0003
                
                # Plot point
                ax.scatter(x, y + jitter_y, s=150, color=colors[i], edgecolor='black', linewidth=1.5, zorder=5, alpha=0.9, label=label)
                
                # Base offsets
                if method == "Full":
                    offset_x, offset_y = 0, 12
                    ha, va = 'center', 'bottom'
                elif method == "Lora":
                    offset_x, offset_y = -12, 0
                    ha, va = 'right', 'center'
                elif method == "Rslora":
                    offset_x, offset_y = 12, 0
                    ha, va = 'left', 'center'
                elif method == "Adalora":
                    offset_x, offset_y = 0, -14
                    ha, va = 'center', 'top'
                elif method == "Dora":
                    offset_x, offset_y = 12, 12
                    ha, va = 'left', 'bottom'
                    
                # Overrides for specific plot layouts
                if model == "gpt2" and not is_chrf:
                    if method == "Adalora":
                        offset_x, offset_y = -10, -10
                        ha, va = 'right', 'top'
                    elif method == "Dora":
                        offset_x, offset_y = 10, 10
                        ha, va = 'left', 'bottom'
                        
                if model == "gpt2-medium" and not is_chrf:
                    if method == "Dora":
                        offset_x, offset_y = 12, -12
                        ha, va = 'left', 'top'
                        
                if model == "gpt2" and is_chrf:
                    if method == "Full":
                        offset_x, offset_y = -12, 0
                        ha, va = 'right', 'center'
                    elif method == "Rslora":
                        offset_x, offset_y = 12, 0
                        ha, va = 'left', 'center'
                        
                ax.annotate(label, (x, y + jitter_y), xytext=(offset_x, offset_y), textcoords="offset points",
                            fontsize=12, fontweight='bold', ha=ha, va=va, zorder=10)
        
        ax.set_title(title, pad=15, fontweight='bold')
        ax.set_xlabel("GPU Compute (Hours)", labelpad=10)
        ax.set_ylabel(metric_label, labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
        
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set_xlim(x_min - (x_max-x_min)*0.15, x_max + (x_max-x_min)*0.15)
        ax.set_ylim(y_min - (y_max-y_min)*0.15, y_max + (y_max-y_min)*0.15)
        
    plt.tight_layout(w_pad=4.0)
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    print(f"Saved {output_filename}")

def main():
    create_tradeoff_plot(
        csv_path='paraphrase_combined.csv', 
        metric_col='best_dev_acc',
        metric_label='Validation Accuracy',
        condition_col='additional_data_used',
        condition_val=False,
        output_filename='paper_tables/paraphrase_tradeoff.png',
        is_chrf=False
    )
    
    create_tradeoff_plot(
        csv_path='sonnet_combined.csv', 
        metric_col='best_chrf', 
        metric_label='Validation chrF',
        condition_col='additional_data_used',
        condition_val=True,
        output_filename='paper_tables/sonnet_tradeoff.png',
        is_chrf=True
    )

if __name__ == '__main__':
    main()
