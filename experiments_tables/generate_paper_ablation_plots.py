import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set publication quality styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

def get_target_module(row):
    # Determine LoRA target
    if row['method'] == 'lora':
        if row['apply_lora_to_attention'] == True and row['apply_lora_to_mlp'] == False:
            return 'Attention'
        elif row['apply_lora_to_attention'] == False and row['apply_lora_to_mlp'] == True:
            return 'MLP'
        elif row['apply_lora_to_attention'] == True and row['apply_lora_to_mlp'] == True:
            return 'Both'
            
    # Determine DoRA target
    elif row['method'] == 'dora':
        if row['fine_tune_mode'] == 'dora_attention':
            return 'Attention'
        elif row['fine_tune_mode'] == 'dora':
            return 'Both'
    elif row['method'] == 'dora_mlp':
        return 'MLP'
        
    return 'Unknown'

def get_base_method(row):
    if row['method'] == 'lora':
        return 'LoRA'
    elif row['method'] in ['dora', 'dora_mlp']:
        return 'DoRA'
    return row['method']

def plot_module_ablation(csv_path, metric_col, metric_label, output_filename, is_chrf=False):
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    # Filter to just unaugmented data for clean module comparison
    df = df[df['additional_data_used'] == False]
    
    # Compute GPU Hours
    df['num_gpus'] = df.get('train_meta.num_gpus', 1).fillna(1).astype(int)
    if 'train_meta.total_train_time_sec' in df.columns:
        df['gpu_hours'] = df['train_meta.total_train_time_sec'] * df['num_gpus'] / 3600.0
    else:
        df['gpu_hours'] = 0.0
    
    # Extract target modules and base methods
    df['Target'] = df.apply(get_target_module, axis=1)
    df['BaseMethod'] = df.apply(get_base_method, axis=1)
    
    # Filter to just LoRA and DoRA where Target is known
    plot_df = df[(df['BaseMethod'].isin(['LoRA', 'DoRA'])) & (df['Target'] != 'Unknown')]
    
    # Get the max performance & mean GPU time for each configuration
    agg_df = plot_df.groupby(['model_size', 'BaseMethod', 'Target']).agg({
        metric_col: 'max',
        'gpu_hours': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Setup aesthetic mappings
    colors = sns.color_palette("Set1")
    method_colors = {'LoRA': colors[1], 'DoRA': colors[4]} # Blue and Orange distinct from tradeoff plots
    target_markers = {'Attention': 'o', 'MLP': 's', 'Both': 'D'}
    
    for ax, model, title in zip(axes, ['gpt2', 'gpt2-medium'], ["GPT-2", "GPT-2 Medium"]):
        model_df = agg_df[agg_df['model_size'] == model]
        if model_df.empty: 
            continue
            
        # Draw the trajectory lines first so they are under the points
        for method in ['LoRA', 'DoRA']:
            method_df = model_df[model_df['BaseMethod'] == method].sort_values(by='gpu_hours')
            if not method_df.empty:
                ax.plot(
                    method_df['gpu_hours'], 
                    method_df[metric_col], 
                    color=method_colors[method], 
                    linestyle='-', 
                    linewidth=2.5, 
                    alpha=0.6,
                    zorder=4,
                    label=method
                )
                
        # Draw the points and labels
        legend_handles = []
        for target in ['Attention', 'MLP', 'Both']:
            for method in ['LoRA', 'DoRA']:
                row = model_df[(model_df['Target'] == target) & (model_df['BaseMethod'] == method)]
                if not row.empty:
                    x, y = row['gpu_hours'].iloc[0], row[metric_col].iloc[0]
                    scatter = ax.scatter(
                        x, y, 
                        s=160, 
                        color=method_colors[method], 
                        marker=target_markers[target],
                        edgecolor='black', 
                        linewidth=1.2,
                        zorder=5
                    )
                    
                    # Manual collision-free annotation per point
                    offset_x, offset_y = 0, 10
                    ha, va = 'center', 'bottom'
                    if target == "Both":
                        offset_x, offset_y = 12, 0
                        ha, va = 'left', 'center'
                    elif target == "MLP":
                        offset_x, offset_y = -12, 0
                        ha, va = 'right', 'center'
                        
                    ax.annotate(
                        target, 
                        (x, y), 
                        xytext=(offset_x, offset_y), 
                        textcoords="offset points", 
                        ha=ha, va=va, 
                        fontsize=10, 
                        fontweight='bold', 
                        zorder=10
                    )
        
        ax.set_title(title, pad=15, fontweight='bold')
        ax.set_xlabel('GPU Compute (Hours)', labelpad=10)
        ax.set_ylabel(metric_label, labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
        
        # Clean up X-axis precision to prevent run-on zeros
        import matplotlib.ticker as ticker
        if is_chrf:
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        else:
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        # Margins to protect text
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set_xlim(x_min - (x_max-x_min)*0.15, x_max + (x_max-x_min)*0.15)
        ax.set_ylim(y_min - (y_max-y_min)*0.15, y_max + (y_max-y_min)*0.15)
        
        ax.legend(title='Method', loc='best', framealpha=0.9)
        
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    print(f"Saved {output_filename}")


def plot_data_scaling(csv_path, metric_col, metric_label, output_filename, is_chrf=False):
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    
    # Standardize method names
    method_map = {
        'full': 'Full', 'lora': 'LoRA', 'rslora': 'rsLoRA', 
        'adalora': 'AdaLoRA', 'dora': 'DoRA' # ignoring dora_mlp here for clean main method comparison
    }
    df = df[df['method'].isin(method_map.keys())]
    df['Method'] = df['method'].map(method_map)
    df['Data Size'] = df['additional_data_used'].map({False: 'Original (Small)', True: 'Augmented (Large)'})
    
    # Get max performance
    agg_df = df.groupby(['model_size', 'Method', 'Data Size'])[metric_col].max().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    data_order = ['Original (Small)', 'Augmented (Large)']
    
    # To draw lines properly, we use pointplot
    for ax, model, title in zip(axes, ['gpt2', 'gpt2-medium'], ["GPT-2", "GPT-2 Medium"]):
        model_df = agg_df[agg_df['model_size'] == model]
        if model_df.empty:
            continue
            
        sns.pointplot(
            data=model_df,
            x='Data Size',
            y=metric_col,
            hue='Method',
            order=data_order,
            ax=ax,
            markers=['o', 's', '^', 'D', 'v'],
            linestyles=['-', '--', '-.', ':', '-'],
            palette='Set1'
        )
        
        ax.set_title(title, pad=15, fontweight='bold')
        ax.set_xlabel('Dataset', labelpad=10)
        ax.set_ylabel(metric_label, labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    print(f"Saved {output_filename}")


def plot_rank_scaling(csv_path, metric_col, metric_label, output_filename, is_chrf=False):
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    # Only consider unaugmented data for clean hyperparameter ablation
    df = df[df['additional_data_used'] == False]
    
    # We need to unify the "Rank" concept. 
    # LoRA/rsLoRA/DoRA use lora_rank. AdaLoRA uses adalora_target_r.
    def get_rank(row):
        if row['method'] == 'adalora':
            # Multiply by a factor or just use target_r. Usually AdaLoRA rank is just target_r.
            return row.get('adalora_target_r', 0)
        return row.get('lora_rank', 0)
        
    df['Effective_Rank'] = df.apply(get_rank, axis=1)
    
    # Filter out methods that don't have a rank or where rank is 0 or NaN
    df = df[df['Effective_Rank'] > 0]
    df = df.dropna(subset=['Effective_Rank'])
    
    # Standardize method names
    method_map = {'lora': 'LoRA', 'rslora': 'rsLoRA', 'adalora': 'AdaLoRA', 'dora': 'DoRA'}
    df = df[df['method'].isin(method_map.keys())]
    df['Method'] = df['method'].map(method_map)
    
    # Compute GPU Hours
    df['num_gpus'] = df.get('train_meta.num_gpus', 1).fillna(1).astype(int)
    if 'train_meta.total_train_time_sec' in df.columns:
        df['gpu_hours'] = df['train_meta.total_train_time_sec'] * df['num_gpus'] / 3600.0
    else:
        df['gpu_hours'] = 0.0
        
    # Get max performance & mean GPU time for each rank and method
    agg_df = df.groupby(['model_size', 'Method', 'Effective_Rank']).agg({
        metric_col: 'max',
        'gpu_hours': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    for ax, model, title in zip(axes, ['gpt2', 'gpt2-medium'], ["GPT-2", "GPT-2 Medium"]):
        model_df = agg_df[agg_df['model_size'] == model]
        if model_df.empty: continue
            
        # Plot metric on primary y-axis
        sns.lineplot(
            data=model_df,
            x='Effective_Rank',
            y=metric_col,
            hue='Method',
            style='Method',
            markers=['o', 's', '^', 'D'],
            dashes=False,
            ax=ax,
            palette='Set1',
            linewidth=2,
            markersize=8,
            legend=True
        )
        
        # Plot GPU time on secondary y-axis
        ax2 = ax.twinx()
        sns.lineplot(
            data=model_df,
            x='Effective_Rank',
            y='gpu_hours',
            hue='Method',
            style='Method',
            markers=['o', 's', '^', 'D'],
            dashes=[(2, 2)] * len(model_df['Method'].unique()),
            ax=ax2,
            palette='Set1',
            linewidth=2,
            markersize=8,
            alpha=0.4,
            legend=False
        )
        
        # Log scale for rank axis makes it cleaner (8, 16, 32, 64)
        ax.set_xscale('log', base=2)
        
        # Format the X-axis ticks to show actual rank numbers cleanly
        ranks = sorted(model_df['Effective_Rank'].unique())
        if ranks:
            ax.set_xticks(ranks)
            # Use integer labels for clean formatting
            ax.set_xticklabels([f"{int(r)}" for r in ranks])
            
        ax.set_title(title, pad=15, fontweight='bold')
        ax.set_xlabel('Low-Rank Dimension ($r$)', labelpad=10)
        ax.set_ylabel(metric_label, labelpad=10)
        ax2.set_ylabel('GPU Compute (Hours)', labelpad=10, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        ax.grid(True, linestyle='--', alpha=0.6)
        ax2.grid(False)
        
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    print(f"Saved {output_filename}")


def plot_rslora_rank_sweep(csv_path, metric_col, metric_label, output_filename):
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    # Filter to only rsLoRA and gpt2-medium
    df = df[(df['method'] == 'rslora') & (df['lora_rank'] > 0) & (df['model_size'] == 'gpt2-medium')]
    df = df.dropna(subset=['lora_rank'])
    
    # Map data type
    df['Data'] = df['additional_data_used'].map({False: 'Original Data', True: 'Augmented Data'})
    
    agg_df = df.groupby(['Data', 'lora_rank'])[metric_col].max().reset_index()
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    sns.lineplot(
        data=agg_df,
        x='lora_rank',
        y=metric_col,
        hue='Data',
        style='Data',
        markers=['o', 's'],
        dashes=False,
        ax=ax,
        palette='Set1',
        linewidth=2.5,
        markersize=9
    )
    
    # Linear scale with explicit ticks
    ranks = sorted(agg_df['lora_rank'].unique())
    ax.set_xticks(ranks)
    ax.set_xticklabels([f"{int(r)}" for r in ranks])
    
    ax.set_title('rsLoRA Rank Sweep (GPT-2 Medium)', pad=15, fontweight='bold')
    ax.set_xlabel('rsLoRA Rank ($r$)', labelpad=10)
    ax.set_ylabel(metric_label, labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Dataset', loc='best', framealpha=0.9)
        
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    print(f"Saved {output_filename}")



def main():
    out_dir = 'paper_tables'
    os.makedirs(out_dir, exist_ok=True)
    
    print("Generating Module Ablation Plots...")
    plot_module_ablation(
        'paraphrase_combined.csv', 
        'best_dev_acc_max' if 'best_dev_acc_max' in pd.read_csv('paraphrase_combined.csv').columns else 'best_dev_acc', 
        'Validation Accuracy', 
        f'{out_dir}/paraphrase_module_ablation.png', 
        is_chrf=False
    )
    plot_module_ablation(
        'sonnet_combined.csv', 
        'best_chrf_max' if 'best_chrf_max' in pd.read_csv('sonnet_combined.csv').columns else 'best_chrf', 
        'Validation chrF', 
        f'{out_dir}/sonnet_module_ablation.png', 
        is_chrf=True
    )
    
    print("Generating Data Scaling Line Charts...")
    plot_data_scaling(
        'paraphrase_combined.csv',
        'best_dev_acc_max' if 'best_dev_acc_max' in pd.read_csv('paraphrase_combined.csv').columns else 'best_dev_acc',
        'Validation Accuracy',
        f'{out_dir}/paraphrase_data_scaling.png',
        is_chrf=False
    )
    plot_data_scaling(
        'sonnet_combined.csv',
        'best_chrf_max' if 'best_chrf_max' in pd.read_csv('sonnet_combined.csv').columns else 'best_chrf',
        'Validation chrF',
        f'{out_dir}/sonnet_data_scaling.png',
        is_chrf=True
    )
    
    print("Generating Rank Scaling Plots...")
    plot_rank_scaling(
        'paraphrase_combined.csv',
        'best_dev_acc_max' if 'best_dev_acc_max' in pd.read_csv('paraphrase_combined.csv').columns else 'best_dev_acc',
        'Validation Accuracy',
        f'{out_dir}/paraphrase_rank_scaling.png',
        is_chrf=False
    )
    plot_rank_scaling(
        'sonnet_combined.csv',
        'best_chrf_max' if 'best_chrf_max' in pd.read_csv('sonnet_combined.csv').columns else 'best_chrf',
        'Validation chrF',
        f'{out_dir}/sonnet_rank_scaling.png',
        is_chrf=True
    )
    
    print("Generating rsLoRA Rank Sweep for Sonnets...")
    plot_rslora_rank_sweep(
        'sonnet_combined.csv',
        'best_chrf_max' if 'best_chrf_max' in pd.read_csv('sonnet_combined.csv').columns else 'best_chrf',
        'Validation chrF',
        f'{out_dir}/sonnet_rslora_sweep.png'
    )

if __name__ == '__main__':
    main()
