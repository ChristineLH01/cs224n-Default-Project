import pandas as pd
import os
import numpy as np

def compute_gpu_metrics(df):
    df['num_gpus'] = df.get('train_meta.num_gpus', 1).fillna(1).astype(int)
    # Total train time in GPU-hours
    if 'train_meta.total_train_time_sec' in df.columns:
        df['gpu_hours'] = df['train_meta.total_train_time_sec'] * df['num_gpus'] / 3600.0
    else:
        df['gpu_hours'] = 0.0
    return df

def write_formats(df, output_dir, filename_prefix, name):
    csv_path = os.path.join(output_dir, f'{filename_prefix}_{name}.csv')
    md_path = os.path.join(output_dir, f'{filename_prefix}_{name}.md')
    tex_path = os.path.join(output_dir, f'{filename_prefix}_{name}.tex')
    
    df.to_csv(csv_path, index=False)
    with open(md_path, 'w') as f:
        f.write(df.to_markdown(index=False))
    with open(tex_path, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))

def generate_main_table(df, metric_col, metric_name, output_dir, filename_prefix):
    grouped = df.groupby(['model_size', 'additional_data_used', 'method']).agg({
        metric_col: ['mean', 'std', 'max'],
        'gpu_hours': 'mean'
    }).reset_index()
    
    grouped.columns = ['Model', 'Augmentation', 'Method', f'{metric_name}_Mean', f'{metric_name}_Std', f'{metric_name}_Max', 'GPU_Hours']
    grouped[f'{metric_name}_Mean'] = grouped[f'{metric_name}_Mean'].round(4)
    grouped[f'{metric_name}_Std'] = grouped[f'{metric_name}_Std'].round(4)
    grouped[f'{metric_name}_Max'] = grouped[f'{metric_name}_Max'].round(4)
    grouped['GPU_Hours'] = grouped['GPU_Hours'].round(2)
    grouped['Augmentation'] = grouped['Augmentation'].replace({True: 'Yes', False: 'No'})
    grouped['Method'] = grouped['Method'].str.capitalize()
    grouped = grouped.sort_values(by=['Model', 'Augmentation', 'Method'])
    write_formats(grouped, output_dir, filename_prefix, 'main')

def generate_lora_ablation_table(df, metric_col, metric_name, output_dir, filename_prefix):
    lora_df = df[df['method'] == 'lora'].copy()
    if len(lora_df) == 0: return
        
    lora_df['Target'] = 'Unknown'
    mask_attn = (lora_df['apply_lora_to_attention'] == True) & (lora_df['apply_lora_to_mlp'] != True)
    mask_mlp = (lora_df['apply_lora_to_attention'] != True) & (lora_df['apply_lora_to_mlp'] == True)
    mask_both = (lora_df['apply_lora_to_attention'] == True) & (lora_df['apply_lora_to_mlp'] == True)
    lora_df.loc[mask_attn, 'Target'] = 'Attn Only'
    lora_df.loc[mask_mlp, 'Target'] = 'MLP Only'
    lora_df.loc[mask_both, 'Target'] = 'Attn + MLP'
    
    grouped = lora_df.groupby(['model_size', 'lora_rank', 'lora_alpha', 'Target']).agg({
        metric_col: ['mean', 'max'],
        'gpu_hours': 'mean'
    }).reset_index()
    
    grouped.columns = ['Model', 'Rank', 'Alpha', 'Target', f'{metric_name}_Mean', f'{metric_name}_Max', 'GPU_Hours']
    grouped = grouped[grouped['Target'] != 'Unknown']
    grouped[f'{metric_name}_Mean'] = grouped[f'{metric_name}_Mean'].round(4)
    grouped[f'{metric_name}_Max'] = grouped[f'{metric_name}_Max'].round(4)
    grouped['GPU_Hours'] = grouped['GPU_Hours'].round(2)
    grouped = grouped.sort_values(by=['Model', 'Rank', 'Alpha', 'Target'])
    write_formats(grouped, output_dir, filename_prefix, 'lora_ablation')

def generate_lr_ablation_table(df, metric_col, metric_name, output_dir, filename_prefix):
    grouped = df.groupby(['model_size', 'method', 'lr']).agg({
        metric_col: ['mean', 'max'],
    }).reset_index()
    
    grouped.columns = ['Model', 'Method', 'LR', f'{metric_name}_Mean', f'{metric_name}_Max']
    grouped['Method'] = grouped['Method'].str.capitalize()
    grouped[f'{metric_name}_Mean'] = grouped[f'{metric_name}_Mean'].round(4)
    grouped[f'{metric_name}_Max'] = grouped[f'{metric_name}_Max'].round(4)
    grouped = grouped.sort_values(by=['Model', 'Method', 'LR'])
    write_formats(grouped, output_dir, filename_prefix, 'lr_ablation')

def generate_comprehensive_table(df, metric_col, metric_name, output_dir, filename_prefix):
    # Select important columns to show raw data for the appendix
    cols_to_keep = [
        'model_size', 'additional_data_used', 'method', 'lr', 'epochs', 'batch_size', 
        'lora_rank', 'lora_alpha', 'lora_dropout', 'apply_lora_to_attention', 'apply_lora_to_mlp',
        'adalora_target_r', metric_col, 'gpu_hours'
    ]
    cols_present = [c for c in cols_to_keep if c in df.columns]
    comp_df = df[cols_present].copy()
    
    # Clean up boolean/NaNs for formatting
    comp_df = comp_df.fillna('-')
    if metric_col in comp_df.columns:
        comp_df[metric_col] = pd.to_numeric(comp_df[metric_col], errors='coerce').round(4)
    comp_df['gpu_hours'] = pd.to_numeric(comp_df['gpu_hours'], errors='coerce').round(2)
    comp_df = comp_df.sort_values(by=['model_size', 'method', metric_col], ascending=[True, True, False])
    
    # Rename columns for presentation
    rename_map = {
        'model_size': 'Model', 'additional_data_used': 'Augmented', 'method': 'Method',
        'lr': 'LR', 'epochs': 'Epochs', 'batch_size': 'BS', 'lora_rank': 'Rank', 
        'lora_alpha': 'Alpha', 'lora_dropout': 'Dropout', 'apply_lora_to_attention': 'Attn', 
        'apply_lora_to_mlp': 'MLP', 'adalora_target_r': 'Ada_Target_R',
        metric_col: metric_name, 'gpu_hours': 'GPU_Hours'
    }
    comp_df = comp_df.rename(columns=rename_map)
    write_formats(comp_df, output_dir, filename_prefix, 'comprehensive')

def main():
    para_path = 'paraphrase_combined.csv'
    sonnet_path = 'sonnet_combined.csv'
    
    out_dir = 'paper_tables'
    os.makedirs(out_dir, exist_ok=True)
    
    print("Reading data...")
    if os.path.exists(para_path):
        para_df = pd.read_csv(para_path)
        para_df = compute_gpu_metrics(para_df)
        generate_main_table(para_df, 'best_dev_acc', 'Best_Dev_Acc', out_dir, 'paraphrase')
        generate_lora_ablation_table(para_df, 'best_dev_acc', 'Best_Dev_Acc', out_dir, 'paraphrase')
        generate_lr_ablation_table(para_df, 'best_dev_acc', 'Best_Dev_Acc', out_dir, 'paraphrase')
        generate_comprehensive_table(para_df, 'best_dev_acc', 'Best_Dev_Acc', out_dir, 'paraphrase')
        print(f"Generated tables for Paraphrase Detection in {out_dir}/")
        
    if os.path.exists(sonnet_path):
        sonnet_df = pd.read_csv(sonnet_path)
        sonnet_df = compute_gpu_metrics(sonnet_df)
        generate_main_table(sonnet_df, 'best_chrf', 'Best_ChrF', out_dir, 'sonnet')
        generate_lora_ablation_table(sonnet_df, 'best_chrf', 'Best_ChrF', out_dir, 'sonnet')
        generate_lr_ablation_table(sonnet_df, 'best_chrf', 'Best_ChrF', out_dir, 'sonnet')
        generate_comprehensive_table(sonnet_df, 'best_chrf', 'Best_ChrF', out_dir, 'sonnet')
        print(f"Generated tables for Sonnet Generation in {out_dir}/")

if __name__ == "__main__":
    main()
