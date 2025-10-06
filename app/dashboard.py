import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Union
import requests
from datetime import datetime
import os

# API base URL - adjust this to your actual API endpoint
API_BASE_URL = "http://localhost:8000"
PIVOT_TABLE_INDEX = ['task', 'subtask', 'prompt_idx', 'prompt', 'difficulty', 'gold', 'avg_accuracy']

class TaskAnalyticsDashboard:
    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url

    def get_usecases(self):
        """Fetch available usecases from API"""
        try:
            response = requests.get(f"{self.api_base_url}/leaderboard/use_cases")
            if response.status_code == 200:
                return response.json().get("use_cases", [])
            return []
        except Exception as e:
            return [f"Error: {str(e)}"]
    
    def get_tasks(self):
        """Fetch available tasks from API"""
        try:
            response = requests.get(f"{self.api_base_url}/tasks/tasks")
            if response.status_code == 200:
                return response.json().get("tasks", [])
            return []
        except Exception as e:
            return [f"Error: {str(e)}"]
    
    def get_task_config(self):
        """Fetch task configuration"""
        try:
            response = requests.get(f"{self.api_base_url}/tasks/config")
            if response.status_code == 200:
                return response.json().get("task_config", {})
            return {}
        except Exception as e:
            return {"error": str(e)}
    
    def get_num_samples(self, task_name: str, limit: Union[int, None] = None):
        """Get number of samples for a task"""
        try:
            params = {"task_name": task_name}
            if limit:
                params["limit"] = limit
            response = requests.get(f"{self.api_base_url}/tasks/num_samples", params=params)
            if response.status_code == 200:
                return response.json()
            return {"error": "Failed to fetch samples"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_results(self, task: str, output_path: str, model: str):
        """Get accuracy results for a task"""
        try:
            params = {"task": task, "output_path": output_path, "model": model}
            response = requests.get(f"{self.api_base_url}/tasks/results", params=params)
            if response.status_code == 200:
                return response.json()
            return {"error": "Failed to fetch results"}
        except Exception as e:
            return {"error": str(e)}
        
    def fetch_leaderboard(self, output_path: str, tasks: List[str], models: List[str], min_num_samples: int, use_case:str = None):
        """Fetch leaderboard data"""
        try:
            params = {
                "eval_dir": output_path,
                "min_num_samples": min_num_samples
            }
            if models == ['']:
                models = []
                params['models'] = None
            if use_case:
                params['use_case'] = use_case
            for task in tasks:
                params.setdefault("tasks", []).append(task)
            for model in models:
                params.setdefault("models", []).append(model)

            headers = {
                "accept": "application/json"
            }
            
            response = requests.get(f"{self.api_base_url}/leaderboard/leaderboard", params=params, headers = headers)
            if response.status_code == 200:
                df_leaderboard = pd.DataFrame(response.json().get("leaderboard", []))
                detailed_data = pd.DataFrame(response.json().get("detailed_data", []))
                return df_leaderboard, detailed_data
            return {"error": "Failed to fetch leaderboard"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_accuracy_per_sample(self, task: str, output_path: str, 
                                model_names: List[str], prompt: Union[int, str, None] = None):
        """Get detailed accuracy per sample with responses"""
        try:
            # Build params with list handling for model_names
            
            # params = [("task", task), ("output_path", output_path)]

            if not model_names:
                model_names = os.listdir(output_path)

            params = {
                "task": task,
                "output_path":output_path, # "eval_results/ifeval_eval",
                "model_names": model_names,
            }
            
            headers = {
            "accept": "application/json"
        }
            
            # if model_names:
            #     for model in model_names:
            #         params.append(("model_names", model))
            
            if prompt is not None:
                params.append(("prompt", prompt))
            
            response = requests.get(f"{self.api_base_url}/tasks/accuracy_per_sample", params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return (
                    pd.DataFrame(data.get("pivot_table", [])),
                    pd.DataFrame(data.get("responses", []))
                )
            return None, None
        except Exception as e:
            print(f"Error: {e}")
            return None, None

# Initialize dashboard
dashboard = TaskAnalyticsDashboard(API_BASE_URL)


def create_multi_model_comparison_chart(all_results, model_names):
    """Create comparison chart across multiple models"""
    if not all_results or len(all_results) == 0:
        return go.Figure()
    
    fig = go.Figure()
    
    # Extract accuracy and num_samples for each model
    accuracies = []
    num_samples_list = []
    
    for results in all_results:
        if isinstance(results, list) and len(results) > 0:
            result = results[0]  # Use first run
            accuracies.append(result.get('accuracy', 0))
            num_samples_list.append(result.get('num_samples', 'N/A'))
        else:
            accuracies.append(0)
            num_samples_list.append('N/A')
    
    # Create bar chart with models on x-axis
    fig.add_trace(go.Bar(
        x=model_names,
        y=accuracies,
        text=[f"Accuracy: {acc:.2%}<br>Samples: {ns}" for acc, ns in zip(accuracies, num_samples_list)],
        hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.2%}<br>Samples: %{customdata}<extra></extra>',
        customdata=num_samples_list,
        marker=dict(
            color=accuracies,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Accuracy")
        )
    ))
    
    fig.update_layout(
        title="Multi-Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Accuracy",
        yaxis=dict(tickformat='.0%'),
        template="plotly_white",
        hovermode='closest'
    )
    
    return fig

def create_comparison_chart(df_pivot):
    """Create clean average accuracy comparison chart"""
    if df_pivot is not None and not df_pivot.empty:
        # Calculate average accuracy for each model
        model_columns = [col for col in df_pivot.columns if col not in PIVOT_TABLE_INDEX]
        
        if len(model_columns) == 0:
            return go.Figure()
        
        # Calculate average accuracy per model
        avg_accuracy = {}
        counts = {}
        for col in model_columns:
            valid_values = df_pivot[col].dropna()
            avg_accuracy[col] = df_pivot[col].mean()
            counts[col] = len(valid_values)
        
        if not avg_accuracy:
            return go.Figure()
        
        # Sort by accuracy for better visualization
        sorted_models = sorted(avg_accuracy.items(), key=lambda x: x[1], reverse=True)
        models = [m[0] for m in sorted_models]
        accuracies = [m[1] for m in sorted_models]
        row_counts = [counts[m] for m in models]
        
        # Create custom hover text
        hover_texts = [
            f"Model: {model}<br>Avg Accuracy: {acc:.2%}<br>Count: {cnt}"
            for model, acc, cnt in zip(models, accuracies, row_counts)
        ]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=accuracies,
                text=[f'{acc:.2%}' for acc in accuracies],
                textposition='auto',
                marker_color='lightblue',
                hovertext=hover_texts,
                hoverinfo='text'
            )
        ])
        
        fig.update_layout(
            title="Average Accuracy Comparison Across Models",
            xaxis_title="Model",
            yaxis_title="Average Accuracy",
            yaxis_tickformat='.0%',
            template="plotly_white",
            showlegend=False,
            height=500
        )
        
        return fig
    return go.Figure()

def create_prompt_distribution_chart(df_pivot):
    """Create distribution chart with prompt ID on x-axis and average accuracy on y-axis"""
    if df_pivot is None or df_pivot.empty or 'prompt_idx' not in df_pivot.columns:
        return go.Figure()
    
    # Get model columns (exclude metadata columns)
    numeric_cols = [col for col in df_pivot.columns if col not in PIVOT_TABLE_INDEX] #[col for col in model_columns if pd.api.types.is_numeric_dtype(df_pivot[col])]
    
    if len(numeric_cols) == 0:
        return go.Figure()
    

    
    # Calculate average accuracy per prompt
    # prompt_accuracy = df_pivot.groupby('prompt_idx')[numeric_cols].mean().mean(axis=1)
    grouped = df_pivot.groupby(['subtask','prompt_idx'])[numeric_cols].mean().mean(axis=1).reset_index()
    # prompt_counts = df_pivot.groupby('prompt_idx').size()
    
    # Create bar chart
    fig = go.Figure()
    # Add one trace per subtask
    subtasks = grouped['subtask'].unique()
    for sub in subtasks:
        sub_df = grouped[grouped['subtask'] == sub]
        fig.add_trace(go.Bar(
            x=sub_df['prompt_idx'],
            y=sub_df[0],  # the aggregated mean
            name=sub,
            visible=(sub == subtasks[0]),  # only show the first by default
            marker=dict(
                color=sub_df[0],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Accuracy", tickformat='.0%'),
                cmin=0, cmax=1
            ),
            text=[f'{acc:.3%}' for acc in sub_df[0]],
            textposition='auto',
            hovertemplate='<b>Prompt %{x}</b><br>Subtask: '+sub+'<br>Avg Accuracy: %{y:.2%}'
        ))

    # Add dropdown menu
    fig.update_layout(
        updatemenus=[dict(
            buttons=[
                dict(label=sub,
                     method="update",
                     args=[{"visible": [s == sub for s in subtasks]},
                           {"title": f"Prompt Accuracy (Subtask={sub})"}])
                for sub in subtasks
            ],
            direction="down",
            x=1.05, y=1.2
        )],
        title=f"Prompt Accuracy (Subtask={subtasks[0]})",
        xaxis_title="Prompt Index",
        yaxis_title="Average Accuracy"
    )

    # fig = go.Figure(data=[
    #     go.Bar(
    #         x=prompt_accuracy.index,
    #         y=prompt_accuracy.values,
    #         text=[f'{acc:.3%}' for acc in prompt_accuracy.values],
    #         textposition='auto',
    #         marker=dict(
    #             # color = 'lightblue',
    #             color=prompt_accuracy.values,
    #             colorscale='RdYlGn',
    #             showscale=True,
    #             colorbar=dict(title="Accuracy", tickformat='.0%'),
    #             cmin=0,   # explicitly start at 0
    #             cmax=1,   # end at 1 (100%)
    #         ),
    #         hovertemplate='<b>Prompt %{x}</b><br>Avg Accuracy: %{y:.2%}',
    #         # customdata=prompt_counts.values
    #     )
    # ])
    
    # fig.update_layout(
    #     title="Average Accuracy Distribution by Prompt ID",
    #     xaxis_title="Prompt ID",
    #     yaxis_title="Average Accuracy",
    #     yaxis_tickformat='.0%',
    #     template="plotly_white",
    #     showlegend=False,
    #     height=500,
    #     xaxis=dict(type='category')
    # )
    
    return fig

def plot_difficulty_distribution(df_pivot):
    """Plot a bar chart showing number of questions per difficulty level."""
    difficulty_summary = df_pivot['difficulty'].value_counts().to_dict()
    levels = list(difficulty_summary.keys())
    counts = list(difficulty_summary.values())
    
    fig = go.Figure(
        data=[go.Bar(
            x=levels,
            y=counts,
            text=counts,
            textposition="auto",
            marker=dict(
                color=["green", "gold", "red"],  # easy, medium, hard
                opacity=0.8
            )
        )]
    )
    
    fig.update_layout(
        title="Question Difficulty Distribution",
        xaxis_title="Difficulty Level",
        yaxis_title="Number of Questions",
        template="plotly_white"
    )
    
    return fig


def create_leaderboard(tasks: list, output_path: str, models: str, min_num_samples: int, use_case:str = None):
    """Analyze task results and generate visualizations for multiple models"""
    if not tasks or not output_path :
        return None, "Please fill in all required fields", pd.DataFrame()
    
    # Parse model names
    if models is None:
        model_list = []
    else:
        model_list = [m.strip() for m in models.split(",")]
    
    # if len(model_list) == 0:
    #     return None, "Please provide at least one model name", pd.DataFrame()
    
    # Fetch leaderboard data
    df_leaderboard, detailed_data = dashboard.fetch_leaderboard(output_path, tasks, model_list, min_num_samples, use_case)
    # df_leaderboard['Average Score'] = df_leaderboard[tasks].mean(axis=1, skipna = True)

    df_leaderboard.rename(columns = {'usecase_score': 'Aggregated Score', 'task_weights': 'Aggregated Weights'}, inplace=True)
    df_leaderboard.sort_values(by = 'Aggregated Score', ascending=False).reset_index(drop=True)
    columns = ['model', 'Aggregated Score'] + tasks
    df_leaderboard = df_leaderboard[columns]
    for col in tasks + ['Aggregated Score']:
        if col in df_leaderboard.columns:
            df_leaderboard[col] = df_leaderboard.get(col, pd.NA).apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    return df_leaderboard
    

def analyze_detailed(task: str, output_path: str, model_names: str, prompt: str = None, difficulty:List = None):
    """Analyze detailed per-sample accuracy"""
    if not task or not output_path:
        return None, None, pd.DataFrame(), "Please fill in required fields"
    
    models_list = [m.strip() for m in model_names.split(",")] if model_names else []
    prompt_val = None
    if prompt:
        try:
            prompt_val = int(prompt)
        except:
            prompt_val = prompt
    
    df_pivot, df_responses = dashboard.get_accuracy_per_sample(task, output_path, models_list, prompt_val)
    
    if df_pivot is None or df_responses is None:
        return None, None, pd.DataFrame(), "Error fetching detailed data"
    
    if difficulty:
        df_pivot = df_pivot.loc[df_pivot['difficulty'].isin(difficulty)].reset_index(drop = True)
    
    chart = create_comparison_chart(df_pivot)
    dist_chart = create_prompt_distribution_chart(df_pivot)
    difficulty_chart = plot_difficulty_distribution(df_pivot)
    
    # Generate summary statistics
    summary = "## Detailed Analysis Summary\n\n"
    if not df_pivot.empty:
        summary += f"**Total Samples**: {len(df_pivot)}\n\n"
        
        # Calculate average accuracy by model
        numeric_cols = [col for col in df_pivot.columns if col not in ['task', 'subtask', 'prompt_idx', 'prompt', 'gold', 'difficulty']] # [col for col in model_columns if pd.api.types.is_numeric_dtype(df_pivot[col])]
        
        if len(numeric_cols) > 0:
            summary += "### Average Accuracy by Model\n\n"
            summary += "| Model | Avg Accuracy | Correct | Total |\n"
            summary += "|-------|--------------|---------|-------|\n"
            
            # Sort by accuracy
            model_stats = []
            for col in numeric_cols:
                avg = df_pivot[col].mean()
                correct = df_pivot[col].sum()
                total = len(df_pivot) - int(df_pivot[col].isna().sum())
                model_stats.append((col, avg, correct, total))
            
            model_stats.sort(key=lambda x: x[1], reverse=True)
            
            for col, avg, correct, total in model_stats:
                summary += f"| {col} | {avg:.2%} | {int(correct)} | {total} |\n"
            
            summary += "\n"
            
            if len(numeric_cols) > 1:
                summary += "### Performance Insights\n\n"

                # Agreement rate - all True OR all False
                all_true = (df_pivot[numeric_cols].all(axis=1)).sum()
                all_false = (~df_pivot[numeric_cols].any(axis=1)).sum()
                agreement = all_true + all_false
                summary += f"**Full Agreement**: {agreement}/{len(df_pivot)} samples ({agreement/len(df_pivot):.1%})\n"
                summary += f"  - All True: {all_true} samples\n"
                summary += f"  - All False: {all_false} samples\n\n"

                # Disagreement rate - mixed True and False
                disagreement = len(df_pivot) - agreement
                summary += f"**Disagreement**: {disagreement}/{len(df_pivot)} samples ({disagreement/len(df_pivot):.1%})\n\n"
    
    return chart, dist_chart, difficulty_chart, df_pivot,  summary

# def categorize_question_difficulty(df_pivot):
#     """Categorize each question into easy/medium/hard based on mean accuracy across models."""
#     # 1️⃣ Identify model columns (exclude metadata)
#     exclude_cols = ['sample_id', 'doc_id', 'index', 'prompt_idx', 'task', 'subtask', 'prompt', 'gold']
#     model_cols = [c for c in df_pivot.columns if c not in exclude_cols]
    
#     # 2️⃣ Compute mean accuracy across models (row-wise)
#     df_pivot['avg_accuracy'] = df_pivot[model_cols].mean(axis=1)
    
#     # 3️⃣ Categorize into buckets
#     def bucket(acc):
#         if acc >= 0.7: return 'Easy (70–100%)'
#         elif acc >= 0.3: return 'Medium (30–70%)'
#         else: return 'Hard (0–30%)'
        
#     df_pivot['difficulty'] = df_pivot['avg_accuracy'].apply(bucket)
    
#     # 4️⃣ Summary counts
#     summary = df_pivot['difficulty'].value_counts().to_dict()
    
#     return df_pivot, summary

def get_task_list():
    """Get available tasks"""
    tasks = dashboard.get_tasks()
    return gr.Dropdown(choices=tasks, value=tasks[0] if tasks else None)

def get_usecases_list():
    """Get available use-cases"""
    usecases = dashboard.get_usecases()
    return gr.Dropdown(choices=usecases, value=usecases[0] if usecases else None)

# Create Gradio Interface
with gr.Blocks(title="Task Analytics Dashboard", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📊 Task Analytics Dashboard")
    gr.Markdown("Analyze model performance on various tasks with interactive charts and detailed breakdowns")
    
    # Store pivot data for filtering
    pivot_data_state = gr.State(value=pd.DataFrame())
    
    with gr.Tabs():
        # Tab 1: Overall Results
        with gr.Tab("📈 Overall Results"):
            
            gr.Markdown("## Task Performance Overview")
            with gr.Row():
                with gr.Column(scale=1):
                    usecase_input = gr.Dropdown(
                        choices=dashboard.get_usecases(),
                        label="Select Use Case",
                        info="Choose a use-case to analyze",
                        multiselect=False
                    )
                    task_input = gr.Dropdown(
                        choices=dashboard.get_tasks(),
                        label="Select Task",
                        info="Choose a task to analyze",
                        multiselect=True
                    )
                   
                    eval_path_input = gr.Textbox(
                        label="Eval Path",
                        placeholder="/path/to/lm-eval-results",
                        info="Path to LM eval harness results directory",
                        value = "eval_results"
                    )
                    model_input = gr.Textbox(
                        label="Model input",
                        placeholder="model A, model B, model C",
                        info="Models, delimited by ,",
                        value = None,
                    )
                    platform = gr.Textbox(
                        label="Platform",
                        placeholder="tower",
                        info="Platform",
                        value = 'tower',
                    )
                    device = gr.Textbox(
                        label="Device",
                        placeholder="GPU",
                        info="Device",
                        value = 'GPU',
                    )
                    chipset = gr.Textbox(
                        label="Chipset",
                        placeholder="chipset",
                        info="Chipset",
                        value = 'NVIDIA',
                    )
                    inference_service = gr.Textbox(
                        label="Inference service",
                        placeholder="Inference service",
                        info="Inference service",
                        value = "ollama",
                    )
                    precision = gr.Textbox(
                        label="Precision",
                        placeholder="4bit",
                        info="Precision of weights",
                        value = '4bit',
                    )
                    min_num_samples = gr.Number(
                        label="Min Number of Samples",
                        value=0
                    )
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                
                with gr.Column(scale=2):
                    leaderboard_table = gr.Dataframe(label="Leaderboard", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    results_text = gr.Markdown("Results will appear here...")
                # with gr.Column():
                #     results_table = gr.Dataframe(label="Results Table")
            
            analyze_btn.click(
                fn=create_leaderboard,
                inputs=[task_input, eval_path_input, model_input, min_num_samples, usecase_input],
                outputs=[leaderboard_table]
            )
        
        # Tab 2: Detailed Analysis
        with gr.Tab("🔬 Detailed Per-Sample Analysis"):


            
            gr.Markdown("## Sample-by-Sample Model Comparison")
            with gr.Row():
                with gr.Column(scale=1):
                    task_input2 = gr.Dropdown(
                        choices=dashboard.get_tasks(),
                        label="Select Task"
                    )
                    output_path_input2 = gr.Textbox(
                        label="Output Path",
                        placeholder="/path/to/lm-eval-results",
                        value = 'eval_results/ifeval_eval'
                    )

                    model_names_input = gr.Textbox(
                        label="Model Names (comma-separated)",
                        placeholder="model1, model2, model3",
                        info="Leave empty for all models"
                    )
                    platform2 = gr.Textbox(
                        label="Platform",
                        placeholder="tower",
                        info="Platform",
                        value = 'tower',
                    )
                    device2 = gr.Textbox(
                        label="Device",
                        placeholder="GPU",
                        info="Device",
                        value = 'GPU',
                    )
                    chipset2 = gr.Textbox(
                        label="Chipset",
                        placeholder="chipset",
                        info="Chipset",
                        value = 'NVIDIA',
                    )
                    inference_service2 = gr.Textbox(
                        label="Inference service",
                        placeholder="Inference service",
                        info="Inference service",
                        value = "ollama",
                    )
                    precision2 = gr.Textbox(
                        label="Precision",
                        placeholder="4bit",
                        info="Precision of weights",
                        value = '4bit',
                    )
                    prompt_input = gr.Textbox(
                        label="Prompt Filter (optional)",
                        placeholder="Enter prompt index or string"
                    )
                    difficulty_input = gr.Dropdown(
                        choices=['Easy (70–100%)', 'Medium (30–70%)', 'Hard (0–30%)'],
                        label="Select difficulty of prompts",
                        info="Filter by difficulty",
                        multiselect=True,
                        value = None
                    )
                    analyze_detailed_btn = gr.Button("🔍 Analyze Detailed", variant="primary")
            
                with gr.Column(scale=2):
                    comparison_chart = gr.Plot(label="Model Comparison")

                    difficulty_chart = gr.Plot(label="Question difficulty Distribution")
            
            with gr.Row():
                distribution_chart = gr.Plot(label="Prompt Distribution")

                # summary_text = gr.Markdown("Summary will appear here...")
                # difficulty_chart = gr.Plot(label="Question difficulty Distribution")
            
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_selector = gr.Dropdown(
                        label="Filter by Prompt ID",
                        choices=[],
                        value=None,
                        interactive=True,
                        info="Select a prompt to filter the table below"
                    )
                    subtask_selector = gr.Dropdown(
                        label="Filter by subtask",
                        choices=[],
                        value=None,
                        interactive=True,
                        info="Select a subtask to filter the table below"
                    )
                    reset_filter_btn = gr.Button("Reset Filter", size="sm")
                with gr.Column(scale=3):
                    filter_info = gr.Markdown("Select a prompt ID from the dropdown to filter")
            
            with gr.Row():
                pivot_table = gr.Dataframe(label="Pivot Table - Accuracy per Sample", interactive=False)
            
            # update output path
            def update_output_path(task):
                return f"eval_results/{task}_eval"
    
            task_input2.change(
                    fn=update_output_path,
                    inputs=task_input2,
                    outputs=output_path_input2
                )
                
            # Handle analyze button click
            def handle_analyze(task, output_path, model_names, prompt, difficulty):
                chart, dist_chart, difficulty_chart, pivot_df, summary = analyze_detailed(task, output_path, model_names, prompt, difficulty)
                
                # Get unique prompt IDs for dropdown
                prompt_choices = []
                if pivot_df is not None and not pivot_df.empty and 'prompt_idx' in pivot_df.columns:
                    prompt_choices = sorted(pivot_df['prompt_idx'].unique().tolist())
                    subtask_choices = sorted(pivot_df['subtask'].unique().tolist())
                
                return (chart, dist_chart, difficulty_chart, pivot_df,  pivot_df, 
                        "Select a prompt ID from the dropdown to filter", 
                        gr.Dropdown(choices=prompt_choices, value=None),
                        gr.Dropdown(choices=subtask_choices, value=None),
                        )
            
            analyze_detailed_btn.click(
                fn=handle_analyze,
                inputs=[task_input2, output_path_input2, model_names_input, prompt_input, difficulty_input],
                outputs=[comparison_chart, distribution_chart, difficulty_chart, pivot_data_state, 
                        pivot_table, filter_info, prompt_selector, subtask_selector]
            )
            
            # Handle prompt selection for filtering
            def on_prompt_select(prompt_id, df_pivot):
                if prompt_id is None:
                    return df_pivot, "Select a prompt ID from the dropdown to filter"
                
                if df_pivot is None or df_pivot.empty:
                    return df_pivot, "No data available"
                
                # Filter dataframe
                filtered_df = df_pivot[df_pivot['prompt_idx'] == prompt_id].copy()
                
                if filtered_df.empty:
                    return df_pivot, f"No data found for Prompt ID: {prompt_id}"
                
                info_text = f"**Filtered by Prompt ID: {prompt_id}**\n\n"
                info_text += f"Total samples with this prompt: {len(filtered_df)}\n\n"
                
                # Calculate accuracy for each model in this prompt
                model_columns = [col for col in filtered_df.columns if col not in ['sample_id', 'doc_id', 'index', 'prompt_idx']]
                numeric_cols = [col for col in model_columns if pd.api.types.is_numeric_dtype(filtered_df[col])]
                
                if numeric_cols:
                    info_text += "### Model Performance for this Prompt:\n\n"
                    info_text += "| Model | Accuracy | Correct | Total |\n"
                    info_text += "|-------|----------|---------|-------|\n"
                    
                    for col in numeric_cols:
                        acc = filtered_df[col].mean()
                        correct = filtered_df[col].sum()
                        total = len(filtered_df)
                        info_text += f"| {col} | {acc:.2%} | {int(correct)} | {total} |\n"
                
                return filtered_df, info_text
            
             # Handle prompt selection for filtering
            def on_subtask_select(subtask, df_pivot):
                if subtask is None:
                    return df_pivot, "Select a Subtask from the dropdown to filter"
                
                if df_pivot is None or df_pivot.empty:
                    return df_pivot, "No data available"
                
                # Filter dataframe
                filtered_df = df_pivot[df_pivot['subtask'] == subtask].copy()
                
                if filtered_df.empty:
                    return df_pivot, f"No data found for Subtask: {subtask}"
                
                info_text = f"**Filtered by Subtask: {subtask}**\n\n"
                info_text += f"Total samples with this prompt: {len(filtered_df)}\n\n"
                
                # Calculate accuracy for each model in this prompt
                model_columns = [col for col in filtered_df.columns if col not in ['sample_id', 'doc_id', 'index', 'prompt_idx']]
                numeric_cols = [col for col in model_columns if pd.api.types.is_numeric_dtype(filtered_df[col])]
                
                if numeric_cols:
                    info_text += "### Model Performance for this Prompt:\n\n"
                    info_text += "| Model | Accuracy | Correct | Total |\n"
                    info_text += "|-------|----------|---------|-------|\n"
                    
                    for col in numeric_cols:
                        acc = filtered_df[col].mean()
                        correct = filtered_df[col].sum()
                        total = len(filtered_df)
                        info_text += f"| {col} | {acc:.2%} | {int(correct)} | {total} |\n"
                
                return filtered_df, info_text
            
            prompt_selector.change(
                fn=on_prompt_select,
                inputs=[prompt_selector, pivot_data_state],
                outputs=[pivot_table, filter_info]
            )
            
            # Reset filter button
            reset_filter_btn.click(
                fn=lambda df: (df, "Select a prompt ID from the dropdown to filter"),
                inputs=[pivot_data_state],
                outputs=[pivot_table, filter_info]
            ).then(
                fn=lambda: None,
                outputs=[prompt_selector]
            )
        


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Task Analytics Dashboard...")
    print("=" * 60)
    print(f"📡 API Base URL: {API_BASE_URL}")
    print("🌐 Dashboard will be available at: http://localhost:7860")
    print("=" * 60)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)