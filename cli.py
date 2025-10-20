# =============================================================================
# Dynamic Microgrid Resilience: Deep Q-Network Energy Optimization (v7)
# =============================================================================
# Purpose: Production-ready DQN framework (7D state, 5 actions) for 
#          microgrid optimization, achieving 100% load met and 190% ROI
#          via stochastic PV/wind/load modeling, aggressive evening 
#          management (80.44% SOC), economic analysis, and NEC 2023-compliant 
#          MEP/BIM integration.
#
# Version: 7.0 (Aggressive DQN Edition)
# Author: Alfonso Davila - Electrical Engineer, Power Distribution Systems, Revit MEP Dynamo BIM Expert
# Contact: davila.alfonso@gmail.com - www.linkedin.com/in/alfonso-davila-3a121087
# Repository: https://github.com/DynMEP/dynamic-microgrid-resilience
# License: MIT License (see LICENSE file in repository)
# Last Updated: October 15, 2025
#
# System Features
#   - Deep Q-Network with experience replay and target network
#   - Stochastic environment (PV 5kW, wind 3kW, battery 18kWh)
#   - Granular battery control (5 actions: charge/discharge at 50%/100%, hold)
#   - Aggressive rewards (-400 evening fail, +250 success) for 80.44% evening SOC
#   - Economic analyzer: 4.3yr payback, 20-year NPV
#   - CLI with visualization (--plot), auto-model loading, parallel processing
#
# Dependencies
#   Python 3.8+, PyTorch 2.0+ (CUDA 11.8+), NumPy 1.24+, Pandas 2.0+, Matplotlib 3.7+
#
# Quick Start:
#   Train:    python cli.py train --episodes 500 --plot --economics
#   Evaluate: python cli.py evaluate --auto-latest --scenarios 20 --plot
#   Compare:  python cli.py compare --capacities 13 15 18 20 --parallel --plot
# =============================================================================

import argparse
import sys
import os
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
from pathlib import Path
import glob
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import threading
import time
import json

from Dynamic_Microgrid_Resilience_v7 import (
    MicrogridConfig, AggressiveMicrogridSimulator, 
    EconomicAnalyzer, DQNAgent, BatterySystem,
    StochasticTimeSeriesGenerator
)

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

class TrainingVisualizer:
    
    def __init__(self, output_dir='plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444'
        }
    
    def plot_training_progress(self, training_df, validation_df, prefix='training'):
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold', y=0.995)
        
        # 1. Load Met Rate
        ax = axes[0, 0]
        ax.plot(training_df['episode'], training_df['load_met_rate'], 
                alpha=0.3, color=self.colors['primary'], label='Training')
        if validation_df is not None and len(validation_df) > 0:
            ax.plot(validation_df['episode'], validation_df['load_met_rate'], 
                    color=self.colors['success'], linewidth=2, marker='o', 
                    markersize=4, label='Validation')
            # Mark best episodes
            best_episodes = validation_df[validation_df['is_best'] == True]
            if len(best_episodes) > 0:
                ax.scatter(best_episodes['episode'], best_episodes['load_met_rate'],
                          color=self.colors['danger'], s=150, marker='*', 
                          zorder=5, label='New Best', edgecolors='black', linewidth=1)
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Target')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Load Met Rate (%)')
        ax.set_title('Load Coverage Convergence', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # 2. Unmet Energy
        ax = axes[0, 1]
        ax.plot(training_df['episode'], training_df['unmet_energy'], 
                alpha=0.3, color=self.colors['primary'])
        if validation_df is not None and len(validation_df) > 0:
            ax.plot(validation_df['episode'], validation_df['unmet_energy'], 
                    color=self.colors['danger'], linewidth=2, marker='o', markersize=4)
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Target <0.5 kWh')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Unmet Energy (kWh)')
        ax.set_title('Unmet Energy Reduction', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Epsilon Decay
        ax = axes[0, 2]
        ax.plot(training_df['episode'], training_df['epsilon'], 
                color=self.colors['secondary'], linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon (Exploration Rate)')
        ax.set_title('Exploration vs Exploitation', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 4. Training Loss
        ax = axes[1, 0]
        if 'avg_loss' in training_df.columns:
            window = min(50, len(training_df) // 20)
            if window > 1:
                smoothed_loss = training_df['avg_loss'].rolling(window=window, center=True).mean()
                ax.plot(training_df['episode'], smoothed_loss, 
                        color=self.colors['primary'], linewidth=2, label=f'Smoothed (window={window})')
                ax.plot(training_df['episode'], training_df['avg_loss'], 
                        alpha=0.2, color=self.colors['primary'], label='Raw')
            else:
                ax.plot(training_df['episode'], training_df['avg_loss'], 
                        color=self.colors['primary'], linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Loss data not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 5. Reward Progress
        ax = axes[1, 1]
        ax.plot(training_df['episode'], training_df['total_reward'], 
                alpha=0.4, color=self.colors['primary'])
        window = min(50, len(training_df) // 20)
        if window > 1:
            smoothed_reward = training_df['total_reward'].rolling(window=window, center=True).mean()
            ax.plot(training_df['episode'], smoothed_reward, 
                    color=self.colors['success'], linewidth=2, label=f'Smoothed (window={window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Cumulative Reward', fontweight='bold')
        if window > 1:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Performance Summary
        ax = axes[1, 2]
        if validation_df is not None and len(validation_df) > 0:
            best_val = validation_df.loc[validation_df['load_met_rate'].idxmax()]
            final_val = validation_df.iloc[-1]
            
            metrics = {
                'Best Load Met': f"{best_val['load_met_rate']:.1f}%",
                'Best Episode': f"{int(best_val['episode'])}",
                'Final Load Met': f"{final_val['load_met_rate']:.1f}%",
                'Final Unmet': f"{final_val['unmet_energy']:.3f} kWh",
                'Total Episodes': f"{len(training_df)}",
                'Converged': '[YES]' if final_val['load_met_rate'] >= 95 else '[NO]'  # ← ASCII
            }
            
            y_pos = 0.85
            for key, value in metrics.items():
                ax.text(0.1, y_pos, f'{key}:', fontweight='bold', fontsize=10, transform=ax.transAxes)
                ax.text(0.6, y_pos, value, fontsize=10, transform=ax.transAxes)
                y_pos -= 0.12
        else:
            ax.text(0.5, 0.5, 'Validation data not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{prefix}_progress.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {output_path}")
        plt.close()
        
        return output_path
    
    def plot_hourly_performance(self, hourly_df, prefix='hourly'):
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Hourly Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
        
        hours = hourly_df['Hour']
        
        # 1. Power Flow
        ax = axes[0, 0]
        ax.fill_between(hours, 0, hourly_df['Avg_PV_Power_kW'], 
                        alpha=0.6, color='#FDB813', label='Solar PV')
        ax.fill_between(hours, hourly_df['Avg_PV_Power_kW'], 
                        hourly_df['Avg_PV_Power_kW'] + hourly_df['Avg_Wind_Power_kW'],
                        alpha=0.6, color='#4A90E2', label='Wind')
        ax.plot(hours, hourly_df['Avg_Load_Demand_kW'], 
               color='#E74C3C', linewidth=3, label='Load Demand', linestyle='--')
        ax.axvspan(18, 22, alpha=0.1, color='red', label='Evening Peak')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Power (kW)')
        ax.set_title('Power Generation & Demand', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 23])
        
        # 2. Battery SOC
        ax = axes[0, 1]
        ax.plot(hours, hourly_df['Avg_Battery_SOC_Percent'], 
               color=self.colors['primary'], linewidth=3, marker='o', markersize=4)
        ax.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Target Evening SOC')
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Min SOC')
        ax.axvspan(15, 17, alpha=0.1, color='orange', label='Pre-Evening Prep')
        ax.axvspan(18, 22, alpha=0.1, color='red', label='Evening Peak')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Battery SOC (%)')
        ax.set_title('Battery State of Charge', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 23])
        ax.set_ylim([0, 100])
        
        # 3. Load Coverage
        ax = axes[1, 0]
        colors = ['#10b981' if x == 100 else '#f59e0b' if x >= 95 else '#ef4444' 
                 for x in hourly_df['Load_Met_Rate_Percent']]
        bars = ax.bar(hours, hourly_df['Load_Met_Rate_Percent'], color=colors, alpha=0.8)
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% Target')
        ax.axvspan(18, 22, alpha=0.1, color='red', label='Evening Peak')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Load Met Rate (%)')
        ax.set_title('Hourly Load Coverage', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim([-0.5, 23.5])
        ax.set_ylim([0, 105])
        
        # 4. Resilience Score
        ax = axes[1, 1]
        ax.plot(hours, hourly_df['Avg_Resilience_Score'], 
               color=self.colors['success'], linewidth=3, marker='s', markersize=5)
        ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Target Resilience')
        ax.axvspan(18, 22, alpha=0.1, color='red', label='Evening Peak')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Resilience Score')
        ax.set_title('System Resilience', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 23])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{prefix}_performance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {output_path}")
        plt.close()
        
        return output_path
    
    def plot_economic_analysis(self, economics_data, prefix='economics'):
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Economic Analysis', fontsize=16, fontweight='bold', y=1.02)
        
        # 1. Cost Breakdown
        ax = axes[0]
        capex = economics_data['capex']
        costs = {
            'Solar PV': capex['pv'],
            'Wind Turbine': capex['wind'],
            'Battery': capex['battery'],
            'Inverter': capex['inverter'],
            'Installation': capex['installation']
        }
        
        colors_pie = ['#FDB813', '#4A90E2', '#10b981', '#f59e0b', '#9CA3AF']
        wedges, texts, autotexts = ax.pie(costs.values(), labels=costs.keys(), 
                                           autopct='%1.1f%%', startangle=90,
                                           colors=colors_pie, explode=[0.05]*len(costs))
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title(f'Capital Cost Breakdown\nTotal: ${capex["total"]:,.0f}', fontweight='bold')
        
        # 2. 20-Year Cash Flow
        ax = axes[1]
        annual_savings = economics_data['annual_savings']['net_annual_savings']
        years = np.arange(1, 21)
        discount_rate = 0.05
        
        cash_flow = [annual_savings / ((1 + discount_rate) ** year) for year in years]
        cumulative = np.cumsum(cash_flow)
        
        ax.bar(years, cash_flow, alpha=0.6, color=self.colors['success'], label='Annual NPV')
        ax.plot(years, cumulative, color=self.colors['danger'], linewidth=3, 
               marker='o', markersize=5, label='Cumulative NPV')
        ax.axhline(y=capex['total'], color='orange', linestyle='--', linewidth=2,
                  label=f'Initial Investment (${capex["total"]:,.0f})')
        
        payback_years = economics_data['payback_years']
        if payback_years <= 20:
            payback_idx = int(payback_years) - 1
            ax.axvline(x=payback_years, color='purple', linestyle=':', linewidth=2,
                      label=f'Payback ({payback_years:.1f} years)')
            ax.scatter([payback_years], [cumulative[payback_idx]], 
                      color='purple', s=200, marker='*', zorder=5, edgecolors='black')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Value ($)')
        ax.set_title('20-Year Cash Flow Analysis', fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 21])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{prefix}_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💰 Saved: {output_path}")
        plt.close()
        
        return output_path

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class ModelManager:
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def get_latest_model(self):
        model_files = glob.glob(os.path.join(self.models_dir, '*_model.pt'))
        
        if not model_files:
            model_files = glob.glob('*_model.pt')
        
        if not model_files:
            return None
        
        latest = max(model_files, key=os.path.getmtime)
        return latest
    
    def list_models(self):
        model_files = glob.glob(os.path.join(self.models_dir, '*_model.pt'))
        model_files.extend(glob.glob('*_model.pt'))
        
        models_info = []
        for model_path in model_files:
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                perf = checkpoint.get('performance', {})
                timestamp = checkpoint.get('timestamp', 'unknown')
                
                models_info.append({
                    'path': model_path,
                    'timestamp': timestamp,
                    'load_met_rate': perf.get('load_met_rate', 0),
                    'size_mb': os.path.getsize(model_path) / (1024 * 1024)
                })
            except:
                continue
        
        models_info.sort(key=lambda x: x['timestamp'], reverse=True)
        return models_info

# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

def run_single_comparison(args):
    capacity, episodes, worker_id = args
    
    print(f"[Worker {worker_id}] Testing {capacity} kWh battery...")
    
    config = MicrogridConfig()
    config.BATTERY_CAPACITY = capacity
    config.EPISODES = episodes
    
    simulator = AggressiveMicrogridSimulator(config)
    simulator.run(verbose=False)
    
    if simulator.validation_results:
        best_val = max(simulator.validation_results, key=lambda x: x['load_met_rate'])
        
        battery_cost = capacity * 450
        total_cost = battery_cost + (5.0 * 1200) + (3.0 * 2500) + (9.2 * 300)
        total_cost *= 1.25
        
        return {
            'capacity_kwh': capacity,
            'load_met_rate': best_val['load_met_rate'],
            'unmet_energy': best_val['unmet_energy'],
            'battery_cost': battery_cost,
            'total_system_cost': total_cost,
            'cost_per_percent': total_cost / best_val['load_met_rate'] if best_val['load_met_rate'] > 0 else float('inf')
        }
    
    return None

# ============================================================================
# REAL-TIME MONITORING
# ============================================================================

class RealTimeMonitor:
    
    def __init__(self, refresh_interval=2):
        self.refresh_interval = refresh_interval
        self.running = False
        self.latest_stats = {}
    
    def start(self, simulator):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, args=(simulator,))
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
    
    def _monitor_loop(self, simulator):
        while self.running:
            if simulator.episode_stats:
                latest = simulator.episode_stats[-1]
                self.latest_stats = latest
                
                os.system('cls' if os.name == 'nt' else 'clear')
                print("="*70)
                print("⚡ REAL-TIME TRAINING MONITOR")
                print("="*70)
                print(f"\nEpisode:          {latest['episode']}")
                print(f"Load Met Rate:    {latest['load_met_rate']:.2f}%")
                print(f"Unmet Energy:     {latest['unmet_energy']:.3f} kWh")
                print(f"Total Reward:     {latest['total_reward']:.1f}")
                print(f"Epsilon:          {latest['epsilon']:.6f}")
                if 'avg_loss' in latest:
                    print(f"Avg Loss:         {latest['avg_loss']:.6f}")
                
                if simulator.validation_results:
                    best_val = max(simulator.validation_results, key=lambda x: x['load_met_rate'])
                    print(f"\n🏆 Best Validation: {best_val['load_met_rate']:.2f}% @ Episode {best_val['episode']}")
                
                print("\nPress Ctrl+C to stop training early")
                print("="*70)
            
            time.sleep(self.refresh_interval)

# ============================================================================
# CLI COMMANDS
# ============================================================================

def run_demo():
    print("\n" + "="*70)
    print("🚀 MICROGRID V7 - QUICK DEMO WITH VISUALIZATION")
    print("="*70 + "\n")
    print("Running 100-episode demo (~1 minute)...")
    print()
    
    config = MicrogridConfig()
    config.EPISODES = 100
    
    simulator = AggressiveMicrogridSimulator(config)
    
    start = datetime.now()
    simulator.run(verbose=False)
    duration = (datetime.now() - start).total_seconds()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f'demo_{timestamp}'
    simulator.export_results(prefix=prefix)
    
    print("\n📊 Generating visualizations...")
    visualizer = TrainingVisualizer()
    
    training_df = pd.read_csv(f'{prefix}_training_progress.csv')
    validation_df = pd.read_csv(f'{prefix}_validation_results.csv')
    hourly_df = pd.read_csv(f'{prefix}_resilience_hourly_summary.csv')
    
    visualizer.plot_training_progress(training_df, validation_df, prefix)
    visualizer.plot_hourly_performance(hourly_df, prefix)
    
    if simulator.validation_results:
        best = max(simulator.validation_results, key=lambda x: x['load_met_rate'])
        
        print("\n" + "="*70)
        print("DEMO RESULTS")
        print("="*70)
        print(f"Training Time:        {duration:.1f}s")
        print(f"Best Load Met Rate:   {best['load_met_rate']:.2f}%")
        print(f"Unmet Energy:         {best['unmet_energy']:.3f} kWh/day")
        print(f"Training Episodes:    100")
        print()
        print("✅ Demo complete! Plots saved in plots/ directory")
        print("   For full performance, run:")
        print("   python cli.py train --full --plot --economics")
        print("="*70 + "\n")


def run_training(args):
    print("\n" + "="*70)
    print("🚀 MICROGRID V7 TRAINING")
    print("="*70 + "\n")
    
    config = MicrogridConfig()
    config.EPISODES = 2000 if args.full else args.episodes
    config.BATTERY_CAPACITY = args.battery
    
    print(f"Configuration:")
    print(f"  Episodes:     {config.EPISODES}")
    print(f"  Battery:      {config.BATTERY_CAPACITY} kWh")
    print(f"  GPU:          {'✅ Available' if torch.cuda.is_available() else '❌ CPU only'}")
    print(f"  Est. Time:    ~{config.EPISODES * 0.6 / 60:.0f} minutes")
    print(f"  Plots:        {'✅ Enabled' if args.plot else '❌ Disabled'}")
    print(f"  Monitor:      {'✅ Enabled' if args.monitor else '❌ Disabled'}")
    print()
    
    simulator = AggressiveMicrogridSimulator(config)
    
    if args.monitor:
        monitor = RealTimeMonitor(refresh_interval=3)
        monitor.start(simulator)
    
    start = datetime.now()
    try:
        simulator.run(verbose=not args.monitor)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        if args.monitor:
            monitor.stop()
    finally:
        if args.monitor:
            monitor.stop()
    
    duration = (datetime.now() - start).total_seconds()
    
    print(f"\n✅ Training complete!")
    print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f'microgrid_v7_{timestamp}'
    simulator.export_results(prefix=prefix)
    
    model_path = f'models/{prefix}_model.pt'
    os.makedirs('models', exist_ok=True)
    
    best_val = max(simulator.validation_results, 
                   key=lambda x: x['load_met_rate']) if simulator.validation_results else None
    
    torch.save({
        'policy_net': simulator.agent.policy_net.state_dict(),
        'target_net': simulator.agent.target_net.state_dict(),
        'config': config.to_dict(),
        'performance': {
            'load_met_rate': best_val['load_met_rate'] if best_val else 0,
            'unmet_energy': best_val['unmet_energy'] if best_val else 0,
            'episode': best_val['episode'] if best_val else 0
        },
        'timestamp': timestamp
    }, model_path)
    print(f"✅ Model saved: {model_path}")
    
    if args.plot:
        print("\n📊 Generating visualizations...")
        visualizer = TrainingVisualizer()
        
        training_df = pd.read_csv(f'{prefix}_training_progress.csv')
        validation_df = pd.read_csv(f'{prefix}_validation_results.csv')
        hourly_df = pd.read_csv(f'{prefix}_resilience_hourly_summary.csv')
        
        visualizer.plot_training_progress(training_df, validation_df, prefix)
        visualizer.plot_hourly_performance(hourly_df, prefix)
    
    if args.economics:
        print("\n" + "="*70)
        print("💰 ECONOMIC ANALYSIS")
        print("="*70)
        
        last_ep = [h for h in simulator.history 
                  if h['episode'] == max(h['episode'] for h in simulator.history)]
        
        if last_ep:
            analyzer = EconomicAnalyzer(config, last_ep)
            economics_data = analyzer.calculate_roi(last_ep)
            analyzer.print_economic_report(last_ep)
            
            if args.plot:
                visualizer.plot_economic_analysis(economics_data, prefix)
        else:
            print("⚠️  No episode data available for economic analysis")
    
    print("\n🎉 Training pipeline complete!")
    print(f"\n📁 Generated files:")
    print(f"   - {prefix}_training_progress.csv")
    print(f"   - {prefix}_validation_results.csv")
    print(f"   - {prefix}_resilience_detailed.csv")
    print(f"   - models/{prefix}_model.pt")
    if args.plot:
        print(f"   - plots/{prefix}_progress.png")
        print(f"   - plots/{prefix}_performance.png")
        if args.economics:
            print(f"   - plots/{prefix}_economics_analysis.png")
    print()


def run_evaluation(args):
    print("\n" + "="*70)
    print("🔬 MODEL EVALUATION")
    print("="*70 + "\n")
    
    if args.auto_latest:
        manager = ModelManager()
        model_path = manager.get_latest_model()
        if not model_path:
            print("❌ Error: No models found. Train a model first.")
            return
        print(f"🔍 Auto-detected latest model: {model_path}\n")
    else:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"❌ Error: Model file not found: {model_path}")
            return
    
    # Load model
    weights_only=False
    checkpoint = torch.load(model_path, map_location='cpu')
    config_dict = checkpoint['config']
    
    # Reconstruct config
    config = MicrogridConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create agent
    agent = DQNAgent(config)
    agent.policy_net.load_state_dict(checkpoint['policy_net'])
    agent.policy_net.eval()
    
    perf = checkpoint.get('performance', {})
    
    print(f"✅ Loaded model: {os.path.basename(model_path)}")
    print(f"   Battery: {config.BATTERY_CAPACITY} kWh")
    print(f"   Training performance: {perf.get('load_met_rate', 0):.2f}%")
    print(f"   Trained at episode: {perf.get('episode', 'unknown')}")
    print()
    
    # Run test scenarios
    battery = BatterySystem(config)
    generator = StochasticTimeSeriesGenerator(config)
    
    results = []
    
    print(f"Running {args.scenarios} test scenarios...")
    print()
    
    for scenario in tqdm(range(args.scenarios), desc="Evaluating", unit="scenario"):
        battery.reset()
        profile = generator.generate_episode_profile(scenario)
        
        load_met_count = 0
        unmet_energy = 0
        
        for h in range(24):
            soc = battery.get_SOC_energy()
            renewable = profile['pv_power'][h] + profile['wind_power'][h]
            load = profile['load_demand'][h]
            
            state = agent.get_enhanced_state(soc, h, renewable, load)
            
            with torch.no_grad():
                action_idx = agent.select_action(state, explore=False)
            action = agent.actions[action_idx]
            
            # Calculate initial energy balance
            net_renewable = renewable - load  # Positive = excess, Negative = shortfall
            
            if net_renewable >= 0:
                excess = net_renewable
                
                if action == 'charge_full':
                    charged = battery.charge(excess)
                    energy_used_from_battery = 0
                    
                elif action == 'charge_half':
                    charged = battery.charge(excess * 0.5)
                    energy_used_from_battery = 0
                    
                else:  
                    energy_used_from_battery = 0
                
                load_met = True
                hour_unmet = 0
                
            else:
                shortfall = abs(net_renewable)
                
                if action == 'discharge_full':
                    energy_used_from_battery = battery.discharge(shortfall)
                    
                elif action == 'discharge_half':
                    energy_used_from_battery = battery.discharge(shortfall * 0.5)
                    
                else:  
                    energy_used_from_battery = 0
                
                total_available = renewable + energy_used_from_battery
                
                if total_available >= load - 0.01:  
                    load_met = True
                    hour_unmet = 0
                else:
                    load_met = False
                    hour_unmet = load - total_available
            
            # ======================================================================
            # Track results
            # ======================================================================
            
            if load_met:
                load_met_count += 1
            unmet_energy += hour_unmet
            #shortfall = max(0, load - renewable - (soc - soc_after))
            
        load_met_rate = (load_met_count / 24) * 100
        results.append({
            'scenario': scenario + 1,
            'load_met_rate': load_met_rate,
            'unmet_energy': unmet_energy,
            'cloud_cover': profile['metadata']['cloud_cover'],
            'load_type': profile['metadata']['load_type']
        })
    
    print()
    
    # Detailed results
    print("SCENARIO RESULTS:")
    print("-" * 70)
    for r in results:
        status = "✅" if r['load_met_rate'] >= 95 else "⚠️"
        print(f"  {status} Scenario {r['scenario']:2d}: {r['load_met_rate']:5.1f}% load met, "
              f"{r['unmet_energy']:5.2f} kWh unmet ({r['load_type']}, "
              f"cloud={r['cloud_cover']:.1f})")
    
    # Summary statistics
    avg_load_met = np.mean([r['load_met_rate'] for r in results])
    std_load_met = np.std([r['load_met_rate'] for r in results])
    min_load_met = min([r['load_met_rate'] for r in results])
    max_load_met = max([r['load_met_rate'] for r in results])
    avg_unmet = np.mean([r['unmet_energy'] for r in results])
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Average Load Met:  {avg_load_met:.2f}% ± {std_load_met:.2f}%")
    print(f"Min Load Met:      {min_load_met:.2f}%")
    print(f"Max Load Met:      {max_load_met:.2f}%")
    print(f"Average Unmet:     {avg_unmet:.3f} kWh/day")
    print(f"Scenarios ≥95%:    {sum(1 for r in results if r['load_met_rate'] >= 95)}/{len(results)}")
    
    # Performance rating
    if avg_load_met >= 99:
        print(f"\n✅ EXCELLENT - Production ready")
    elif avg_load_met >= 95:
        print(f"\n✅ GOOD - Acceptable for deployment")
    elif avg_load_met >= 90:
        print(f"\n⚠️  FAIR - Consider more training")
    else:
        print(f"\n❌ POOR - Retrain with more episodes")
    
    print("="*70 + "\n")
    
    # Export results
    df = pd.DataFrame(results)
    output_file = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Results saved: {output_file}")
    
    if args.plot:
        print("📊 Generating evaluation plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Load Met Rate Distribution
        ax = axes[0]
        ax.hist([r['load_met_rate'] for r in results], bins=20, 
                color='#2E86AB', alpha=0.7, edgecolor='black')
        ax.axvline(x=avg_load_met, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {avg_load_met:.1f}%')
        ax.axvline(x=95, color='green', linestyle='--', linewidth=2, 
                   label='95% Target')
        ax.set_xlabel('Load Met Rate (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Load Coverage Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Performance by Load Type
        ax = axes[1]
        load_types = list(set([r['load_type'] for r in results]))
        type_performance = {lt: [] for lt in load_types}
        for r in results:
            type_performance[r['load_type']].append(r['load_met_rate'])
        
        positions = range(len(load_types))
        box_data = [type_performance[lt] for lt in load_types]
        bp = ax.boxplot(box_data, positions=positions, tick_labels=load_types,
                        patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('#10b981')
            patch.set_alpha(0.7)
        
        ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% Target')
        ax.set_ylabel('Load Met Rate (%)')
        ax.set_title('Performance by Load Type', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file = f"plots/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        os.makedirs('plots', exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {plot_file}")
        plt.close()
    
    print()


def run_comparison(args):
    print("\n" + "="*70)
    print("🔋 BATTERY CAPACITY COMPARISON")
    print("="*70 + "\n")
    
    print(f"Testing capacities: {args.capacities} kWh")
    print(f"Episodes per test: {args.episodes}")
    print(f"Parallel processing: {'✅ Enabled' if args.parallel else '❌ Disabled'}")
    print()
    
    results = []
    
    if args.parallel:
        num_workers = min(cpu_count(), len(args.capacities))
        print(f"Using {num_workers} parallel workers\n")
        
        worker_args = [(cap, args.episodes, i+1) for i, cap in enumerate(args.capacities)]
        
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(run_single_comparison, worker_args),
                total=len(worker_args),
                desc="Comparing",
                unit="config"
            ))
        
        results = [r for r in results if r is not None]
    
    else:
        for i, capacity in enumerate(args.capacities):
            print(f"[{i+1}/{len(args.capacities)}] Testing {capacity} kWh battery...")
            
            result = run_single_comparison((capacity, args.episodes, i+1))
            if result:
                results.append(result)
            
            print(f"     Result: {result['load_met_rate']:.1f}% load met, "
                  f"${result['total_system_cost']:,.0f} total cost\n")
    
    if not results:
        print("❌ No valid results obtained")
        return
    
    # Find optimal
    best_perf = max(results, key=lambda x: x['load_met_rate'])
    eligible_for_value = [r for r in results if r['load_met_rate'] >= 95]
    best_value = min(eligible_for_value, key=lambda x: x['total_system_cost']) if eligible_for_value else best_perf
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70 + "\n")
    
    # Table
    print(f"{'Capacity':>10s} {'Load Met':>10s} {'Unmet':>10s} {'Total Cost':>12s} {'Value':>10s}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['capacity_kwh']):
        marker = "[*]" if r == best_value else ("[!]" if r == best_perf else "   ")
        print(f"{marker} {r['capacity_kwh']:>7.0f} kWh {r['load_met_rate']:>9.1f}% "
              f"{r['unmet_energy']:>9.2f} kWh ${r['total_system_cost']:>10,.0f} "
              f"${r['cost_per_percent']:>9,.0f}/pct")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print(f"🏆 Best Performance:  {best_perf['capacity_kwh']} kWh @ {best_perf['load_met_rate']:.1f}% load met")
    print(f"⭐ Best Value:        {best_value['capacity_kwh']} kWh @ ${best_value['total_system_cost']:,.0f}")
    
    if best_value['load_met_rate'] >= 98:
        print(f"\n✅ Recommended: {best_value['capacity_kwh']} kWh")
        print(f"   Excellent performance at optimal cost")
    elif best_perf['load_met_rate'] - best_value['load_met_rate'] > 5:
        print(f"\n⚡ Consider: {best_perf['capacity_kwh']} kWh")
        print(f"   Worth extra ${best_perf['total_system_cost'] - best_value['total_system_cost']:,.0f} "
              f"for +{best_perf['load_met_rate'] - best_value['load_met_rate']:.1f}% performance")
    else:
        print(f"\n✅ Recommended: {best_value['capacity_kwh']} kWh")
        print(f"   Best value for performance")
    
    print("="*70 + "\n")
    
    # Export
    df = pd.DataFrame(results)
    output_file = f"capacity_comparison_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Comparison saved: {output_file}")
    
    if args.plot:
        print("📊 Generating comparison plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Battery Capacity Comparison', fontsize=16, fontweight='bold')
        
        capacities = [r['capacity_kwh'] for r in sorted(results, key=lambda x: x['capacity_kwh'])]
        load_mets = [r['load_met_rate'] for r in sorted(results, key=lambda x: x['capacity_kwh'])]
        costs = [r['total_system_cost'] for r in sorted(results, key=lambda x: x['capacity_kwh'])]
        
        # 1. Performance vs Capacity
        ax = axes[0]
        ax.plot(capacities, load_mets, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% Target')
        ax.axhline(y=100, color='gold', linestyle='--', alpha=0.5, label='100% Perfect')
        
        # Mark best performance
        ax.scatter([best_perf['capacity_kwh']], [best_perf['load_met_rate']], 
                  color='red', s=300, marker='*', zorder=5, label='Best Performance', 
                  edgecolors='black', linewidth=2)
        
        ax.set_xlabel('Battery Capacity (kWh)', fontweight='bold')
        ax.set_ylabel('Load Met Rate (%)', fontweight='bold')
        ax.set_title('Performance vs Capacity', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([min(load_mets) - 5, 105])
        
        # 2. Cost vs Capacity
        ax = axes[1]
        ax.plot(capacities, costs, marker='s', linewidth=2, markersize=8, color='#10b981')
        
        # Mark best value
        ax.scatter([best_value['capacity_kwh']], [best_value['total_system_cost']], 
                  color='gold', s=300, marker='*', zorder=5, label='Best Value', 
                  edgecolors='black', linewidth=2)
        
        ax.set_xlabel('Battery Capacity (kWh)', fontweight='bold')
        ax.set_ylabel('Total System Cost ($)', fontweight='bold')
        ax.set_title('Cost vs Capacity', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        plt.tight_layout()
        plot_file = f"plots/comparison_{datetime.now().strftime('%Y%m%d')}.png"
        os.makedirs('plots', exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {plot_file}")
        plt.close()
    
    print()


def show_status():
    print("\n" + "="*70)
    print("📊 SYSTEM STATUS")
    print("="*70 + "\n")
    
    print("Dependencies:")
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        print(f"     GPU: {'✅ Available' if torch.cuda.is_available() else '❌ Not available'}")
        if torch.cuda.is_available():
            print(f"     Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  ❌ PyTorch not installed")
    
    try:
        import pandas
        print(f"  ✅ Pandas {pandas.__version__}")
    except ImportError:
        print("  ❌ Pandas not installed")
    
    try:
        import numpy
        print(f"  ✅ NumPy {numpy.__version__}")
    except ImportError:
        print("  ❌ NumPy not installed")
    
    try:
        import matplotlib
        print(f"  ✅ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("  ❌ Matplotlib not installed")
    
    print()
    
    print("Saved Models:")
    manager = ModelManager()
    models = manager.list_models()
    
    if models:
        for i, model in enumerate(models[:5], 1):
            print(f"  📦 {os.path.basename(model['path'])}")
            print(f"     Performance: {model['load_met_rate']:.1f}% load met")
            print(f"     Timestamp: {model['timestamp']}")
            print(f"     Size: {model['size_mb']:.1f} MB")
        
        if len(models) > 5:
            print(f"\n  ... and {len(models) - 5} more models")
    else:
        print("  No saved models found")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='🔋 Microgrid v7 - Enhanced Production CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo with plots (100 episodes, ~1 min)
  python cli.py demo

  # Full training with plots and economics (2000 episodes, ~20 min)
  python cli.py train --full --plot --economics
  
  # Training with real-time monitoring
  python cli.py train --episodes 1000 --plot --monitor
  
  # Auto-load latest model for evaluation
  python cli.py evaluate --auto-latest --scenarios 20 --plot
  
  # Parallel battery comparison with visualization
  python cli.py compare --capacities 13 15 18 20 --parallel --plot
  
  # Check system status and models
  python cli.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    demo_parser = subparsers.add_parser('demo', help='Quick demo with visualization (100 episodes)')
    
    train_parser = subparsers.add_parser('train', help='Train new model')
    train_parser.add_argument('--episodes', type=int, default=500, 
                             help='Training episodes (default: 500)')
    train_parser.add_argument('--full', action='store_true',
                             help='Run full 2000 episodes')
    train_parser.add_argument('--economics', action='store_true',
                             help='Include economic analysis')
    train_parser.add_argument('--battery', type=float, default=18.0,
                             help='Battery capacity in kWh (default: 18)')
    train_parser.add_argument('--plot', action='store_true',
                             help='Generate training visualizations')
    train_parser.add_argument('--monitor', action='store_true',
                             help='Enable real-time training monitor')
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model', help='Path to model checkpoint')
    eval_parser.add_argument('--auto-latest', action='store_true',
                           help='Automatically load latest model')
    eval_parser.add_argument('--scenarios', type=int, default=10,
                           help='Number of test scenarios (default: 10)')
    eval_parser.add_argument('--plot', action='store_true',
                           help='Generate evaluation plots')
    
    compare_parser = subparsers.add_parser('compare', help='Compare battery sizes')
    compare_parser.add_argument('--capacities', nargs='+', type=float,
                               default=[13, 15, 18, 20],
                               help='Battery capacities to test (kWh)')
    compare_parser.add_argument('--episodes', type=int, default=200,
                               help='Episodes per test (default: 200)')
    compare_parser.add_argument('--parallel', action='store_true',
                               help='Use parallel processing')
    compare_parser.add_argument('--plot', action='store_true',
                               help='Generate comparison plots')
    
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        run_demo()
    elif args.command == 'train':
        run_training(args)
    elif args.command == 'evaluate':
        if not args.auto_latest and not args.model:
            print("❌ Error: Either --model or --auto-latest must be specified")
            eval_parser.print_help()
        else:
            run_evaluation(args)
    elif args.command == 'compare':
        run_comparison(args)
    elif args.command == 'status':
        show_status()
    else:
        parser.print_help()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
                