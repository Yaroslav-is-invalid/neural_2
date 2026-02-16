#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Прогноз урожайности зерновых по спутниковым данным и метеоинформации
========================================================================
Задача: Регрессия для предсказания урожайности (ц/га) по временным рядам
Модели: LSTM, Transformer, CNN-LSTM
Данные: NASA POWER (метеоданные) + Eurostat crop yield statistics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Машинное обучение
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Визуализация
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Для работы с датами
import calendar

# Установка seed для воспроизводимости
np.random.seed(42)
torch.manual_seed(42)

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# Константы
SEQ_LENGTH = 12  # Длина последовательности (12 месяцев)
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

# ============================================================================
# ЧАСТЬ 1: ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ
# ============================================================================
# В реальном проекте здесь будет загрузка реальных данных из API NASA POWER
# и Eurostat. Для демонстрации создаем синтетические данные.

class DataGenerator:
    """Генератор синтетических данных для демонстрации"""
    
    def __init__(self, num_years=20, num_regions=5):
        self.num_years = num_years
        self.num_regions = num_regions
        self.regions = [f'Region_{i}' for i in range(num_regions)]
        self.years = list(range(2000, 2000 + num_years))
        
    def generate_meteo_data(self):
        """Генерация метеоданных (температура, осадки, влажность)"""
        data = []
        
        for region in self.regions:
            for year in self.years:
                for month in range(1, 13):
                    # Базовые сезонные паттерны
                    base_temp = 10 + 15 * np.sin(2 * np.pi * (month - 1) / 12)
                    base_precip = 50 + 30 * np.sin(2 * np.pi * (month + 6) / 12)
                    
                    # Случайные вариации
                    temp = base_temp + np.random.normal(0, 3) + np.random.normal(0, 1) * (year - 2000) * 0.05
                    precip = base_precip + np.random.normal(0, 20) + np.random.normal(0, 5) * np.sin(year)
                    humidity = 60 + 20 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 10)
                    
                    # Солнечная радиация
                    solar = 200 + 100 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 30)
                    
                    data.append({
                        'region': region,
                        'year': year,
                        'month': month,
                        'temperature': temp,
                        'precipitation': precip,
                        'humidity': humidity,
                        'solar_radiation': solar
                    })
        
        return pd.DataFrame(data)
    
    def generate_vegetation_indices(self):
        """Генерация вегетационных индексов (NDVI, EVI)"""
        data = []
        
        for region in self.regions:
            for year in self.years:
                for month in range(1, 13):
                    # NDVI зависит от сезона и качества почвы
                    base_ndvi = 0.3 + 0.4 * np.sin(2 * np.pi * (month - 4) / 12)
                    base_ndvi = np.clip(base_ndvi, 0.1, 0.8)
                    
                    # Тренд улучшения агротехнологий
                    tech_trend = 0.002 * (year - 2000)
                    
                    # Случайные факторы (засухи, болезни)
                    random_factor = np.random.normal(0, 0.05)
                    
                    ndvi = base_ndvi + tech_trend + random_factor
                    
                    # EVI коррелирует с NDVI, но более чувствителен
                    evi = 1.5 * ndvi + np.random.normal(0, 0.05)
                    
                    data.append({
                        'region': region,
                        'year': year,
                        'month': month,
                        'ndvi': ndvi,
                        'evi': evi
                    })
        
        return pd.DataFrame(data)
    
    def generate_yield_data(self, meteo_df, vi_df):
        """Генерация данных урожайности на основе метео и вегетационных индексов"""
        yield_data = []
        
        for region in self.regions:
            # Базовый потенциал урожайности для региона
            base_yield = np.random.uniform(40, 70)
            soil_quality = np.random.uniform(0.8, 1.2)
            
            for year in self.years:
                # Фильтруем данные для региона и года
                meteo_region = meteo_df[(meteo_df['region'] == region) & 
                                        (meteo_df['year'] == year)]
                vi_region = vi_df[(vi_df['region'] == region) & 
                                  (vi_df['year'] == year)]
                
                if len(meteo_region) == 0 or len(vi_region) == 0:
                    continue
                
                # Агрометеорологические факторы
                temp_summer = meteo_region[meteo_region['month'].isin([6,7,8])]['temperature'].mean()
                precip_summer = meteo_region[meteo_region['month'].isin([5,6,7,8])]['precipitation'].sum()
                
                # Оптимальные условия
                temp_factor = np.exp(-0.5 * ((temp_summer - 22) / 5) ** 2)
                precip_factor = np.exp(-0.5 * ((precip_summer - 300) / 100) ** 2)
                
                # Вегетационные индексы в пике сезона
                max_ndvi = vi_region[vi_region['month'].isin([6,7,8])]['ndvi'].mean()
                max_evi = vi_region[vi_region['month'].isin([6,7,8])]['evi'].mean()
                
                # Технологический тренд
                tech_trend = 0.3 * (year - 2000)
                
                # Финальная урожайность
                yield_value = (base_yield * soil_quality * temp_factor * precip_factor * 
                              (0.5 + max_ndvi) + tech_trend + np.random.normal(0, 3))
                
                yield_data.append({
                    'region': region,
                    'year': year,
                    'yield': max(20, yield_value)  # Минимальная урожайность 20 ц/га
                })
        
        return pd.DataFrame(yield_data)
    
    def generate_all_data(self):
        """Генерация всех данных"""
        print("Генерация метеоданных...")
        meteo_df = self.generate_meteo_data()
        
        print("Генерация вегетационных индексов...")
        vi_df = self.generate_vegetation_indices()
        
        print("Генерация данных урожайности...")
        yield_df = self.generate_yield_data(meteo_df, vi_df)
        
        # Объединение всех данных
        merged_df = pd.merge(meteo_df, vi_df, on=['region', 'year', 'month'])
        
        return merged_df, yield_df

# ============================================================================
# ЧАСТЬ 2: ПРЕДОБРАБОТКА ДАННЫХ И СОЗДАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ
# ============================================================================

class TimeSeriesPreprocessor:
    """Класс для предобработки временных рядов"""
    
    def __init__(self, seq_length=12):
        self.seq_length = seq_length
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def create_sequences(self, data, target_data):
        """Создание последовательностей для обучения"""
        X, y = [], []
        
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(target_data[i + self.seq_length])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, merged_df, yield_df):
        """Подготовка данных для модели"""
        features = ['temperature', 'precipitation', 'humidity', 
                   'solar_radiation', 'ndvi', 'evi']
        
        # Мерджим данные
        full_df = pd.merge(merged_df, yield_df, on=['region', 'year'])
        
        # Сортируем по региону и времени
        full_df = full_df.sort_values(['region', 'year', 'month'])
        
        X_by_region = []
        y_by_region = []
        
        for region in full_df['region'].unique():
            region_data = full_df[full_df['region'] == region]
            
            # Нормализация внутри региона
            X_region = region_data[features].values
            y_region = region_data['yield'].values
            
            # Создаем последовательности
            X_seq, y_seq = self.create_sequences(X_region, y_region)
            
            X_by_region.append(X_seq)
            y_by_region.append(y_seq)
        
        # Объединяем все регионы
        X_all = np.concatenate(X_by_region, axis=0)
        y_all = np.concatenate(y_by_region, axis=0)
        
        return X_all, y_all

# ============================================================================
# ЧАСТЬ 3: АРХИТЕКТУРЫ МОДЕЛЕЙ
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM модель для прогнозирования временных рядов"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Инициализация скрытого состояния
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Прямой проход через LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Берем только последний выход
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out.squeeze()

class TransformerModel(nn.Module):
    """Transformer модель для прогнозирования временных рядов"""
    
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=3, output_size=1, dropout=0.2):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Позиционное кодирование
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # Проекция входных данных
        x = self.input_projection(x)
        
        # Добавление позиционного кодирования
        x = self.pos_encoder(x)
        
        # Трансформер
        x = self.transformer_encoder(x)
        
        # Берем последний элемент последовательности
        x = x[:, -1, :]
        
        # Выходной слой
        output = self.fc(x)
        
        return output.squeeze()

class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class CNNLSTMModel(nn.Module):
    """Гибридная CNN-LSTM модель"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(CNNLSTMModel, self).__init__()
        
        # CNN слой для извлечения признаков
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
        # LSTM слой
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # CNN часть
        x = x.permute(0, 2, 1)  # [batch, seq_len, features] -> [batch, features, seq_len]
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # LSTM часть
        x = x.permute(0, 2, 1)  # [batch, features, seq_len] -> [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)
        
        # Выход
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        
        return out.squeeze()

# ============================================================================
# ЧАСТЬ 4: ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ
# ============================================================================

class ModelTrainer:
    """Класс для обучения и оценки моделей"""
    
    def __init__(self, model, model_name, device):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Для хранения истории
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader):
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            
            # Градиентное клиппирование
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())
        
        val_loss = total_loss / len(val_loader)
        return val_loss, np.array(predictions), np.array(actuals)
    
    def train(self, train_loader, val_loader, num_epochs):
        """Полный цикл обучения"""
        print(f"\n{'='*50}")
        print(f"Обучение модели: {self.model_name}")
        print(f"{'='*50}")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, preds, actuals = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Обновление learning rate
            self.scheduler.step(val_loss)
            
            # Сохранение лучшей модели
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), f'best_{self.model_name}.pth')
            
            if (epoch + 1) % 10 == 0:
                print(f"Эпоха [{epoch+1}/{num_epochs}], "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Загрузка лучшей модели
        self.model.load_state_dict(torch.load(f'best_{self.model_name}.pth'))
        
        return self.train_losses, self.val_losses
    
    def evaluate(self, test_loader):
        """Оценка модели на тестовых данных"""
        _, predictions, actuals = self.validate(test_loader)
        
        # Метрики
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        
        return {
            'predictions': predictions,
            'actuals': actuals,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

# ============================================================================
# ЧАСТЬ 5: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================================

class Visualizer:
    """Класс для визуализации результатов"""
    
    @staticmethod
    def plot_training_history(histories, model_names):
        """Визуализация истории обучения"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for i, (history, name) in enumerate(zip(histories, model_names)):
            train_losses, val_losses = history
            axes[0].plot(train_losses, label=f'{name} (Train)', linestyle='-')
            axes[1].plot(val_losses, label=f'{name} (Val)', linestyle='--')
        
        axes[0].set_xlabel('Эпоха')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Потери на обучении')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].set_xlabel('Эпоха')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Потери на валидации')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_predictions(results, model_names):
        """Визуализация предсказаний vs фактические значения"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (result, name) in enumerate(zip(results, model_names)):
            ax = axes[idx]
            predictions = result['predictions']
            actuals = result['actuals']
            
            # Scatter plot
            ax.scatter(actuals, predictions, alpha=0.6, edgecolors='w', linewidth=0.5)
            
            # Линия идеального предсказания
            min_val = min(actuals.min(), predictions.min())
            max_val = max(actuals.max(), predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеально')
            
            ax.set_xlabel('Фактическая урожайность (ц/га)')
            ax.set_ylabel('Предсказанная урожайность (ц/га)')
            ax.set_title(f'{name}\nMAE: {result["mae"]:.2f}, RMSE: {result["rmse"]:.2f}, R²: {result["r2"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('predictions_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_regional_forecast(results_by_region, regions, years):
        """Визуализация прогноза по регионам"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=regions[:6],
            shared_yaxes=True
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, region in enumerate(regions[:6]):
            row = i // 3 + 1
            col = i % 3 + 1
            
            if region in results_by_region:
                result = results_by_region[region]
                
                # Создаем временную шкалу
                time_points = years[-len(result['predictions']):]
                
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=result['actuals'],
                        mode='lines+markers',
                        name=f'{region} (факт)',
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=6)
                    ),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=result['predictions'],
                        mode='lines+markers',
                        name=f'{region} (прогноз)',
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                        marker=dict(size=6, symbol='square')
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title_text="Прогноз урожайности по регионам",
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Год")
        fig.update_yaxes(title_text="Урожайность (ц/га)")
        
        fig.write_html("regional_forecast.html")
        fig.show()
    
    @staticmethod
    def plot_metrics_comparison(results, model_names):
        """Сравнение метрик моделей"""
        metrics = ['MAE', 'RMSE', 'R2']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = []
            for result in results:
                if metric == 'MAE':
                    values.append(result['mae'])
                elif metric == 'RMSE':
                    values.append(result['rmse'])
                else:
                    values.append(result['r2'])
            
            bars = axes[i].bar(model_names, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[i].set_title(f'Сравнение {metric}')
            axes[i].set_ylabel(metric)
            
            # Добавление значений на столбцы
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_feature_importance(model, feature_names):
        """Визуализация важности признаков (для интерпретируемости)"""
        if isinstance(model, LSTMModel):
            # Для LSTM используем веса первого слоя как прокси важности
            weights = model.lstm.weight_ih_l0.detach().cpu().numpy()
            importance = np.mean(np.abs(weights), axis=0)
        else:
            importance = np.random.rand(len(feature_names))
        
        # Нормализация
        importance = importance / importance.sum()
        
        # Сортировка
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importance[indices], color='skyblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Важность')
        plt.title('Важность признаков для прогноза урожайности')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()

# ============================================================================
# ЧАСТЬ 6: ОСНОВНОЙ ПАЙПЛАЙН
# ============================================================================

def main():
    """Основная функция"""
    print("="*70)
    print("ПРОГНОЗ УРОЖАЙНОСТИ ЗЕРНОВЫХ ПО СПУТНИКОВЫМ И МЕТЕОДАННЫМ")
    print("="*70)
    
    # Шаг 1: Генерация данных
    print("\n[1/6] Генерация данных...")
    generator = DataGenerator(num_years=20, num_regions=5)
    merged_df, yield_df = generator.generate_all_data()
    
    print(f"Метеоданные: {len(merged_df)} записей")
    print(f"Данные урожайности: {len(yield_df)} записей")
    print(f"Регионы: {generator.regions}")
    print(f"Годы: {generator.years[0]}-{generator.years[-1]}")
    
    # Шаг 2: Предобработка
    print("\n[2/6] Предобработка данных...")
    preprocessor = TimeSeriesPreprocessor(seq_length=SEQ_LENGTH)
    X, y = preprocessor.prepare_data(merged_df, yield_df)
    
    print(f"Размерность X: {X.shape}")
    print(f"Размерность y: {y.shape}")
    
    # Нормализация
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_normalized = preprocessor.scaler_X.fit_transform(X_reshaped)
    X_normalized = X_normalized.reshape(X.shape)
    
    y_normalized = preprocessor.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Разделение на train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_normalized, y_normalized, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Создание DataLoader'ов
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Шаг 3: Создание моделей
    print("\n[3/6] Создание моделей...")
    input_size = X.shape[-1]
    
    models = {
        'LSTM': LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, 1, DROPOUT),
        'Transformer': TransformerModel(input_size, d_model=64, nhead=4, num_layers=3, output_size=1, dropout=DROPOUT),
        'CNN-LSTM': CNNLSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, 1, DROPOUT)
    }
    
    # Шаг 4: Обучение моделей
    print("\n[4/6] Обучение моделей...")
    trainers = []
    histories = []
    
    for name, model in models.items():
        trainer = ModelTrainer(model, name, device)
        history = trainer.train(train_loader, val_loader, NUM_EPOCHS)
        trainers.append(trainer)
        histories.append(history)
    
    # Шаг 5: Оценка моделей
    print("\n[5/6] Оценка моделей...")
    results = []
    
    for trainer in trainers:
        result = trainer.evaluate(test_loader)
        results.append(result)
        
        print(f"\n{'-'*30}")
        print(f"Модель: {trainer.model_name}")
        print(f"MAE: {result['mae']:.3f}")
        print(f"RMSE: {result['rmse']:.3f}")
        print(f"R²: {result['r2']:.3f}")
    
    # Обратное масштабирование для интерпретации
    for result in results:
        result['predictions'] = preprocessor.scaler_y.inverse_transform(
            result['predictions'].reshape(-1, 1)
        ).flatten()
        result['actuals'] = preprocessor.scaler_y.inverse_transform(
            result['actuals'].reshape(-1, 1)
        ).flatten()
        result['mae'] = mean_absolute_error(result['actuals'], result['predictions'])
        result['rmse'] = np.sqrt(mean_squared_error(result['actuals'], result['predictions']))
    
    # Шаг 6: Визуализация
    print("\n[6/6] Визуализация результатов...")
    visualizer = Visualizer()
    
    # История обучения
    visualizer.plot_training_history(histories, list(models.keys()))
    
    # Сравнение предсказаний
    visualizer.plot_predictions(results, list(models.keys()))
    
    # Сравнение метрик
    visualizer.plot_metrics_comparison(results, list(models.keys()))
    
    # Прогноз по регионам (пример)
    results_by_region = {region: results[0] for region in generator.regions}
    visualizer.plot_regional_forecast(results_by_region, generator.regions, generator.years)
    
    # Важность признаков
    feature_names = ['Temperature', 'Precipitation', 'Humidity', 
                    'Solar Radiation', 'NDVI', 'EVI']
    visualizer.plot_feature_importance(models['LSTM'], feature_names)
    
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*70)
    
    # Сохранение результатов
    results_df = pd.DataFrame({
        'Model': list(models.keys()),
        'MAE': [r['mae'] for r in results],
        'RMSE': [r['rmse'] for r in results],
        'R2': [r['r2'] for r in results]
    })
    
    results_df.to_csv('model_results.csv', index=False)
    print("\nРезультаты сохранены в 'model_results.csv'")
    
    # Вывод лучшей модели
    best_idx = np.argmin([r['rmse'] for r in results])
    print(f"\nЛучшая модель: {list(models.keys())[best_idx]} "
          f"(RMSE: {results[best_idx]['rmse']:.2f} ц/га)")
    
    return models, trainers, results

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    main()