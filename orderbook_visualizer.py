"""
# Описание Файла: Визуализатор Глубины Рынка (Core Visualization Engine)

Назначение
ОСНОВНОЙ МОДУЛЬ ВИЗУАЛИЗАЦИИ.
Этот файл — ядро нашего количественного анализа. Он отвечает за чтение сырых данных книги ордеров и сделок из файлов `*.npz` и преобразование их в огромные, детализированные изображения (тепловые карты ликвидности).

Квантовая Логика и Визуализация
1.  Загрузка данных: Читает матрицы цен (A/B), объемов (vA/vB) и ленту сделок (T) из `.npz`.
2.  Биннинг и Разрешение: Использует технику *биннинга* (усреднение по временным интервалам) для преобразования необработанных временных меток в фиксированную ось X шириной 12000 пикселей (Ultra-HD).
3.  Тепловая Карта (Ликвидность): Строит две тепловые карты (верхняя — абсолютная цена, нижняя — относительное смещение цены), где цвет и яркость показывают уровень ликвидности (Volume) на каждом ценовом уровне в конкретный момент времени. Для выделения как мелких, так и крупных объемов применяется логарифмическая шкала (`np.log1p`).
4.  Сделки (Trades): Накладывает линию сделок поверх тепловой карты, используя объем (`tr_sz`) для определения размера точки и направления (агрессор-покупатель/продавец).

В итоге мы получаем 'Глубинную Карту' — визуальное представление эволюции рынка по ценовым уровням (Ask/Bid) и исполненным сделкам (Trade Tape) за определенный период.

Михаил Шардин [ https://shardin.name/ ], 
9 декабря 2025 года.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from pathlib import Path
import warnings

# Увеличиваем лимит пикселей для Matplotlib для больших изображений
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

warnings.filterwarnings('ignore')

class OrderBookVisualizer:
    def __init__(self, npz_path, output_dir='output_images'):
        self.npz_path = npz_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Загрузка {npz_path}...")
        self.data = np.load(npz_path)
        
        self.ts = self.data['ts']
        self.ask_prices = self.data['A']
        self.bid_prices = self.data['B']
        self.ask_volumes = self.data['vA']
        self.bid_volumes = self.data['vB']
        self.trades = self.data['T']
        
        self.ticker = Path(npz_path).stem
        
        # --- DEBUG LOGGING ---
        print(f"[DEBUG Init] {self.ticker}:")
        print(f"  - TS shape: {self.ts.shape}, dtype: {self.ts.dtype}")
        
        if self.trades is not None:
            print(f"  - Trades shape: {self.trades.shape}, dtype: {self.trades.dtype}")
        
        if len(self.ts) > 0:
             ts0 = self.ts[0]
             print(f"  - Start TS: {ts0}")
        # ---------------------
        
        self.cmap = LinearSegmentedColormap.from_list(
            'rainbow_liquidity', 
            ['#000033', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000', '#550000'],
            N=256
        )
        self.cmap.set_bad(color='black')
        
        print(f"Загружено: {self.ticker}, Точек: {len(self.ts):,}")

    def _safe_to_datetime(self, ts_val):
        """Безопасная конвертация времени"""
        if isinstance(ts_val, (np.datetime64, np.timedelta64)):
            return ts_val.item()
        return datetime.fromtimestamp(ts_val)

    def _extract_trades_data(self):
        """Универсальное извлечение данных о сделках"""
        if self.trades is None or len(self.trades) == 0:
            return None, None, None
            
        try:
            # Вариант 1: Обычная 2D матрица (N, 3)
            if self.trades.ndim == 2 and self.trades.shape[1] >= 3:
                return self.trades[:, 0], self.trades[:, 1], self.trades[:, 2]
            
            # Вариант 2: Структурированный массив
            if self.trades.dtype.names:
                names = self.trades.dtype.names
                return self.trades[names[0]], self.trades[names[1]], self.trades[names[2]]
            
            print("  [Warning] Неизвестный формат массива Trades. Сделки пропущены.")
            return None, None, None
            
        except Exception as e:
            print(f"  [Error] Ошибка разбора сделок: {e}")
            return None, None, None

    def create_combined_depth_visualization(self, start_idx=0, end_idx=None, 
                                           height_inches=30, dpi=300):
        if end_idx is None: end_idx = len(self.ts)
        raw_length = end_idx - start_idx
        
        # === НАСТРОЙКИ РАЗРЕШЕНИЯ (ULTRA HD) ===
        TARGET_WIDTH_PX = 12000      
        PRICE_LEVELS_TOP = 1200      
        PRICE_LEVELS_BOTTOM = 500    
        
        use_binning = raw_length > TARGET_WIDTH_PX
        n_bins = TARGET_WIDTH_PX if use_binning else raw_length
        
        bin_edges = np.linspace(start_idx, end_idx, n_bins + 1).astype(int)
        
        # --- ПОДГОТОВКА ДАННЫХ ---
        step = max(1, raw_length // 50000) 
        sample_prices = np.concatenate([
            self.ask_prices[:, start_idx:end_idx:step].flatten(),
            self.bid_prices[:, start_idx:end_idx:step].flatten()
        ])
        sample_prices = sample_prices[~np.isnan(sample_prices) & (sample_prices > 0)]
        
        if len(sample_prices) > 0:
            min_price = np.percentile(sample_prices, 0.5)
            max_price = np.percentile(sample_prices, 99.5)
        else:
            min_price = 0
            max_price = 100
        
        price_grid_top = np.linspace(min_price, max_price, PRICE_LEVELS_TOP)
        rel_range_pct = 0.5 
        rel_grid_bottom = np.linspace(-rel_range_pct, rel_range_pct, PRICE_LEVELS_BOTTOM)
        
        matrix_top = np.zeros((PRICE_LEVELS_TOP, n_bins))
        matrix_bottom = np.zeros((PRICE_LEVELS_BOTTOM, n_bins))
        
        bin_mid_prices = np.zeros(n_bins)
        bin_timestamps = []
        
        # Заполнение матриц
        for i in range(n_bins):
            b_start = bin_edges[i]
            b_end = bin_edges[i+1]
            if b_start == b_end: continue
            
            slice_ask_p = self.ask_prices[:, b_start:b_end].flatten()
            slice_ask_v = self.ask_volumes[:, b_start:b_end].flatten()
            slice_bid_p = self.bid_prices[:, b_start:b_end].flatten()
            slice_bid_v = self.bid_volumes[:, b_start:b_end].flatten()
            
            best_asks = self.ask_prices[0, b_start:b_end]
            best_bids = self.bid_prices[0, b_start:b_end]
            valid_ba = best_asks[~np.isnan(best_asks)]
            valid_bb = best_bids[~np.isnan(best_bids)]
            
            if len(valid_ba) == 0 or len(valid_bb) == 0:
                mid_price = bin_mid_prices[i-1] if i > 0 else (min_price + max_price)/2
            else:
                mid_price = (np.median(valid_ba) + np.median(valid_bb)) / 2
            
            bin_mid_prices[i] = mid_price
            
            ts_idx = (b_start + b_end) // 2
            ts_val = self.ts[ts_idx]
            bin_timestamps.append(ts_val)

            p_chunk = np.concatenate([slice_ask_p, slice_bid_p])
            v_chunk = np.concatenate([slice_ask_v, slice_bid_v])
            
            mask_top = (~np.isnan(p_chunk)) & (v_chunk > 0) & (p_chunk >= min_price) & (p_chunk <= max_price)
            p_valid = p_chunk[mask_top]
            v_valid = v_chunk[mask_top]
            
            if len(p_valid) > 0:
                idx_top = np.searchsorted(price_grid_top, p_valid)
                idx_top = np.clip(idx_top, 0, PRICE_LEVELS_TOP - 1)
                np.add.at(matrix_top[:, i], idx_top, v_valid)
            
            if mid_price > 0:
                p_rel_pct = ((p_valid - mid_price) / mid_price) * 100
                mask_btm = (p_rel_pct >= -rel_range_pct) & (p_rel_pct <= rel_range_pct)
                p_rel_valid = p_rel_pct[mask_btm]
                v_rel_valid = v_valid[mask_btm]
                
                if len(p_rel_valid) > 0:
                    idx_btm = np.searchsorted(rel_grid_bottom, p_rel_valid)
                    idx_btm = np.clip(idx_btm, 0, PRICE_LEVELS_BOTTOM - 1)
                    np.add.at(matrix_bottom[:, i], idx_btm, v_rel_valid)

        matrix_top_log = np.log1p(matrix_top)
        matrix_bottom_log = np.log1p(matrix_bottom)
        
        # --- ОТРИСОВКА (LARGE SCALE) ---
        width_inches = max(50, n_bins / 200) 
        
        fig = plt.figure(figsize=(width_inches, height_inches), facecolor='#0a0a0a')
        
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.02)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        x_indices = np.arange(n_bins)
        
        # 1. HEATMAP ВЕРХНЯЯ
        ax1.pcolormesh(x_indices, price_grid_top, matrix_top_log, 
                       cmap=self.cmap, shading='nearest', rasterized=True)
        
        # 2. ТРЕЙДЫ
        tr_ts_all, tr_px_all, tr_sz_all = self._extract_trades_data()
        
        if tr_ts_all is not None:
            t_start_val = self.ts[start_idx]
            t_end_val = self.ts[end_idx-1]
            
            ts_is_dt = np.issubdtype(self.ts.dtype, np.datetime64)
            tr_is_dt = np.issubdtype(tr_ts_all.dtype, np.datetime64)
            
            if ts_is_dt and not tr_is_dt:
                try:
                   t_start_cmp = t_start_val.astype('float64')
                   t_end_cmp = t_end_val.astype('float64')
                except:
                   t_start_cmp = t_start_val
                   t_end_cmp = t_end_val
            else:
                t_start_cmp = t_start_val
                t_end_cmp = t_end_val

            try:
                t_mask = (tr_ts_all >= t_start_cmp) & (tr_ts_all <= t_end_cmp)
                
                tr_ts = tr_ts_all[t_mask]
                tr_px = tr_px_all[t_mask]
                tr_sz = tr_sz_all[t_mask]
                
                if len(tr_ts) > 0:
                    search_ts = self.ts
                    query_ts = tr_ts
                    
                    if ts_is_dt and not np.issubdtype(query_ts.dtype, np.datetime64):
                         query_ts = query_ts.astype(self.ts.dtype)
                    
                    orig_indices = np.searchsorted(search_ts, query_ts)
                    tr_x_coords = np.interp(orig_indices, bin_edges, np.arange(len(bin_edges)))
                    
                    # Увеличиваем размер точек
                    sizes = np.clip(np.log1p(np.abs(tr_sz)) * 5, 5, 80)
                    
                    # === ИЗМЕНЕНИЕ ЗДЕСЬ ===
                    # Тонкая линия (0.6), чуть прозрачная (0.7)
                    ax1.plot(tr_x_coords, tr_px, color='white', linewidth=0.6, alpha=0.7)
                    
                    # Точки сделок
                    ax1.scatter(tr_x_coords, tr_px, s=sizes, 
                               c='white', alpha=0.8, edgecolors='black', linewidth=0.5)
            except Exception as e:
                print(f"  [Error] Ошибка отрисовки трейдов: {e}")

        # 3. HEATMAP НИЖНЯЯ
        ax2.pcolormesh(x_indices, rel_grid_bottom, matrix_bottom_log,
                       cmap=self.cmap, shading='nearest', rasterized=True)
        
        # --- ОФОРМЛЕНИЕ ---
        def format_date(x, pos):
            idx = int(np.clip(x, 0, len(bin_timestamps)-1))
            ts_val = bin_timestamps[idx]
            dt = self._safe_to_datetime(ts_val)
            return dt.strftime('%H:%M\n%d.%m')

        ax2.xaxis.set_major_formatter(FuncFormatter(format_date))
        ax2.xaxis.set_major_locator(plt.MaxNLocator(30))
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('black')
            ax.tick_params(axis='x', colors='#a0a0a0', labelsize=14)
            ax.tick_params(axis='y', colors='#a0a0a0', labelsize=14)
            ax.grid(True, color='#333333', linestyle='--', alpha=0.3)
            for spine in ax.spines.values():
                spine.set_color('#444444')

        ax1.set_ylabel('Price', color='#00ff88', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Offset %', color='#00ff88', fontsize=16, fontweight='bold')
        ax2.axhline(0, color='white', linestyle='-', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        
        # --- ИМЯ ФАЙЛА ---
        chunk_start_ts = self.ts[start_idx]
        dt_actual = self._safe_to_datetime(chunk_start_ts)
        
        dt_monday = dt_actual - timedelta(days=dt_actual.weekday())
        dt_sunday = dt_monday + timedelta(days=6)
        
        date_fmt = "%Y%m%d"
        filename = f"{self.ticker}_{dt_monday.strftime(date_fmt)}_{dt_sunday.strftime(date_fmt)}.png"
        
        filepath = self.output_dir / filename
        
        if filepath.exists():
            filename = f"{self.ticker}_{dt_monday.strftime(date_fmt)}_{dt_sunday.strftime(date_fmt)}_{start_idx}.png"
            filepath = self.output_dir / filename
        
        print(f"Сохранение High-Res ({width_inches:.1f}x{height_inches} in): {filepath}")
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close(fig)
        
        return filepath

    def generate_all_visualizations(self, time_windows=None):
        if time_windows is None:
            total_time = len(self.ts)
            time_windows = [(0, total_time, 'full_period')]
            
        results = []
        for start, end, label in time_windows:
            try:
                path = self.create_combined_depth_visualization(start, end)
                results.append({'label': label, 'path': str(path)})
            except Exception as e:
                print(f"ERROR in {label}: {e}")
                import traceback
                traceback.print_exc()
        return results