"""
# Описание Файла: Главный Пакетный Обработчик (Batch Processor & Orchestrator)

Назначение
ГЛАВНЫЙ ИСПОЛНЯЕМЫЙ МОДУЛЬ.
Это "дирижер" всего процесса. Он управляет последовательностью операций для всех рыночных инструментов (тикеров), для которых есть файлы `.npz`.

Последовательность Работы
1.  Поиск и Разбивка: Сканирует папку с файлами `*.npz`.
2.  Логика Времени (Quant Logic): Автоматически разбивает каждый длинный временной ряд на управляемые недельные интервалы. Это позволяет избежать создания слишком больших изображений и упрощает навигацию по данным.
3.  Многопоточность: Использует несколько ядер процессора (`ProcessPoolExecutor`) для одновременной обработки разных тикеров, значительно ускоряя процесс.
4.  Выполнение: Последовательно вызывает:
    *   `orderbook_visualizer.py` для создания высокодетализированных PNG-изображений (одна неделя = одно изображение).
    *   `deepzoom_converter.py` для преобразования этих изображений в наборы тайлов Deep Zoom.
5.  Интерфейс: Генерирует финальный интерактивный HTML-просмотрщик, который позволяет удобно анализировать все сгенерированные тепловые карты в браузере.

Михаил Шардин [ https://shardin.name/ ], 
9 декабря 2025 года.

"""

import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import numpy as np

class BatchProcessor:
    """
    Пакетный процессор для множества .npz файлов
    """
    
    def __init__(self, npz_directory='npz_data', 
                 output_images_dir='output_images',
                 deepzoom_dir='deepzoom_output',
                 n_workers=None):
        self.npz_dir = Path(npz_directory)
        self.output_images_dir = Path(output_images_dir)
        self.deepzoom_dir = Path(deepzoom_dir)
        
        self.output_images_dir.mkdir(exist_ok=True, parents=True)
        self.deepzoom_dir.mkdir(exist_ok=True, parents=True)
        
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 2)
        self.n_workers = n_workers
        
        print(f"Пакетный процессор инициализирован")
        print(f"Воркеров: {self.n_workers}")
    
    def find_npz_files(self):
        npz_files = list(self.npz_dir.glob('*.npz'))
        print(f"\nНайдено .npz файлов: {len(npz_files)}")
        for f in npz_files:
            print(f"  - {f.name}")
        return npz_files
    
    def _safe_to_datetime(self, ts_val):
        """Конвертирует numpy scalar или float timestamp в python datetime"""
        if isinstance(ts_val, (np.datetime64, np.timedelta64)):
            # Конвертация numpy datetime64 -> python datetime
            return ts_val.item()
        else:
            # Unix timestamp (float/int)
            return datetime.fromtimestamp(ts_val)

    def process_single_ticker(self, npz_path, time_windows=None):
        try:
            from orderbook_visualizer import OrderBookVisualizer
            
            ticker = Path(npz_path).stem
            print(f"\n{'='*60}")
            print(f"Обработка тикера: {ticker}")
            print(f"{'='*60}")
            
            ticker_output_dir = self.output_images_dir / ticker
            ticker_output_dir.mkdir(exist_ok=True, parents=True)
            
            viz = OrderBookVisualizer(npz_path, output_dir=str(ticker_output_dir))
            
            # --- DEBUG INFO ---
            timestamps = viz.ts
            if len(timestamps) > 0:
                print(f"[DEBUG] TS Type: {type(timestamps)}")
                print(f"[DEBUG] TS Dtype: {timestamps.dtype}")
                print(f"[DEBUG] TS First: {timestamps[0]}")
                print(f"[DEBUG] TS Last:  {timestamps[-1]}")
            else:
                print(f"[DEBUG] TS Array is EMPTY!")
                return {'status': 'empty_data'}
            # ------------------

            # --- ЛОГИКА РАЗБИВКИ ПО НЕДЕЛЯМ ---
            if time_windows is None:
                time_windows = []
                total_len = len(timestamps)
                
                # Проверяем, является ли массив datetime64
                is_np_dt = np.issubdtype(timestamps.dtype, np.datetime64)
                
                if total_len > 0:
                    # 1. Получаем дату начала как Python datetime
                    dt_first = self._safe_to_datetime(timestamps[0])
                    dt_last = self._safe_to_datetime(timestamps[-1])
                    
                    print(f"[DEBUG] Start Date: {dt_first}")
                    print(f"[DEBUG] End Date:   {dt_last}")

                    # 2. Откатываемся к понедельнику (00:00:00)
                    start_of_week = dt_first - timedelta(days=dt_first.weekday())
                    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
                    
                    current_week_start_dt = start_of_week
                    week_seconds = 7 * 24 * 3600
                    
                    print("Генерация недельных интервалов...")
                    
                    while current_week_start_dt < dt_last:
                        next_week_start_dt = current_week_start_dt + timedelta(days=7)
                        
                        # Подготовка значений для поиска (searchsorted)
                        if is_np_dt:
                            # Если массив datetime64, ищем datetime64
                            search_start = np.datetime64(current_week_start_dt)
                            search_end = np.datetime64(next_week_start_dt)
                        else:
                            # Если массив float, ищем timestamp float
                            search_start = current_week_start_dt.timestamp()
                            search_end = next_week_start_dt.timestamp()
                        
                        # Ищем индексы
                        idx_start = np.searchsorted(timestamps, search_start)
                        idx_end = np.searchsorted(timestamps, search_end)
                        
                        # Если есть данные в этой неделе
                        if idx_end > idx_start:
                            idx_end = min(idx_end, total_len)
                            
                            # Для имени файла
                            w_label = f"Week_{current_week_start_dt.strftime('%Y%m%d')}"
                            time_windows.append((idx_start, idx_end, w_label))
                        
                        current_week_start_dt = next_week_start_dt
            
            print(f"Сформировано {len(time_windows)} недельных интервалов.")
            
            results = viz.generate_all_visualizations(time_windows=time_windows)
            
            return {
                'ticker': ticker,
                'status': 'success',
                'npz_path': str(npz_path),
                'output_dir': str(ticker_output_dir),
                'visualizations': len(results),
                'results': results
            }
            
        except Exception as e:
            print(f"ОШИБКА при обработке {npz_path}: {e}")
            traceback.print_exc()
            return {
                'ticker': Path(npz_path).stem,
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def process_all_tickers(self, sequential=False):
        npz_files = self.find_npz_files()
        
        if not npz_files:
            print("Не найдено .npz файлов для обработки")
            return []
        
        all_results = []
        
        print(f"\n{'='*60}")
        print(f"Начало обработки {len(npz_files)} тикеров")
        
        start_time = datetime.now()
        
        if sequential:
            for npz_file in npz_files:
                result = self.process_single_ticker(npz_file)
                all_results.append(result)
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(self.process_single_ticker, npz_file): npz_file
                    for npz_file in npz_files
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        all_results.append({'status': 'error', 'error': str(e)})
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            'processed_at': start_time.isoformat(),
            'duration_seconds': duration,
            'total_tickers': len(npz_files),
            'results': all_results
        }
        
        summary_path = self.output_images_dir / 'processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"ЗАВЕРШЕНО! Время: {duration:.1f} сек")
        print(f"{'='*60}\n")
        
        return all_results
    
    def convert_to_deepzoom(self):
        from deepzoom_converter import DeepZoomConverter
        
        print(f"\n{'='*60}")
        print(f"Конвертация в Deep Zoom формат")
        print(f"{'='*60}\n")
        
        ticker_dirs = [d for d in self.output_images_dir.iterdir() if d.is_dir()]
        all_image_info = []
        
        for ticker_dir in ticker_dirs:
            ticker = ticker_dir.name
            if ticker in ['deepzoom_output']: continue
            
            print(f"\nКонвертация тикера: {ticker}")
            
            ticker_deepzoom_dir = self.deepzoom_dir / ticker
            ticker_deepzoom_dir.mkdir(exist_ok=True, parents=True)
            
            converter = DeepZoomConverter(
                input_dir=str(ticker_dir),
                output_dir=str(ticker_deepzoom_dir)
            )
            
            results = converter.convert_all_images(pattern='*.png')
            
            for result in results:
                if result.get('status') == 'success':
                    result['ticker'] = ticker
                    all_image_info.append(result)
        
        if all_image_info:
            tickers_data = {}
            for info in all_image_info:
                ticker = info['ticker']
                if ticker not in tickers_data:
                    tickers_data[ticker] = []
                
                tickers_data[ticker].append({
                    'name': f"{ticker}/{info['name']}",
                    'dzi': f"{ticker}/{info['name']}.dzi"
                })
            
            self.create_main_viewer(tickers_data)
        
        print(f"\n{'='*60}")
        print(f"Deep Zoom конвертация завершена")
        print(f"{'='*60}")
        
        return all_image_info
    
    def create_main_viewer(self, tickers_data):
        html_path = self.deepzoom_dir / 'index.html'
        
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Order Book Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>
    <style>
        body { margin: 0; background: #0a0a0a; color: #fff; font-family: sans-serif; overflow: hidden; }
        #container { display: flex; height: 100vh; }
        #sidebar { width: 300px; background: #1a1a1a; border-right: 1px solid #333; overflow-y: auto; padding: 15px; }
        #viewer-container { flex: 1; position: relative; background: #000; }
        #openseadragon { width: 100%; height: 100%; }
        h1 { color: #00ff88; font-size: 20px; text-align: center; }
        .item { padding: 8px; cursor: pointer; border-bottom: 1px solid #333; font-size: 13px; }
        .item:hover { background: #333; color: #00ff88; }
        .item.active { background: #2a4a2a; color: #00ff88; border-left: 3px solid #00ff88; }
        .group-header { padding: 10px; background: #252525; color: #00ccff; font-weight: bold; margin-top: 10px; }
    </style>
</head>
<body>
    <div id="container">
        <div id="sidebar">
            <h1>📊 Market Depth</h1>
            <div id="list"></div>
        </div>
        <div id="viewer-container">
            <div id="openseadragon"></div>
        </div>
    </div>
    <script>
        const data = """ + json.dumps(tickers_data) + """;
        const viewer = OpenSeadragon({
            id: "openseadragon",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
            animationTime: 0.5,
            blendTime: 0.1,
            maxZoomPixelRatio: 5,
            minZoomLevel: 0.1,
            visibilityRatio: 0.5,
            zoomPerScroll: 1.4
        });

        const listDiv = document.getElementById('list');
        
        Object.keys(data).forEach(ticker => {
            const header = document.createElement('div');
            header.className = 'group-header';
            header.textContent = ticker;
            listDiv.appendChild(header);
            
            data[ticker].sort((a, b) => a.name.localeCompare(b.name));

            data[ticker].forEach(img => {
                const item = document.createElement('div');
                item.className = 'item';
                item.textContent = img.name.split('/').pop().replace(ticker + '_', '').replace('.dzi', '');
                item.onclick = function() {
                    document.querySelectorAll('.item').forEach(i => i.classList.remove('active'));
                    item.classList.add('active');
                    viewer.open(img.dzi);
                };
                listDiv.appendChild(item);
            });
        });
    </script>
</body>
</html>"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✓ HTML viewer создан: {html_path}")

def main():
    processor = BatchProcessor(n_workers=4)
    print("\n📊 ШАГ 1: Генерация (разбивка по неделям)...")
    processor.process_all_tickers(sequential=True)
    print("\n🔍 ШАГ 2: Deep Zoom...")
    processor.convert_to_deepzoom()
    print("\n✅ ГОТОВО! Запустите локальный сервер.")

if __name__ == '__main__':
    main()