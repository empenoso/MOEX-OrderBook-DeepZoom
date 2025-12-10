"""
# Описание Файла: Конвертер Deep Zoom (Tiling Engine)

Назначение
ВСПОМОГАТЕЛЬНЫЙ МОДУЛЬ.
Сгенерированные тепловые карты ликвидности в `orderbook_visualizer.py` имеют сверхвысокое разрешение (Ultra-HD), что делает их непригодными для быстрого просмотра в интернете. Этот конвертер решает эту проблему.

Метод (Deep Zoom)
Он разбивает огромное изображение на тысячи мелких "тайлов" (плиток) и создает специальный файл-манифест (`.dzi`). Это позволяет интерактивно просматривать огромное изображение в браузере (используя библиотеку OpenSeadragon, как в Google Maps) — быстро масштабировать и перемещаться без необходимости загружать весь файл целиком.

Кратко: обеспечивает практическую возможность анализа высокодетализированных временных рядов без ущерба для скорости загрузки.

Михаил Шардин [ https://shardin.name/ ], 
9 декабря 2025 года.

"""

import os
import math
import shutil
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Снимаем защиту Pillow от "бомб декомпрессии"
Image.MAX_IMAGE_PIXELS = None

class DeepZoomConverter:
    """
    Конвертер изображений в Deep Zoom формат используя Pillow
    """
    
    def __init__(self, input_dir, output_dir='deepzoom_output', tile_size=256):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.overlap = 1 
        self.format = "jpg"
        self.quality = 85
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Deep Zoom конвертер (Pillow backend) инициализирован")
    
    def convert_image(self, image_path):
        """Конвертация одного изображения"""
        try:
            image_path = Path(image_path)
            image_name = image_path.stem
            
            print(f"\nКонвертация: {image_path.name}")
            
            dzi_path = self.output_dir / f"{image_name}.dzi"
            files_dir = self.output_dir / f"{image_name}_files"
            
            if files_dir.exists():
                shutil.rmtree(files_dir)
            files_dir.mkdir(parents=True)
            
            with Image.open(image_path) as img:
                # FIX: Конвертируем в RGB, так как JPEG не поддерживает прозрачность (RGBA)
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    img = img.convert('RGB')
                
                width, height = img.size
                print(f"  Размер исх: {width} x {height}")
                
                max_dim = max(width, height)
                num_levels = int(math.ceil(math.log(max_dim, 2))) + 1
                
                current_img = img
                
                for level in tqdm(range(num_levels - 1, -1, -1), desc="  Генерация уровней"):
                    level_dir = files_dir / str(level)
                    level_dir.mkdir(exist_ok=True)
                    
                    lvl_width, lvl_height = current_img.size
                    cols = int(math.ceil(lvl_width / self.tile_size))
                    rows = int(math.ceil(lvl_height / self.tile_size))
                    
                    for col in range(cols):
                        for row in range(rows):
                            x = col * self.tile_size
                            y = row * self.tile_size
                            
                            w = min(self.tile_size, lvl_width - x)
                            h = min(self.tile_size, lvl_height - y)
                            
                            tile = current_img.crop((x, y, x + w, y + h))
                            
                            tile_name = f"{col}_{row}.{self.format}"
                            # Save handles the RGB mode correctly now
                            tile.save(level_dir / tile_name, quality=self.quality)
                    
                    if level > 0:
                        new_w = max(1, lvl_width // 2)
                        new_h = max(1, lvl_height // 2)
                        current_img = current_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                self._write_dzi(dzi_path, width, height)
                
            print(f"  ✓ Готово: {dzi_path}")
            
            return {
                'status': 'success',
                'name': image_name,
                'original': str(image_path),
                'dzi': str(dzi_path),
                'width': width,
                'height': height
            }
            
        except Exception as e:
            print(f"  ✗ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'name': image_path.stem if image_path else 'unknown',
                'error': str(e)
            }
    
    def _write_dzi(self, path, width, height):
        content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="{self.format}"
       Overlap="0" 
       TileSize="{self.tile_size}" >
    <Size Height="{height}" Width="{width}"/>
</Image>"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    def convert_all_images(self, pattern='*.png'):
        image_files = list(self.input_dir.glob(pattern))
        if not image_files:
            print(f"Не найдено изображений с шаблоном {pattern}")
            return []
        
        print(f"\nНайдено изображений: {len(image_files)}")
        results = []
        for image_file in image_files:
            results.append(self.convert_image(image_file))
        return results