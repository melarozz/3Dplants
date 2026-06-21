# 3Dplants

Программный комплекс для количественного фенотипирования 3D-реконструкций проростков кукурузы: подготовка и разметка, CPD-регистрация, трекинг роста органов, моделирование освещённости и морфометрия.

## Возможности

| Этап | Команда | Описание                                   |
|------|---------|--------------------------------------------|
| E0 | `plant3d prep-images` | Удаление фона и центрирование фото         |
| E1 | `plant3d generate-3d` | 3D-реконструкция                           |
| E2 | `plant3d prep-seg` / `post-seg` / `segment-session` | Подготовка PCD, разметка в SSE, сборка PLY |
| E3 | `plant3d track-original` | Регистрация временных состояний            |
| E5 | `plant3d track-growth` | Поля смещений органов                      |
| E6 | `plant3d light` | Расчёт освещенности                        |
| E7 | `plant3d morph` | Морфометрические признаки листьев          |

## Требования

- Python **3.10+**
- Linux / macOS / Windows
- [Semantic Segmentation Editor](https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor) — только для ручной разметки (`segment-session`, порт **3000**)
- Node.js + Meteor — для запуска SSE

## Установка

```bash
git clone https://github.com/melarozz/3Dplants.git
cd 3Dplants
./scripts/build.sh
```

## Быстрый старт

```bash
source venv/bin/activate

# Регистрация двух моделей 
plant3d track-original scripts/data/plant1.glb scripts/data/plant2.glb -o outputs/cpd

# Освещённость
plant3d light scripts/data/mesh-19.glb -o outputs/light

# Подготовка к разметке
plant3d prep-seg scripts/data/mesh-19.glb -o outputs/prep

# Морфометрия (нужен размеченный PLY)
plant3d morph outputs/post/labeled.ply -o outputs/morph
```

Каждый запуск сохраняет `run_manifest.json`

## Переменные окружения

| Переменная | Назначение |
|------------|------------|
| `PLANT3D_SSE_DIR` | Путь к Semantic Segmentation Editor |
| `PLANT3D_HUNYUAN_CMD` | Команда внешнего скрипта Hunyuan3D для `generate-3d` |

Пример SSE:

```bash
export PLANT3D_SSE_DIR=/path/to/semantic-segmentation-editor
plant3d segment-session model.stl -o outputs/seg --editor-dir "$PLANT3D_SSE_DIR"
```

## Архитектура

Единый пакет `plant_3d` — три слоя. Алгоритмы CPD, освещённости и эвристической сегментации живут в **infrastructure**.

```
src/plant_3d/
  interfaces/        # CLI 
  application/       # use-cases, порты, типы, профили, манифесты
  infrastructure/    # реализации этапов E0–E7
    cpd/             # CPD-регистрация
    light/           # освещение
    segmentation/    # эвристическая сегментация leaf/stem
    seg_prep/        # подготовка к SSE
    growth_tracking/
    morphometrics/
    io/
    cpd_bridge.py
    light_bridge.py
    ...
```

## Лицензия

MIT — см. [LICENSE](LICENSE).
