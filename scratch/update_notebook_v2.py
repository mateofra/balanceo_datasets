import json
import os

notebook_path = "ejecutar_flujo_datasets.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = [
    {
        "cell_type": "markdown",
        "id": "labeling_header",
        "metadata": {},
        "source": [
            "## 11. Etiquetado Asistido MANO (MVP)\n",
            "Este paso aplica heurísticas geométricas y clustering para auto-sugerir etiquetas a las 10.000 muestras MANO."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "labeling_step_ab",
        "metadata": {},
        "outputs": [],
        "source": [
            "log_step(\"Paso A/B: Auto-sugerencia y Clustering\")\n",
            "run_cmd([\n",
            "    \"uv\", \"run\", \"python\", \"src/labeling/assisted_labeling_mano.py\"\n",
            "], cwd=ROOT_DIR)"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "labeling_ui_header",
        "metadata": {},
        "source": [
            "### Interfaz de Validación por Lotes (Paso C)\n",
            "Usa el dropdown para seleccionar un cluster y validar las etiquetas en bloque."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "labeling_ui_launch",
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.labeling.validation_ui import launch_ui\n",
            "launch_ui(csv_path=\"output/mano_assisted_labeling_step_ab.csv\", \n",
            "          samples_dir=CONFIG[\"MANO_SAMPLES_DIR\"])"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "labeling_prop_header",
        "metadata": {},
        "source": [
            "### Propagación de Etiquetas (Paso D)\n",
            "Una vez validadas, propaga las etiquetas a los manifiestos finales."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "labeling_prop_run",
        "metadata": {},
        "outputs": [],
        "source": [
            "log_step(\"Paso D: Propagación de Etiquetas\")\n",
            "run_cmd([\n",
            "    \"uv\", \"run\", \"python\", \"src/labeling/propagate_labels.py\"\n",
            "], cwd=ROOT_DIR)"
        ]
    }
]

# Find where to insert (after Step 6 - Secuencias MANO)
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if 'source' in cell and any('Paso 6: Generacion de secuencias temporales MANO' in line for line in cell['source']):
        # Found Step 6 heading. We want to insert after its visualization.
        for j in range(i+1, len(nb['cells'])):
            if 'visualize_step(\"Paso 6' in "".join(nb['cells'][j].get('source', [])):
                insert_idx = j + 1
                break
        if insert_idx != -1:
            break

if insert_idx != -1:
    print(f"Inserting {len(new_cells)} cells at index {insert_idx}")
    nb['cells'] = nb['cells'][:insert_idx] + new_cells + nb['cells'][insert_idx:]
    
    # Update subsequent headings
    current_step = 12
    for cell in nb['cells'][insert_idx + len(new_cells):]:
        if cell['cell_type'] == 'markdown' and cell['source'] and cell['source'][0].startswith('## '):
            parts = cell['source'][0].split('.')
            if len(parts) > 1 and parts[0].replace('## ', '').strip().isdigit():
                cell['source'][0] = f"## {current_step}. " + ".".join(parts[1:]).strip() + "\n"
                current_step += 1

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated successfully.")
else:
    print("Could not find insertion point.")
