import pandas as pd
import numpy as np
import json
from pathlib import Path

def create_visual_tool():
    csv_path = Path("output/mano_refined_suggestions.csv")
    if not csv_path.exists():
        print("❌ Error: No se encuentra el CSV de sugerencias.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Tomar una submuestra o los primeros N para que el HTML no pese GBs
    # Priorizar aquellos con confianza media/baja
    df_sample = df.sort_values('confidence').head(500).copy()
    
    # Cargar landmarks para cada muestra
    data = []
    print(f"📦 Empaquetando {len(df_sample)} muestras en el HTML...")
    
    for idx, row in df_sample.iterrows():
        try:
            lms = np.load(row['path_landmarks']).tolist()
            data.append({
                'id': row['sample_id'],
                'label': row['suggested_label'],
                'confidence': float(row['confidence']),
                'landmarks': lms
            })
        except:
            continue
            
    # Template HTML
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Validador de Etiquetas MANO</title>
    <style>
        body { font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 15px; }
        .card { background: #2a2a2a; border-radius: 8px; padding: 10px; border: 1px solid #444; transition: 0.2s; }
        .card:hover { border-color: #3498db; }
        .card.selected { border-color: #f1c40f; box-shadow: 0 0 10px rgba(241, 196, 15, 0.5); }
        canvas { background: #000; width: 100%; height: 160px; border-radius: 4px; }
        .info { font-size: 12px; margin-top: 5px; }
        .label-tag { background: #3498db; padding: 2px 5px; border-radius: 3px; font-weight: bold; }
        .controls { position: sticky; top: 0; background: #1a1a1a; padding: 15px 0; z-index: 100; border-bottom: 1px solid #444; margin-bottom: 20px; }
        button { padding: 8px 15px; border-radius: 4px; border: none; cursor: pointer; margin-right: 5px; }
        .btn-ok { background: #27ae60; color: white; }
        .btn-change { background: #e67e22; color: white; }
        .btn-amb { background: #c0392b; color: white; }
        .export-btn { background: #9b59b6; color: white; float: right; }
    </style>
</head>
<body>
    <div class="controls">
        <h2>Visualizador de Sugerencias MANO</h2>
        <span>Filtro: <input type="text" id="filter" placeholder="Gesto..."></span>
        <button class="export-btn" onclick="exportCSV()">Exportar Correcciones CSV</button>
        <p style="font-size: 13px; color: #aaa;">Usa click para seleccionar. Teclas: [A]ceptar, [U]nknown, [1-9] Cambiar Etiqueta.</p>
    </div>
    
    <div class="grid" id="main-grid"></div>

    <script>
        const SAMPLES = """ + json.dumps(data) + """;
        const CONNECTIONS = [
            [0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20],
            [5,9], [9,13], [13,17]
        ];

        function drawHand(ctx, lms) {
            const w = ctx.canvas.width;
            const h = ctx.canvas.height;
            ctx.clearRect(0, 0, w, h);
            ctx.strokeStyle = "#00ff00";
            ctx.lineWidth = 2;
            
            // Centrar y escalar
            let minX = Math.min(...lms.map(p => p[0])), maxX = Math.max(...lms.map(p => p[0]));
            let minY = Math.min(...lms.map(p => p[1])), maxY = Math.max(...lms.map(p => p[1]));
            
            const scale = Math.min(w/(maxX-minX+0.1), h/(maxY-minY+0.1)) * 0.8;
            const offsetX = w/2 - (maxX+minX)/2 * scale;
            const offsetY = h/2 - (maxY+minY)/2 * scale;

            CONNECTIONS.forEach(chain => {
                ctx.beginPath();
                for(let i=0; i<chain.length; i++) {
                    let p = lms[chain[i]];
                    let x = p[0] * scale + offsetX;
                    let y = p[1] * scale + offsetY;
                    if(i===0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.stroke();
            });

            lms.forEach(p => {
                ctx.fillStyle = "#3498db";
                ctx.beginPath();
                ctx.arc(p[0]*scale+offsetX, p[1]*scale+offsetY, 2, 0, Math.PI*2);
                ctx.fill();
            });
        }

        function render() {
            const grid = document.getElementById('main-grid');
            grid.innerHTML = '';
            SAMPLES.forEach((s, idx) => {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = `
                    <canvas id="cvs-${idx}"></canvas>
                    <div class="info">
                        <b>${s.id.split('_')[1]}</b> | <span class="label-tag">${s.label}</span><br>
                        Conf: ${s.confidence.toFixed(2)}
                    </div>
                `;
                grid.appendChild(card);
                const cvs = document.getElementById(`cvs-${idx}`);
                cvs.width = 180; cvs.height = 160;
                drawHand(cvs.getContext('2d'), s.landmarks);
            });
        }

        function exportCSV() {
            let csv = "sample_id,final_label\\n";
            SAMPLES.forEach(s => { csv += `${s.id},${s.label}\\n`; });
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = 'mano_validated_labels.csv'; a.click();
        }

        window.onload = render;
    </script>
</body>
</html>
    """
    
    output_path = Path("output/validador_etiquetas.html")
    with open(output_path, "w") as f:
        f.write(html_template)
        
    print(f"✨ Herramienta visual generada en {output_path}")

if __name__ == "__main__":
    create_visual_tool()
